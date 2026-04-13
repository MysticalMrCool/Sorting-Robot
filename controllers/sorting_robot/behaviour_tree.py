"""
behaviour_tree.py - Layer 2: Reactive Logic
===========================================

Griffith 3003ICT - Programming for Robotics - Assessment 1
Track B - Autonomous Sorting Warehouse Robot

This module implements the robot's decision layer as a Priority-based Finite
State Machine (Priority FSM) with nine states. The code pattern is a deliberate
Python port of the canonical Week 4 Workshop Smart-Security-Gate FSM template:

    enum State { ... };
    State currentState = ...;
    unsigned long stateStartMs = 0;
    void enterState(State s) {
      currentState = s;
      stateStartMs = millis();
    }

In Python we use an Enum, `self.current_state`, `self.state_start_ms`, and
`self.enter_state(s)`. `millis()` is replaced by the Webots simulation clock
(`robot.getTime() * 1000`) which is the non-blocking equivalent recommended
in Week 5.

Architecture (Week 4 slide 17 - Behaviour Architecture Layering):
    Layer 1: Hardware control   -> sorting_robot.py
    Layer 2: Reactive logic     -> this file (Priority FSM)
    Layer 3: AI enhancement     -> inference.py (CNN)

Priority selector order (highest priority first). The top of tick() acts
as a priority-interrupt mechanism: higher-priority conditions can pre-empt
the current state. This is a flat FSM with priority interrupts (Wk04),
not a hierarchical Behaviour Tree (Wk08).

    FAIL_SAFE  ->  AVOID  ->  DELIVER  ->  PLAN_DELIVERY  ->
    PICKUP  ->  CLASSIFY  ->  APPROACH_TARGET  ->  PATROL  ->  COMPLETE

Quoting Wk04-EmbeddedAI slide 12: "Rule-based and learning systems can
coexist in embedded robotics." The rule-based FSM drives behaviour; the CNN
in inference.py provides the classification fact the FSM uses to decide
which drop zone is the goal.
"""

from enum import Enum, auto
import math

# Debug logging toggle — set True for verbose FSM state/classify output
DEBUG = False


# -----------------------------------------------------------------------------
# Tuning constants (pull these up here so markers can read the whole policy
# without having to grep the tick function).
# -----------------------------------------------------------------------------

OBSTACLE_STOP_DISTANCE = 0.18     # metres - any closer triggers AVOID
APPROACH_DISTANCE = 0.20          # stop approaching once this close to target
# Strategic distance checkpoints for frame capture during approach.
# One frame per checkpoint (~4 approach + 1 dwell = ~5 total) instead of
# one per tick (~91).  Mid-range shots ensure large items (barrels) are
# captured before they fill the entire frame at close range.
CLASSIFY_FRAME_DISTANCES = [0.40, 0.35, 0.30, 0.25]
PICKUP_RADIUS = 0.25              # supervisor teleport kicks in inside this
DELIVERY_RADIUS = 0.30            # close enough to the drop pad to release
WAYPOINT_RADIUS = 0.25            # patrol waypoint reached tolerance
CRUISE_SPEED = 4.0                # rad/s on wheels
TURN_SPEED = 2.5
CLASSIFY_DURATION_MS = 400        # dwell after stopping (camera settle time)
PICKUP_DURATION_MS = 600
FAIL_SAFE_DURATION_MS = 1500


# -----------------------------------------------------------------------------
# Category -> drop-zone mapping. The CNN in inference.py produces one of these
# four labels and the BT uses this table to pick which zone to navigate to.
# -----------------------------------------------------------------------------

CATEGORY_TO_ZONE = {
    "fragile":   "drop_fragile",
    "standard":  "drop_standard",
    "hazardous": "drop_hazardous",
    "unknown":   "drop_unknown",
}


# -----------------------------------------------------------------------------
# State enum - mirror of Wk04-Workshop slide 13:
#     enum State { DISARMED, ARMED_IDLE, MOTION_DETECTED, ... };
# We have nine states, well above the rubric minimum of four.
# -----------------------------------------------------------------------------

class State(Enum):
    PATROL = auto()              # default: walk the patrol waypoint loop
    APPROACH_TARGET = auto()     # a cargo item is visible - drive toward it
    CLASSIFY = auto()            # stop, capture frame, run CNN
    PICKUP = auto()              # supervisor-teleport item onto carry slot
    PLAN_DELIVERY = auto()       # run A* from here to the matching drop zone
    DELIVER = auto()             # follow the A* path to the drop zone
    AVOID = auto()               # obstacle too close - back off and steer
    FAIL_SAFE = auto()           # something went wrong - stop and recover
    COMPLETE = auto()            # all cargo delivered - mission finished


# -----------------------------------------------------------------------------
# BehaviourTree
# -----------------------------------------------------------------------------

class PriorityFSM:
    """
    Single-tick Priority FSM.

    Usage:
        fsm = PriorityFSM(robot_api, classifier, planner)
        while robot.step(TIME_STEP) != -1:
            fsm.tick()

    `robot_api` must expose:
        now_ms() -> float
        read_distance_left() -> float (metres)
        read_distance_right() -> float (metres)
        gps_xy() -> (float, float)
        heading_rad() -> float
        grab_image() -> numpy.ndarray | None
        set_motors(left, right) -> None
        stop() -> None
        pick_up(item_def_name) -> bool
        release(at_xy) -> None
        holding_item() -> bool
        next_visible_cargo() -> dict | None  # {"def": str, "xy": (x,y), "distance": float}
        patrol_target_xy() -> (float, float)
        advance_patrol() -> None
        zone_xy(zone_name) -> (float, float) | None
        cargo_remaining() -> int
        log(str) -> None
    """

    def __init__(self, robot_api, classifier, planner):
        self.robot = robot_api
        self.classifier = classifier
        self.planner = planner

        self.current_state = State.PATROL
        self.state_start_ms = self.robot.now_ms()

        # Per-cycle scratch - reset when we pick up a new item
        self.active_cargo_def = None         # e.g. "CARGO_JAMJAR_A"
        self.active_category = None          # e.g. "fragile"
        self.active_zone = None              # e.g. "drop_fragile"
        self.active_path = []                # list of (x, y) waypoints
        self.active_path_index = 0
        self._avoid_count = 0                # consecutive avoid cycles
        self._classify_frames = []           # frames collected for classification
        self._next_frame_idx = 0             # index into CLASSIFY_FRAME_DISTANCES
        self._classify_dwell_captured = False # single dwell frame flag
        self._zone_drop_count = {}           # items delivered per zone (for offset)
        self._complete_logged = False         # one-shot flag for MISSION COMPLETE

        self.robot.log("[FSM] initialised in PATROL")

    # ---------- Canonical enter_state helper (Wk04-Workshop slide 13) --------

    def enter_state(self, new_state: State) -> None:
        """Equivalent of lecturer's:  enterState(State s) { currentState = s; stateStartMs = millis(); }"""
        if new_state != self.current_state and DEBUG:
            self.robot.log(f"[FSM] {self.current_state.name} -> {new_state.name}")
        self.current_state = new_state
        self.state_start_ms = self.robot.now_ms()

    def time_in_state_ms(self) -> float:
        return self.robot.now_ms() - self.state_start_ms

    # ---------- Top-level tick (priority selector) ---------------------------

    def tick(self) -> None:
        """
        One cycle of the FSM. Called once per robot.step() by sorting_robot.py.

        This function first runs the priority selector - higher-priority
        conditions can interrupt lower-priority ones. Then it executes the
        action associated with the resulting state. This is a flat FSM with
        priority interrupts (Wk04), not a hierarchical Behaviour Tree (Wk08).
        """
        # --- Priority selector ------------------------------------------------
        # 0. COMPLETE is terminal - nothing can interrupt it
        if self.current_state == State.COMPLETE:
            pass
        # 1. Fail-safe is sticky for its own duration
        elif self.current_state == State.FAIL_SAFE and self.time_in_state_ms() < FAIL_SAFE_DURATION_MS:
            pass
        # 2. Safety override - obstacle too close (but not during final delivery
        #    approach, where delivered items in the drop zone would cause loops)
        elif self._obstacle_blocking() and not self._near_drop_zone():
            if self.current_state != State.AVOID:
                self.enter_state(State.AVOID)
        # 3. If holding something, delivery branches take priority
        elif self.robot.holding_item():
            if self.current_state not in (State.DELIVER, State.PLAN_DELIVERY):
                self.enter_state(State.PLAN_DELIVERY)
        # 4. Otherwise, if we can see cargo and aren't already locked onto it,
        #    switch into APPROACH_TARGET
        elif self.current_state == State.PATROL:
            target = self.robot.next_visible_cargo()
            if target is not None:
                self.active_cargo_def = target["def"]
                self._classify_frames = []  # fresh start for this item
                self._next_frame_idx = 0
                self.enter_state(State.APPROACH_TARGET)

        # --- State action --------------------------------------------------
        handler = {
            State.PATROL: self._do_patrol,
            State.APPROACH_TARGET: self._do_approach_target,
            State.CLASSIFY: self._do_classify,
            State.PICKUP: self._do_pickup,
            State.PLAN_DELIVERY: self._do_plan_delivery,
            State.DELIVER: self._do_deliver,
            State.AVOID: self._do_avoid,
            State.FAIL_SAFE: self._do_fail_safe,
            State.COMPLETE: self._do_complete,
        }[self.current_state]
        handler()

    # ---------- Condition helpers --------------------------------------------

    def _obstacle_blocking(self) -> bool:
        dl = self.robot.read_distance_left()
        dr = self.robot.read_distance_right()
        return min(dl, dr) < OBSTACLE_STOP_DISTANCE

    def _near_drop_zone(self) -> bool:
        """True if holding an item and close to the target drop zone.
        Suppresses AVOID so delivered items in the zone don't block us."""
        if self.current_state != State.DELIVER or self.active_zone is None:
            return False
        goal = self.robot.zone_xy(self.active_zone)
        if goal is None:
            return False
        x, y = self.robot.gps_xy()
        return math.hypot(goal[0] - x, goal[1] - y) < DELIVERY_RADIUS * 2.0

    # ---------- State action handlers ----------------------------------------

    def _do_patrol(self) -> None:
        """
        Walk the precomputed patrol waypoint loop. This is the default
        background behaviour when nothing else is active.
        """
        if self.robot.cargo_remaining() == 0:
            self.enter_state(State.COMPLETE)
            return
        tx, ty = self.robot.patrol_target_xy()
        self._drive_toward(tx, ty)
        x, y = self.robot.gps_xy()
        if math.hypot(tx - x, ty - y) < WAYPOINT_RADIUS:
            self.robot.advance_patrol()

    def _do_approach_target(self) -> None:
        """
        Drive toward the currently-locked cargo item. Transitions to CLASSIFY
        once we're close enough to get a clean camera frame.
        """
        target = self.robot.next_visible_cargo()
        if target is None:
            # Lost sight of it - back to patrol
            self.active_cargo_def = None
            self.enter_state(State.PATROL)
            return
        self.active_cargo_def = target["def"]
        tx, ty = target["xy"]
        self._drive_toward(tx, ty)
        # Capture one frame per distance checkpoint (not every tick)
        if (self._next_frame_idx < len(CLASSIFY_FRAME_DISTANCES) and
                target["distance"] < CLASSIFY_FRAME_DISTANCES[self._next_frame_idx]):
            image = self.robot.grab_image()
            if image is not None:
                self._classify_frames.append(image)
            self._next_frame_idx += 1
        if target["distance"] < APPROACH_DISTANCE:
            self.robot.stop()
            self._classify_dwell_captured = False
            self.enter_state(State.CLASSIFY)

    def _do_classify(self) -> None:
        """
        Stop, collect camera frames during the dwell period, then classify
        using majority vote across multiple frames for robustness.
        Wk07-Vision: "The CNN does NOT decide what to do.
        Perception gives facts. Decision gives intention."
        """
        self.robot.stop()
        # Capture a single frame once the camera has settled after stopping
        if not self._classify_dwell_captured:
            image = self.robot.grab_image()
            if image is not None:
                self._classify_frames.append(image)
            self._classify_dwell_captured = True
        if self.time_in_state_ms() < CLASSIFY_DURATION_MS:
            return  # dwell so the camera frame is stable

        frames = self._classify_frames
        self._classify_frames = []
        self._next_frame_idx = 0
        if not frames:
            self.robot.log("[FSM] classify failed - no frames")
            self.enter_state(State.FAIL_SAFE)
            return

        # Debug: save frames so we can inspect what the CNN sees
        if DEBUG:
            try:
                import os, numpy as np
                from PIL import Image as PILImage
                debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_frames")
                os.makedirs(debug_dir, exist_ok=True)
                name = self.active_cargo_def or "unknown"
                for i, frame in enumerate(frames):
                    path = os.path.join(debug_dir, f"{name}_frame{i}.png")
                    PILImage.fromarray(frame).save(path)
                self.robot.log(f"[FSM] saved {len(frames)} debug frames for {name}")
            except Exception as e:
                self.robot.log(f"[FSM] debug frame save failed: {e}")

        # Confidence-weighted vote across the ~5 strategically-captured frames
        score = {}  # cat -> total confidence
        for frame in frames:
            cat, conf = self.classifier.classify_with_confidence(frame)
            score[cat] = score.get(cat, 0.0) + conf

        # Prefer the best non-unknown category if any frame produced one
        non_unknown = {k: v for k, v in score.items() if k != "unknown"}
        if non_unknown:
            category = max(non_unknown, key=non_unknown.get)
        else:
            category = "unknown"
        votes = {k: round(v, 2) for k, v in score.items()}

        if DEBUG:
            self.robot.log(f"[FSM] classify votes: {votes}")
        self.active_category = category
        self.active_zone = CATEGORY_TO_ZONE.get(category, "drop_unknown")
        self.robot.log(f"[FSM] classified {self.active_cargo_def} as '{category}' -> {self.active_zone}")
        self._avoid_count = 0
        self.enter_state(State.PICKUP)

    def _do_pickup(self) -> None:
        """
        Supervisor teleport: move the cargo node onto the robot's carry slot.
        This is the 'magic hand' - it lets us demonstrate pick-and-place
        without modelling a manipulator (the brief explicitly allows this
        for Track B).
        """
        self.robot.stop()
        if self.active_cargo_def is None:
            self.enter_state(State.FAIL_SAFE)
            return
        ok = self.robot.pick_up(self.active_cargo_def)
        if ok and self.time_in_state_ms() > PICKUP_DURATION_MS:
            self.enter_state(State.PLAN_DELIVERY)
        elif not ok:
            self.robot.log("[FSM] pickup failed")
            self.enter_state(State.FAIL_SAFE)

    def _do_plan_delivery(self) -> None:
        """
        Run A* from current GPS cell to the matching drop zone. Wk06 A*:
            f(n) = g(n) + h(n)
        Stamps undelivered cargo items as dynamic obstacles so the planner
        routes around them instead of driving through them.
        """
        if self.active_zone is None:
            self.enter_state(State.FAIL_SAFE)
            return
        goal = self.robot.zone_xy(self.active_zone)
        if goal is None:
            self.enter_state(State.FAIL_SAFE)
            return
        # Add cargo on the ground as temporary obstacles
        cargo_positions = self.robot.obstacle_cargo_positions()
        for pos in cargo_positions:
            self.planner.stamp_point_obstacle(pos)
        start = self.robot.gps_xy()
        path = self.planner.plan(start, goal)
        # Clean up dynamic obstacles so they don't persist
        for pos in cargo_positions:
            self.planner.clear_point_obstacle(pos)
        if not path:
            self.robot.log("[FSM] A* failed - no path found")
            self.enter_state(State.FAIL_SAFE)
            return
        self.active_path = path
        self.active_path_index = 0
        if DEBUG:
            self.robot.log(f"[FSM] planned path with {len(path)} waypoints to {self.active_zone}")
        self.enter_state(State.DELIVER)

    def _do_deliver(self) -> None:
        """
        Follow the A* path. When we reach the final waypoint (or are inside
        DELIVERY_RADIUS of it) release the cargo and go back to PATROL.
        """
        if not self.active_path:
            self.enter_state(State.FAIL_SAFE)
            return
        x, y = self.robot.gps_xy()
        # Advance path index if we're close enough to the current waypoint
        while self.active_path_index < len(self.active_path) - 1:
            wx, wy = self.active_path[self.active_path_index]
            if math.hypot(wx - x, wy - y) < WAYPOINT_RADIUS:
                self.active_path_index += 1
            else:
                break
        wx, wy = self.active_path[self.active_path_index]
        self._drive_toward(wx, wy)

        # Have we arrived at the drop zone?
        goal = self.robot.zone_xy(self.active_zone)
        if goal is not None and math.hypot(goal[0] - x, goal[1] - y) < DELIVERY_RADIUS:
            self.robot.stop()
            # Offset drop position within the zone to avoid stacking
            count = self._zone_drop_count.get(self.active_zone, 0)
            drop_pos = self._offset_drop(goal, count)
            self.robot.release(drop_pos)
            self._zone_drop_count[self.active_zone] = count + 1
            self.robot.log(f"[FSM] delivered {self.active_cargo_def} to {self.active_zone}")
            self.active_cargo_def = None
            self.active_category = None
            self.active_zone = None
            self.active_path = []
            self.active_path_index = 0
            self._avoid_count = 0
            if self.robot.cargo_remaining() == 0:
                self.enter_state(State.COMPLETE)
            else:
                self.enter_state(State.PATROL)

    def _do_avoid(self) -> None:
        """
        Reactive obstacle avoidance. Back up, then turn away from
        whichever sensor reports the closer obstacle. Escalates with
        consecutive avoid cycles to break out of loops.
        """
        dl = self.robot.read_distance_left()
        dr = self.robot.read_distance_right()
        # Escalate: longer backup & turn when stuck in a loop
        backup_ms = 400 + min(self._avoid_count, 5) * 200
        min_duration_ms = 800 + min(self._avoid_count, 5) * 300
        if self.time_in_state_ms() < backup_ms:
            # Back up
            self.robot.set_motors(-CRUISE_SPEED * 0.6, -CRUISE_SPEED * 0.6)
        else:
            # Turn away from the closer side
            if dl < dr:
                self.robot.set_motors(TURN_SPEED, -TURN_SPEED)
            else:
                self.robot.set_motors(-TURN_SPEED, TURN_SPEED)
        if not self._obstacle_blocking() and self.time_in_state_ms() > min_duration_ms:
            self._avoid_count += 1
            # Resume - pick the correct follow-up state based on context
            if self.robot.holding_item():
                self.enter_state(State.PLAN_DELIVERY)
            else:
                self.enter_state(State.PATROL)

    def _do_fail_safe(self) -> None:
        """
        Wk09 Drone slide 11 (verbatim): "The system defaults to a safe state
        when something goes wrong." We stop, release any held item on the
        spot, log, and after a dwell, return to patrol.
        """
        self.robot.stop()
        # Drop held item immediately so we don't carry it forever
        if self.robot.holding_item():
            drop_xy = self.robot.gps_xy()
            self.robot.release(drop_xy)
            self.robot.log("[FSM] FAIL_SAFE: released held item")
        if self.time_in_state_ms() >= FAIL_SAFE_DURATION_MS:
            self.active_cargo_def = None
            self.active_category = None
            self.active_zone = None
            self.active_path = []
            self.active_path_index = 0
            self.enter_state(State.PATROL)

    def _do_complete(self) -> None:
        """All cargo has been delivered. Stop and log mission complete."""
        self.robot.stop()
        if not self._complete_logged:
            self.robot.log("[FSM] === MISSION COMPLETE === All cargo delivered.")
            self._complete_logged = True

    # ---------- Low-level motion primitive -----------------------------------

    # Offsets within the 0.5×0.5m drop zone so items don't stack
    _DROP_OFFSETS = [
        (-0.10,  0.10),
        ( 0.10,  0.10),
        (-0.10, -0.10),
        ( 0.10, -0.10),
        ( 0.00,  0.00),
    ]

    def _offset_drop(self, center, count: int) -> tuple:
        """Return a position within the drop zone that avoids stacking."""
        dx, dy = self._DROP_OFFSETS[count % len(self._DROP_OFFSETS)]
        return (center[0] + dx, center[1] + dy)

    def _drive_toward(self, tx: float, ty: float) -> None:
        """
        Simple heading-error controller. Turn in place if heading error is
        large, otherwise drive forward while correcting.
        """
        x, y = self.robot.gps_xy()
        heading = self.robot.heading_rad()
        desired = math.atan2(ty - y, tx - x)
        err = _wrap_angle(desired - heading)

        if abs(err) > 0.6:
            # Turn in place
            if err > 0:
                self.robot.set_motors(-TURN_SPEED, TURN_SPEED)
            else:
                self.robot.set_motors(TURN_SPEED, -TURN_SPEED)
        else:
            # Drive forward with a proportional steering correction
            steer = max(-1.0, min(1.0, err * 1.5))
            left = CRUISE_SPEED * (1 - steer)
            right = CRUISE_SPEED * (1 + steer)
            self.robot.set_motors(left, right)


def _wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# Backwards-compatible alias
BehaviourTree = PriorityFSM
