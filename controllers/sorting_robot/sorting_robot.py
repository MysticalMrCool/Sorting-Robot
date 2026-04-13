"""
sorting_robot.py - Layer 1: Hardware Control + Main Loop
========================================================

Griffith 3003ICT - Programming for Robotics - Assessment 1
Track B - Autonomous Sorting Warehouse Robot (Webots R2025a)

Wk04-EmbeddedAI slide 17 - Behaviour Architecture Layering (verbatim):
    Layer 1: Hardware control   <- THIS FILE
    Layer 2: Reactive logic     <- behaviour_tree.py
    Layer 3: AI enhancement     <- inference.py + model.py

Wk05-Webots control loop pattern:
    while robot.step(TIME_STEP) != -1:
        read_sensors()
        update_state()
        execute_state()

This file is the thin Webots-API wrapper. It deliberately does no decision
making - it just exposes a clean object (`RobotAPI`) that the behaviour
tree can call without knowing anything about Webots.

Defensive-startup rules from the project brief:
  - All prints use flush=True so the Webots console shows output immediately
  - Heavy imports (torch, numpy) are wrapped in try/except and fall through
    to safe defaults so the controller still runs on a barebones Python
"""

from __future__ import annotations

import math
import os
import sys
import traceback


def _boot_log(msg: str) -> None:
    """Single helper so we can trace controller startup from the Webots console."""
    print(f"[sorting_robot] {msg}", flush=True)


_boot_log("controller starting")

# --- Webots imports ---------------------------------------------------------

try:
    from controller import Supervisor  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - only triggers outside Webots
    _boot_log(f"FATAL: could not import Webots controller module: {exc}")
    _boot_log("This controller must be run from inside Webots (R2025a).")
    raise

# --- Optional heavy deps ----------------------------------------------------

try:
    import numpy as np  # type: ignore
    _NUMPY_OK = True
except Exception as exc:
    _boot_log(f"WARNING: numpy not available ({exc}); some features disabled")
    np = None  # type: ignore
    _NUMPY_OK = False

# --- Our own modules --------------------------------------------------------

# Make this file's directory importable even when Webots launches from elsewhere
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from behaviour_tree import PriorityFSM
    from pathfinding import AStarPlanner
    from inference import Classifier
except Exception:
    _boot_log("FATAL: could not import project modules:")
    traceback.print_exc()
    raise


# ---------------------------------------------------------------------------
# Static warehouse map - mirrors the obstacle and drop-zone layout in
# sorting_warehouse.wbt. Keeping the numbers here (not pulling them from
# the scene graph via supervisor APIs) makes the planner deterministic and
# easy to reason about for the report.
# ---------------------------------------------------------------------------

ARENA_BOUNDS = (-2.0, -1.5, 2.0, 1.5)  # xmin, ymin, xmax, ymax
GRID_RESOLUTION = 0.1
ROBOT_RADIUS = 0.11

# (p1, p2) axis-aligned boxes - same as OBSTACLE_1 / OBSTACLE_2 in the .wbt
STATIC_OBSTACLES = [
    ((-0.4, 0.55), (0.4, 0.65)),   # obstacle_1
    ((-0.4, -0.65), (0.4, -0.55)),  # obstacle_2
]

# Drop zone world positions - must match the .wbt translations
DROP_ZONES = {
    "drop_fragile":   (-1.6, 1.1),
    "drop_standard":  (1.6, 1.1),
    "drop_hazardous": (-1.6, -1.1),
    "drop_unknown":   (1.6, -1.1),
}

# Patrol waypoints - a simple rectangular loop around the cargo area
PATROL_WAYPOINTS = [
    (-1.2, -0.9),
    (1.2, -0.9),
    (1.2, 0.9),
    (-1.2, 0.9),
]

# Distance at which the robot considers a Recognition object "visible"
VISIBLE_RANGE = 1.2

# Items we treat as cargo (as opposed to walls / drop pads)
CARGO_DEF_NAMES = [
    "CARGO_JAMJAR_A",
    "CARGO_JAMJAR_B",
    "CARGO_BISCUIT",
    "CARGO_APPLE",
    "CARGO_CAN",
    "CARGO_OILBARREL_A",
    "CARGO_OILBARREL_B",
]


# ---------------------------------------------------------------------------
# RobotAPI - everything the BT needs, nothing it doesn't
# ---------------------------------------------------------------------------

class RobotAPI:
    """
    Thin wrapper around Webots Supervisor + devices. Decision-free.

    The Behaviour Tree only sees this object - it never touches Webots
    directly. That's the whole point of Layer 1.
    """

    def __init__(self, supervisor: Supervisor):
        self.supervisor = supervisor
        self.time_step = int(supervisor.getBasicTimeStep())

        # --- Devices ---------------------------------------------------------
        self.left_motor = supervisor.getDevice("left wheel motor")
        self.right_motor = supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.left_sensor_pos = supervisor.getDevice("left wheel sensor")
        self.right_sensor_pos = supervisor.getDevice("right wheel sensor")
        self.left_sensor_pos.enable(self.time_step)
        self.right_sensor_pos.enable(self.time_step)

        self.ds_left = supervisor.getDevice("ds_left")
        self.ds_right = supervisor.getDevice("ds_right")
        self.ds_left.enable(self.time_step)
        self.ds_right.enable(self.time_step)

        self.camera = supervisor.getDevice("camera")
        self.camera.enable(self.time_step)
        try:
            self.camera.recognitionEnable(self.time_step)
            self._recognition_ok = True
        except Exception as exc:
            _boot_log(f"WARNING: camera recognitionEnable failed: {exc}")
            self._recognition_ok = False

        self.gps = supervisor.getDevice("gps")
        self.gps.enable(self.time_step)

        self.compass = supervisor.getDevice("compass")
        self.compass.enable(self.time_step)

        # --- Supervisor handles ---------------------------------------------
        self.self_node = supervisor.getSelf()
        self.cargo_nodes = {}
        for def_name in CARGO_DEF_NAMES:
            node = supervisor.getFromDef(def_name)
            if node is None:
                _boot_log(f"WARNING: could not find node DEF {def_name}")
            else:
                self.cargo_nodes[def_name] = node

        # --- State -----------------------------------------------------------
        self._held_item = None      # DEF name of the currently-held item
        self._patrol_index = 0
        self._delivered = set()
        self.distance_left_reading = 999.0
        self.distance_right_reading = 999.0

    # ---------- Sensor reads (one per tick) ---------------------------------

    def read_sensors(self) -> None:
        """Called once at the top of every tick by the main loop."""
        self.distance_left_reading = _ds_to_metres(self.ds_left.getValue())
        self.distance_right_reading = _ds_to_metres(self.ds_right.getValue())

    def read_distance_left(self) -> float:
        return self.distance_left_reading

    def read_distance_right(self) -> float:
        return self.distance_right_reading

    def gps_xy(self) -> tuple:
        v = self.gps.getValues()
        return float(v[0]), float(v[1])

    def heading_rad(self) -> float:
        """
        Derive heading from the compass's north-vector.
        In Webots the Compass returns a 3-vector pointing toward world north.
        """
        n = self.compass.getValues()
        return math.atan2(n[0], n[1])

    def now_ms(self) -> float:
        return self.supervisor.getTime() * 1000.0

    # ---------- Camera / Recognition ----------------------------------------

    def grab_image(self):
        """Return the camera frame as an H x W x 3 numpy array, or None."""
        if np is None:
            return None
        raw = self.camera.getImage()
        if raw is None:
            return None
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        # Webots returns BGRA bytes
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        rgb = arr[..., [2, 1, 0]].copy()
        return rgb

    def next_visible_cargo(self):
        """
        Query the camera's built-in Recognition to find the closest cargo
        item currently in the field of view. Returns a dict with:
            def: supervisor DEF name
            xy:  world-frame (x, y) of the item
            distance: metres from the robot
        or None if nothing visible.
        """
        if not self._recognition_ok:
            return self._fallback_visible_cargo()
        try:
            objects = self.camera.getRecognitionObjects()
        except Exception:
            return self._fallback_visible_cargo()

        rx, ry = self.gps_xy()
        best = None
        best_dist = float("inf")
        for obj in objects:
            # Webots recognition gives node id; we use it to resolve the DEF
            try:
                node = self.supervisor.getFromId(obj.getId())
            except Exception:
                node = None
            if node is None:
                continue
            def_name = node.getDef()
            if def_name not in self.cargo_nodes:
                continue
            if def_name in self._delivered or def_name == self._held_item:
                continue
            pos = node.getPosition()
            dx = pos[0] - rx
            dy = pos[1] - ry
            dist = math.hypot(dx, dy)
            if dist > VISIBLE_RANGE:
                continue
            if dist < best_dist:
                best_dist = dist
                best = {
                    "def": def_name,
                    "xy": (float(pos[0]), float(pos[1])),
                    "distance": dist,
                }
        return best

    def _fallback_visible_cargo(self):
        """
        If the Recognition device is unavailable, fall back to a pure
        supervisor-proximity query. This keeps the demo alive on machines
        where the recognition feature isn't compiled in.
        """
        rx, ry = self.gps_xy()
        heading = self.heading_rad()
        best = None
        best_dist = float("inf")
        for def_name, node in self.cargo_nodes.items():
            if def_name in self._delivered or def_name == self._held_item:
                continue
            pos = node.getPosition()
            dx = pos[0] - rx
            dy = pos[1] - ry
            dist = math.hypot(dx, dy)
            if dist > VISIBLE_RANGE:
                continue
            # Only "see" things within roughly a 60 degree cone in front
            bearing = math.atan2(dy, dx)
            err = abs(_wrap(bearing - heading))
            if err > 1.05:  # ~60 deg
                continue
            if dist < best_dist:
                best_dist = dist
                best = {
                    "def": def_name,
                    "xy": (float(pos[0]), float(pos[1])),
                    "distance": dist,
                }
        return best

    # ---------- Motor primitives --------------------------------------------

    def set_motors(self, left: float, right: float) -> None:
        self.left_motor.setVelocity(_clamp(left, -10.0, 10.0))
        self.right_motor.setVelocity(_clamp(right, -10.0, 10.0))

    def stop(self) -> None:
        self.set_motors(0.0, 0.0)

    # ---------- Supervisor pick-and-place -----------------------------------

    def pick_up(self, def_name: str) -> bool:
        node = self.cargo_nodes.get(def_name)
        if node is None:
            return False
        # Teleport the node onto our carry slot and freeze its physics
        carry_x, carry_y = self.gps_xy()
        carry_z = 0.18
        try:
            translation_field = node.getField("translation")
            translation_field.setSFVec3f([carry_x, carry_y, carry_z])
            # Best-effort: kill any residual linear/angular velocity so the
            # item doesn't inherit physics impulses from its old position.
            try:
                node.resetPhysics()
            except Exception:
                pass
            self._held_item = def_name
            return True
        except Exception as exc:
            _boot_log(f"pick_up failed for {def_name}: {exc}")
            return False

    def release(self, at_xy) -> None:
        if self._held_item is None:
            return
        node = self.cargo_nodes.get(self._held_item)
        if node is None:
            self._held_item = None
            return
        try:
            translation_field = node.getField("translation")
            translation_field.setSFVec3f([float(at_xy[0]), float(at_xy[1]), 0.05])
            try:
                node.resetPhysics()
            except Exception:
                pass
        except Exception as exc:
            _boot_log(f"release failed: {exc}")
        self._delivered.add(self._held_item)
        self._held_item = None

    def holding_item(self) -> bool:
        return self._held_item is not None

    # ---------- Patrol book-keeping -----------------------------------------

    def patrol_target_xy(self) -> tuple:
        return PATROL_WAYPOINTS[self._patrol_index]

    def advance_patrol(self) -> None:
        self._patrol_index = (self._patrol_index + 1) % len(PATROL_WAYPOINTS)

    def zone_xy(self, zone_name: str):
        return DROP_ZONES.get(zone_name)

    def cargo_remaining(self) -> int:
        return len(self.cargo_nodes) - len(self._delivered) - (1 if self._held_item else 0)

    def obstacle_cargo_positions(self) -> list:
        """Return (x, y) positions of cargo items on the ground that the robot
        should route around. Excludes the held item and delivered items."""
        positions = []
        for def_name, node in self.cargo_nodes.items():
            if def_name in self._delivered or def_name == self._held_item:
                continue
            try:
                pos = node.getPosition()
                positions.append((float(pos[0]), float(pos[1])))
            except Exception:
                pass
        return positions

    def log(self, msg: str) -> None:
        print(msg, flush=True)

    # ---------- While-we're-carrying-something: follow the robot -------------

    def update_carried_item(self) -> None:
        """
        Called every tick while we're holding something. Keeps the item
        glued to the carry slot even as the robot moves.
        """
        if self._held_item is None:
            return
        node = self.cargo_nodes.get(self._held_item)
        if node is None:
            return
        x, y = self.gps_xy()
        try:
            node.getField("translation").setSFVec3f([x, y, 0.18])
            node.resetPhysics()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ds_to_metres(raw: float) -> float:
    """
    Convert the DistanceSensor's raw lookupTable output to a metric distance.

    The lookupTable in the .wbt maps:
        0.00 m -> 1000
        0.05 m -> 1000
        0.50 m -> 0
    There is a dead zone from 0.00-0.05 m where raw stays at 1000.
    The linear region maps raw 0-1000 to 0.50-0.05 m.
    """
    if raw >= 1000:
        return 0.0   # within dead zone (0-5 cm)
    if raw <= 0:
        return 0.5
    return 0.05 + 0.45 * (1.0 - raw / 1000.0)


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _boot_log("booting supervisor")
    supervisor = Supervisor()
    api = RobotAPI(supervisor)
    _boot_log(f"time_step={api.time_step} ms, cargo_nodes={list(api.cargo_nodes.keys())}")

    # Classifier: try the trained CNN, fall back to colour histogram
    weights_path = os.path.join(_HERE, "model.pt")
    classifier = Classifier(weights_path=weights_path)

    # Planner: built once from the known static map
    planner = AStarPlanner(
        bounds=ARENA_BOUNDS,
        resolution=GRID_RESOLUTION,
        obstacles=STATIC_OBSTACLES,
        robot_radius=ROBOT_RADIUS,
    )
    _boot_log("A* planner ready")

    # Behaviour tree
    bt = PriorityFSM(api, classifier, planner)
    _boot_log("priority FSM ready - entering main loop")

    # ---- Wk05 main control loop ---------------------------------------------
    # readSensors() -> updateState() -> executeState()
    DEBUG = False
    tick_count = 0
    while supervisor.step(api.time_step) != -1:
        api.read_sensors()
        bt.tick()
        api.update_carried_item()
        tick_count += 1
        # Heartbeat every ~1 second (at 32 ms step) so we can confirm the loop
        # is running and see where the robot is.
        if DEBUG and tick_count % 31 == 0:
            x, y = api.gps_xy()
            dl = api.read_distance_left()
            dr = api.read_distance_right()
            lv = api.left_motor.getVelocity()
            rv = api.right_motor.getVelocity()
            _boot_log(
                f"tick={tick_count} state={bt.current_state.name} "
                f"pos=({x:+.2f},{y:+.2f}) ds=({dl:.2f},{dr:.2f}) "
                f"motors=({lv:+.2f},{rv:+.2f})"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _boot_log("FATAL: unhandled exception in main loop")
        traceback.print_exc()
        raise
