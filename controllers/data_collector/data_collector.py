"""
data_collector.py - Auto-labelled Training Data Collector
=========================================================

Griffith 3003ICT - Programming for Robotics - Assessment 1

Uses Supervisor teleportation to instantly place the robot at observation
poses around each cargo item, bypassing all navigation/collision issues.
The camera's Recognition API auto-labels every frame.

Approach (Supervisor teleportation):
  1. For each cargo item, generate a dense ring of observation poses
     at multiple distances and angles, plus jittered variants.
  2. Teleport the robot to each pose, face the camera at the cargo,
     wait a few sim steps for the camera to render, then save the frame.
  3. Generate a grid of background poses across the arena for "unknown"
     (no cargo visible) frames.

This collects balanced data (>=500 per class) in ~3 simulated minutes
with zero risk of getting stuck or colliding.

Output layout (picked up by controllers/sorting_robot/train.py):

    controllers/data_collector/data/
        fragile/     (frame_00001.png, ...)
        standard/    (...)
        hazardous/   (...)
        unknown/     (...)

Usage:
  1. Set the Robot's controller field to "data_collector" in the scene tree.
  2. File -> Revert World, then press Run. Wait for "DONE" in console.
  3. Restore controller to "sorting_robot".
  4. cd controllers/sorting_robot && python train.py --data ../data_collector/data
"""

from __future__ import annotations

import math
import os
import random
import sys
import traceback


def _log(msg: str) -> None:
    print(f"[data_collector] {msg}", flush=True)


_log("starting")

try:
    from controller import Supervisor  # type: ignore[import-not-found]
except Exception as exc:
    _log(f"FATAL: Webots controller module missing: {exc}")
    raise

try:
    import numpy as np  # type: ignore
except Exception as exc:
    _log(f"FATAL: numpy required: {exc}")
    raise

try:
    from PIL import Image  # type: ignore
except Exception as exc:
    _log(f"FATAL: Pillow required: {exc}")
    raise


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CARGO_CATEGORIES = {
    "CARGO_JAMJAR_A":    "fragile",
    "CARGO_JAMJAR_B":    "fragile",
    "CARGO_BISCUIT":     "fragile",
    "CARGO_APPLE":       "standard",
    "CARGO_CAN":         "standard",
    "CARGO_OILBARREL_A": "hazardous",
    "CARGO_OILBARREL_B": "hazardous",
}

MAX_PER_CLASS = 500          # target frames per category
# The unknown class is intentionally larger -- it needs to absorb both the
# empty-arena background poses AND the new wall-facing distractor poses.
# Outlier exposure: enriching unknown with diverse non-cargo views teaches
# the CNN a generalised 'not one of the three trained categories' feature,
# which is what gives us reliable OOD detection on untrained items.
UNKNOWN_MAX_PER_CLASS = 800
ROBOT_Z = 0.05              # spawn height (matches world file)
SETTLE_STEPS = 3            # sim steps after teleport for camera to update

# Observation orbit geometry
NUM_ANGLES = 24              # every 15 degrees
ORBIT_DISTS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
JITTER_N = 2                # extra jittered copies per valid base pose
JITTER_POS = 0.03           # position jitter (metres)
JITTER_HDG = 0.08           # heading jitter (radians)

# Operable arena bounds. The actual arena (ARENA_BOUNDS in sorting_robot.py)
# is (-3.5, -3.0, 3.5, 3.0); we shrink by a small margin so the robot's body
# stays clear of the walls when teleported here. These bounds are used for
# BOTH the cargo orbit generator AND the wall-facing distractor poses.
AX_MIN, AY_MIN = -3.2, -2.7
AX_MAX, AY_MAX = 3.35, 2.7
WALL_XMIN, WALL_YMIN = AX_MIN, AY_MIN
WALL_XMAX, WALL_YMAX = AX_MAX, AY_MAX
WALL_EDGE_OFFSET = 0.35     # how far from the wall to stand when facing it

# Static obstacle bounding boxes (axis-aligned, world coordinates).
# These mirror STATIC_OBSTACLES in sorting_robot.py with a small inflation
# for robot clearance so the orbit generator doesn't teleport us inside one.
STATIC_OBS = [
    # Shipping containers
    ((-0.74, 0.23), (0.78, 0.79)),    # Container 5 (Green Middle)
    ((1.50, 1.24), (2.63, 2.36)),     # Container 4 (Yellow Angled)
    ((2.33, -0.75), (3.29, 0.72)),    # Containers 1-3 (Combined Block)
    # Warehouse racks
    ((-0.05, -2.71), (2.19, -2.37)),  # Rack 1
    ((-1.75, -2.71), (-0.66, -2.38)), # Rack 2
    ((-3.23, -2.72), (-2.28, -2.13)), # Rack 3
    ((-1.14, 2.19), (1.13, 2.52)),    # Rack 4
]

# Older 'shelf zone' list is no longer needed -- the current arena uses the
# racks above instead. Kept empty for backward-compatible iteration sites.
SHELF_ZONES = []

# Don't teleport inside another cargo item
CARGO_CLEARANCE = 0.14

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_HERE, "data")


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def _pose_blocked(x: float, y: float) -> bool:
    """Return True if a robot placed here would overlap an obstacle or wall."""
    if not (AX_MIN < x < AX_MAX and AY_MIN < y < AY_MAX):
        return True
    for (x0, y0), (x1, y1) in STATIC_OBS:
        if x0 < x < x1 and y0 < y < y1:
            return True
    for (x0, y0), (x1, y1) in SHELF_ZONES:
        if x0 < x < x1 and y0 < y < y1:
            return True
    return False


def _too_close(x: float, y: float, cargo_pos: dict, skip: str) -> bool:
    """Return True if pose is inside the footprint of a non-target cargo."""
    for dn, (cx, cy) in cargo_pos.items():
        if dn == skip:
            continue
        if math.hypot(x - cx, y - cy) < CARGO_CLEARANCE:
            return True
    return False


# ---------------------------------------------------------------------------
# Pose generation
# ---------------------------------------------------------------------------

def _build_cargo_poses(cargo_pos: dict) -> list:
    """Observation ring poses around every cargo item, with jitter variants."""
    poses = []
    for def_name, (cx, cy) in cargo_pos.items():
        for dist in ORBIT_DISTS:
            for i in range(NUM_ANGLES):
                angle = 2.0 * math.pi * i / NUM_ANGLES
                px = cx + dist * math.cos(angle)
                py = cy + dist * math.sin(angle)
                if _pose_blocked(px, py) or _too_close(px, py, cargo_pos, def_name):
                    continue
                heading = math.atan2(cy - py, cx - px)
                poses.append((def_name, px, py, heading))
                # Jittered variants for frame diversity
                for _ in range(JITTER_N):
                    jx = px + random.uniform(-JITTER_POS, JITTER_POS)
                    jy = py + random.uniform(-JITTER_POS, JITTER_POS)
                    jh = heading + random.uniform(-JITTER_HDG, JITTER_HDG)
                    if not _pose_blocked(jx, jy):
                        poses.append((def_name, jx, jy, jh))
    random.shuffle(poses)
    return poses


def _build_bg_poses(cargo_pos: dict) -> list:
    """Grid of poses across the arena for 'unknown' (background) frames."""
    poses = []
    x = AX_MIN + 0.10
    while x < AX_MAX - 0.05:
        y = AY_MIN + 0.10
        while y < AY_MAX - 0.05:
            if not _pose_blocked(x, y):
                for hi in range(12):
                    heading = 2.0 * math.pi * hi / 12
                    poses.append((x, y, heading))
            y += 0.30
        x += 0.30
    random.shuffle(poses)
    return poses


def _build_wall_poses() -> list:
    """
    Close-up wall-facing poses spanning all four walls. These produce
    'unknown' frames that contain wall texture, edges, and unusual close-up
    visual content -- a much harder negative class than empty arena floor.

    This is the outlier-exposure piece: by enriching the unknown class with
    diverse non-cargo views, the CNN learns a generalised 'not one of the
    trained categories' feature. Any untrained item (e.g. the held-out
    traffic cone) should then activate this feature at inference time.
    """
    poses = []
    eo = WALL_EDGE_OFFSET
    # North wall (walk along x, stand at y=ymax-eo, face north)
    x = WALL_XMIN + 0.5
    while x < WALL_XMAX - 0.5:
        poses.append((x, WALL_YMAX - eo, math.pi / 2))
        x += 0.4
    # South wall (face south)
    x = WALL_XMIN + 0.5
    while x < WALL_XMAX - 0.5:
        poses.append((x, WALL_YMIN + eo, -math.pi / 2))
        x += 0.4
    # East wall (face east)
    y = WALL_YMIN + 0.5
    while y < WALL_YMAX - 0.5:
        poses.append((WALL_XMAX - eo, y, 0.0))
        y += 0.4
    # West wall (face west)
    y = WALL_YMIN + 0.5
    while y < WALL_YMAX - 0.5:
        poses.append((WALL_XMIN + eo, y, math.pi))
        y += 0.4
    random.shuffle(poses)
    return poses


# ---------------------------------------------------------------------------
# Teleportation
# ---------------------------------------------------------------------------

def _teleport(self_node, tf, rf, x: float, y: float, heading: float) -> None:
    """Instantly move the robot to (x, y) facing heading."""
    tf.setSFVec3f([x, y, ROBOT_Z])
    rf.setSFRotation([0, 0, 1, heading])
    self_node.resetPhysics()


# ---------------------------------------------------------------------------
# Frame capture & labelling
# ---------------------------------------------------------------------------

def _label_from_recognition(supervisor, camera, w: int, h: int) -> str:
    """
    Pick the recognised cargo object closest to the image centre
    and return its category string. Returns 'unknown' if nothing visible.
    """
    try:
        objects = camera.getRecognitionObjects()
    except Exception:
        return "unknown"
    if not objects:
        return "unknown"

    cx, cy = w / 2.0, h / 2.0
    best = None
    best_dist = float("inf")
    for obj in objects:
        try:
            px, py = obj.getPositionOnImage()
        except Exception:
            continue
        dist = math.hypot(px - cx, py - cy)
        try:
            node = supervisor.getFromId(obj.getId())
        except Exception:
            continue
        if node is None:
            continue
        def_name = node.getDef()
        if def_name not in CARGO_CATEGORIES:
            continue
        if dist < best_dist:
            best_dist = dist
            best = def_name

    if best is None:
        return "unknown"
    return CARGO_CATEGORIES[best]


def _capture(camera, supervisor, counts: dict, known_label: str = None) -> bool:
    """Grab a frame, label it, save to disk. Return True if saved.

    If known_label is provided (cargo poses), use it directly — this avoids
    mislabelling when Recognition fails at close range.  For background
    poses, known_label is None and we fall back to Recognition.
    """
    raw = camera.getImage()
    if raw is None:
        return False
    w, h = camera.getWidth(), camera.getHeight()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    rgb = arr[..., [2, 1, 0]].copy()

    if known_label is not None:
        label = known_label
    else:
        label = _label_from_recognition(supervisor, camera, w, h)
    cap = UNKNOWN_MAX_PER_CLASS if label == 'unknown' else MAX_PER_CLASS
    if counts[label] >= cap:
        return False

    idx = counts[label] + 1
    out_path = os.path.join(DATA_ROOT, label, f"frame_{idx:05d}.png")
    try:
        Image.fromarray(rgb).save(out_path)
        counts[label] = idx
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())

    # Wipe any frames left over from previous data-collection runs. Without
    # this, partial new runs leave stale PNGs interleaved with fresh ones
    # and training pulls in data from old arena layouts. Each run starts
    # from frame_00001.png, so without clearing we'd silently retain whatever
    # had a higher index from the previous run.
    if os.path.isdir(DATA_ROOT):
        wiped = 0
        for cat_dir in ("fragile", "standard", "hazardous", "unknown"):
            cat_path = os.path.join(DATA_ROOT, cat_dir)
            if not os.path.isdir(cat_path):
                continue
            for fname in os.listdir(cat_path):
                if fname.startswith("frame_") and fname.endswith(".png"):
                    try:
                        os.remove(os.path.join(cat_path, fname))
                        wiped += 1
                    except OSError:
                        pass
        if wiped:
            _log(f'cleared {wiped} stale frames from previous run')

    # Outlier-exposure note: if any 'test' / held-out items (e.g. CARGO_CONE)
    # are still present in the scene, they will appear in some wall and
    # background frames and get labelled 'unknown' via Recognition fallthrough.
    # That'd defeat the held-out test. Remove or move them out of the arena
    # before running data collection.
    test_items_in_scene = []
    for test_def in ("CARGO_CONE",):
        if supervisor.getFromDef(test_def) is not None:
            test_items_in_scene.append(test_def)
    if test_items_in_scene:
        _log(f'WARNING: held-out test items present in scene: {test_items_in_scene}')
        _log("WARNING: remove them from the scene tree (or move out of arena)")
        _log('WARNING: before running data collection, otherwise they will be')
        _log("WARNING: trained as 'unknown' and stop being a real OOD test.")

    camera = supervisor.getDevice("camera")
    camera.enable(time_step)
    try:
        camera.recognitionEnable(time_step)
    except Exception as exc:
        _log(f"FATAL: Recognition not available: {exc}")
        return

    # One step to initialise sensors
    supervisor.step(time_step)

    self_node = supervisor.getSelf()
    trans_field = self_node.getField("translation")
    rot_field = self_node.getField("rotation")

    # Read cargo world positions via Supervisor
    cargo_pos = {}
    for def_name in CARGO_CATEGORIES:
        node = supervisor.getFromDef(def_name)
        if node is not None:
            pos = node.getPosition()
            cargo_pos[def_name] = (pos[0], pos[1])
    _log(f"found {len(cargo_pos)}/{len(CARGO_CATEGORIES)} cargo nodes")

    # Prepare output directories
    cats = ["fragile", "standard", "hazardous", "unknown"]
    for cat in cats:
        os.makedirs(os.path.join(DATA_ROOT, cat), exist_ok=True)
    counts = {c: 0 for c in cats}

    # Build pose plans
    cargo_poses = _build_cargo_poses(cargo_pos)
    bg_poses = _build_bg_poses(cargo_pos)
    _log(f"plan: {len(cargo_poses)} cargo + {len(bg_poses)} background poses")

    saved = 0

    # --- Phase 1: Cargo observations ----------------------------------------
    for def_name, px, py, heading in cargo_poses:
        cat = CARGO_CATEGORIES[def_name]
        if counts[cat] >= MAX_PER_CLASS:
            continue
        if all(counts[c] >= (UNKNOWN_MAX_PER_CLASS if c == 'unknown' else MAX_PER_CLASS) for c in counts):
            break

        _teleport(self_node, trans_field, rot_field, px, py, heading)
        for _ in range(SETTLE_STEPS):
            if supervisor.step(time_step) == -1:
                return

        cat = CARGO_CATEGORIES[def_name]
        if _capture(camera, supervisor, counts, known_label=cat):
            saved += 1
            if saved % 100 == 0:
                _log(f"  progress ({saved} saved): {counts}")

    _log(f"after cargo phase: {counts}")

    # --- Phase 2: Background for 'unknown' class ---------------------------
    for bx, by, heading in bg_poses:
        if counts["unknown"] >= UNKNOWN_MAX_PER_CLASS:
            break

        _teleport(self_node, trans_field, rot_field, bx, by, heading)
        for _ in range(SETTLE_STEPS):
            if supervisor.step(time_step) == -1:
                return

        if _capture(camera, supervisor, counts):
            saved += 1
            if saved % 100 == 0:
                _log(f"  progress ({saved} saved): {counts}")

    _log(f"after background phase: {counts}")

    # --- Phase 3: Wall-facing distractor poses for outlier exposure -------
    # These enrich the 'unknown' class with diverse close-up wall/edge views
    # so the CNN learns a generalised 'not trained cargo' feature.
    wall_poses = _build_wall_poses()
    _log(f"phase 3: {len(wall_poses)} wall-facing distractor poses")
    for wx, wy, heading in wall_poses:
        if counts['unknown'] >= UNKNOWN_MAX_PER_CLASS:
            break
        _teleport(self_node, trans_field, rot_field, wx, wy, heading)
        for _ in range(SETTLE_STEPS):
            if supervisor.step(time_step) == -1:
                return
        if _capture(camera, supervisor, counts):
            saved += 1
            if saved % 100 == 0:
                _log(f"  progress ({saved} saved): {counts}")

    _log(f"DONE: {counts} ({saved} total frames saved)")

    # Park robot at origin
    _teleport(self_node, trans_field, rot_field, 0.0, 0.0, 0.0)
    supervisor.step(time_step)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log("FATAL: unhandled exception")
        traceback.print_exc()
        raise
