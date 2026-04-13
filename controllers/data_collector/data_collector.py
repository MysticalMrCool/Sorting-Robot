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
ROBOT_Z = 0.05              # spawn height (matches world file)
SETTLE_STEPS = 3            # sim steps after teleport for camera to update

# Observation orbit geometry
NUM_ANGLES = 24              # every 15 degrees
ORBIT_DISTS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
JITTER_N = 2                # extra jittered copies per valid base pose
JITTER_POS = 0.03           # position jitter (metres)
JITTER_HDG = 0.08           # heading jitter (radians)

# Arena interior limits (inside walls with margin for robot body)
AX_MIN, AY_MIN = -1.70, -1.20
AX_MAX, AY_MAX = 1.70, 1.20

# Static obstacle bounding boxes (padded for robot clearance)
STATIC_OBS = [
    ((-0.52, 0.42), (0.52, 0.78)),   # OBSTACLE_1 north bar
    ((-0.52, -0.78), (0.52, -0.42)),  # OBSTACLE_2 south bar
]

# Corner shelf exclusion zones
SHELF_ZONES = [
    ((1.60,  0.12), (1.96,  0.88)),  # SHELF_NE
    ((-1.96, 0.12), (-1.60, 0.88)),  # SHELF_NW
    ((1.60, -0.88), (1.96, -0.12)),  # SHELF_SE
    ((-1.96, -0.88), (-1.60, -0.12)),  # SHELF_SW
]

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
    if counts[label] >= MAX_PER_CLASS:
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
        if all(v >= MAX_PER_CLASS for v in counts.values()):
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
        if counts["unknown"] >= MAX_PER_CLASS:
            break

        _teleport(self_node, trans_field, rot_field, bx, by, heading)
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
