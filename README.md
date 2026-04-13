# Sorting Warehouse

Griffith University 3003ICT — Programming for Robotics — Assessment 1 (Track B).

A vision-based autonomous sorting-warehouse robot, built in **Webots R2025a**.
A single wheeled robot patrols the warehouse, classifies cargo with an on-board
CNN, plans paths with A\*, picks up items via a Supervisor teleport, and
delivers each item to the matching drop zone.

## Quick Start (clone → run in 2 minutes)

**Prerequisites:** Webots R2025a, Python 3.10+ on system PATH with `numpy`
and `Pillow` installed.

```
pip install numpy Pillow
```

1. Clone this repo.
2. Open `worlds/sorting_warehouse.wbt` in Webots.
3. Press **Play** ▶️. The controller is already wired up.

That's it. The pre-trained CNN weights (`model_weights.npz`) are included in
the repo. The robot uses a pure-numpy inference path at runtime — **no
PyTorch required** to run the demo.

> **Note:** Webots uses your *system* Python, not any virtual environment.
> The venv (`.venv/`) is only needed if you want to retrain the CNN.

## Architecture

```
Layer 1: Hardware control   → controllers/sorting_robot/sorting_robot.py
Layer 2: Reactive logic     → controllers/sorting_robot/behaviour_tree.py
Layer 3: AI enhancement     → controllers/sorting_robot/inference.py
                               controllers/sorting_robot/model.py
```

The behaviour tree is a **Priority FSM** with **nine states**:

| Priority | State | Purpose |
|----------|-------|---------|
| 1 (highest) | `FAIL_SAFE` | Stop, release held item, recover |
| 2 | `AVOID` | Reactive obstacle avoidance (escalating) |
| 3 | `DELIVER` | Follow A\* path to drop zone |
| 4 | `PLAN_DELIVERY` | Run A\* from current position to target zone |
| 5 | `PICKUP` | Supervisor-teleport item onto carry slot |
| 6 | `CLASSIFY` | Stop, capture frame, run CNN majority vote |
| 7 | `APPROACH_TARGET` | Drive toward visible cargo |
| 8 | `PATROL` | Walk figure-8 waypoint loop |
| 9 (lowest) | `COMPLETE` | All cargo delivered — mission finished |

The FSM structure (`enum`, `enter_state` helper, `state_start_ms` dwell
timing) follows the same pattern as the Workshop 4 Smart-Security-Gate
template.

## Classification Approach

The CNN classifies cargo into four categories: **fragile**, **standard**,
**hazardous**, and **unknown**.

During approach, the robot captures frames at **strategic distance
checkpoints** (0.40 m, 0.35 m, 0.30 m, 0.25 m) plus one dwell frame after
stopping — **5 frames total per item**. A confidence-weighted majority vote
across all frames produces the final classification. This avoids the
problems of single-frame close-range classification (where large items like
barrels fill the entire frame).

Three-tier inference fallback:
1. **PyTorch CNN** — if `model.pt` + torch are available (training env)
2. **Numpy CNN** — pure-numpy forward pass using `model_weights.npz` (runtime default)
3. **Colour histogram** — basic heuristic if no weights are found

## Files

```
worlds/
    sorting_warehouse.wbt           Webots R2025a world

controllers/sorting_robot/         Main controller
    sorting_robot.py                Layer 1 — hardware wrapper + main loop
    behaviour_tree.py               Layer 2 — Priority FSM (9 states)
    pathfinding.py                  A* grid planner
    inference.py                    CNN wrapper + fallback chain
    model.py                        PyTorch CNN definition (SortingCNN)
    model_weights.npz               Pre-trained numpy weights (included in repo)
    train.py                        Offline CNN trainer (requires venv)
    model.pt                        PyTorch checkpoint (gitignored — too large)

controllers/data_collector/
    data_collector.py               Auto-labelled training data collector
                                    (Supervisor teleportation + Recognition API)

requirements.txt                    Python deps for training (venv only)
```

## Retraining the CNN (optional)

Group members do **not** need to retrain to run the demo. The numpy weights
are already included. If you want to retrain:

1. Create and activate a venv:
   ```
   python -m venv .venv
   .\.venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
2. Collect training data — change the `SORTING_ROBOT` controller field to
   `"data_collector"`, press Play, let it run for ~2–3 simulated minutes.
3. Train:
   ```
   cd controllers\sorting_robot
   python train.py --data ..\data_collector\data --epochs 20
   ```
4. Export numpy weights (so the runtime can use them without PyTorch):
   ```python
   import torch, numpy as np
   state = torch.load('model.pt', map_location='cpu')
   np.savez('model_weights.npz', **{k: v.numpy() for k, v in state.items()})
   ```

## Debug Logging

Verbose debug output is controlled by a `DEBUG` flag at the top of each
module. Set `DEBUG = True` in `sorting_robot.py`, `behaviour_tree.py`, or
`inference.py` to enable detailed logging for that layer. By default all
debug output is off for a clean demo console.

## Rubric Alignment

| Criterion | Where |
|-----------|-------|
| Technical complexity (30%) | CNN + A\* + BT + supervisor pick-and-place |
| Architecture (15%) | Three-layer split (hardware / reactive / AI) |
| Behaviour & control (15%) | 9-state Priority FSM |
| Safety / fail-safe (10%) | AVOID and FAIL\_SAFE states |
| Code quality (10%) | Modular split, type hints, defensive startup |
| Video (5%) | 3–5 minute Webots run |
| Individual (5%) | Per-member ownership split (see below) |
| Advanced (10%) | Hybrid BT+CNN; auto-labelled training set |

## Suggested Individual Work Split

| Member | Scope |
|--------|-------|
| A | Data collection + offline training + model.pt |
| B | behaviour\_tree.py tuning + demo staging + video |
| C | pathfinding.py + report |
| D | sorting\_warehouse.wbt dressing + architecture diagram + report |

## Inputs / Outputs (rubric floor: ≥2 each)

**Inputs (7):** camera, ds\_left, ds\_right, gps, compass, left wheel sensor,
right wheel sensor

**Outputs (3):** left wheel motor, right wheel motor, supervisor teleport
