# Sorting Warehouse

3003ICT Programming for Robotics — Assessment 1 (Track B)

Autonomous sorting robot in Webots R2025a. The robot patrols, classifies cargo using a CNN, plans paths with A\*, and delivers items to the correct drop zones.

## Setup

You need **Webots R2025a** and **Python 3.10+** with `numpy` and `Pillow` installed on your system:

```
pip install numpy Pillow
```

Then just open `worlds/sorting_warehouse.wbt` and press Play. The trained model weights are already in the repo so everything works out of the box. **You don't need PyTorch or a venv to run it** — those are only for training.

> Webots uses your system Python, not a venv. Make sure numpy and Pillow are installed globally.

## How It Works

Three-layer architecture:

| Layer | File | Role |
|-------|------|------|
| Hardware control | `sorting_robot.py` | Webots API wrapper, sensors, motors |
| Reactive logic | `behaviour_tree.py` | Priority FSM with 9 states |
| AI | `inference.py` + `model.py` | CNN classification + fallback |

The FSM follows the same pattern as the Workshop 4 Smart-Security-Gate (enum states, `enter_state` helper, dwell timing). States in priority order:

`FAIL_SAFE` → `AVOID` → `DELIVER` → `PLAN_DELIVERY` → `PICKUP` → `CLASSIFY` → `APPROACH_TARGET` → `PATROL` → `COMPLETE`

The robot captures 5 frames at different distances during approach, then uses a confidence-weighted vote to classify each item as fragile, standard, hazardous, or unknown. Inference runs as a pure-numpy forward pass using `model_weights.npz` — no PyTorch needed at runtime.

## Files

```
worlds/sorting_warehouse.wbt        The Webots world

controllers/sorting_robot/
    sorting_robot.py                Hardware wrapper + main loop
    behaviour_tree.py               Priority FSM (9 states)
    pathfinding.py                  A* grid planner
    inference.py                    CNN inference + colour-histogram fallback
    model.py                        CNN definition
    model_weights.npz               Trained weights (included in repo)
    train.py                        Offline trainer (needs venv + PyTorch)
    model.pt                        Full PyTorch checkpoint (gitignored)

controllers/data_collector/
    data_collector.py               Training data collector (supervisor teleportation)
```

## Debug Logging

Each module has a `DEBUG` flag at the top. Set it to `True` if you need verbose output for that layer. Off by default.

## Inputs / Outputs

**Inputs (7):** camera, ds\_left, ds\_right, gps, compass, left wheel sensor, right wheel sensor

**Outputs (3):** left wheel motor, right wheel motor, supervisor teleport
