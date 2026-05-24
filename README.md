# Sorting Warehouse

3003ICT Programming for Robotics — Assessment 1 (Track B)

Autonomous sorting robot in Webots R2025a. The robot patrols, classifies cargo using a CNN, plans paths with A\*, and delivers items to the correct drop zones.

## Demonstration Video

5-minute project demonstration:

https://github.com/user-attachments/assets/6b3cbbda-fd16-4877-8187-a4ed23fe9ae5

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

`COMPLETE` → `FAIL_SAFE` → `AVOID` → `DELIVER` → `PLAN_DELIVERY` → `PICKUP` → `CLASSIFY` → `APPROACH_TARGET` → `PATROL`

The robot captures 4 frames at staggered distances during approach plus 1 settled-camera frame at the stop point, then uses a confidence-weighted vote across all 5 to classify each item as fragile, standard, hazardous, or unknown. Inference runs as a pure-numpy forward pass using `model_weights.npz` — no PyTorch needed at runtime.

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

## Requirements Coverage

How the brief's technical requirements map to this codebase.

| Brief requirement | Where it lives |
|---|---|
| Core — 2+ inputs | 7 sensors wired in `RobotAPI.__init__` (`sorting_robot.py`) |
| Core — 2+ outputs | Left/right wheel motors + supervisor `pick_up`/`release` (`sorting_robot.py`) |
| Core — FSM with 4+ states | 9-state Priority FSM (`behaviour_tree.py`, `class State`) |
| Core — Multi-condition decision logic | Priority selector at the top of `PriorityFSM.tick()` |
| Core — Safety / fail-safe | `AVOID` state + `FAIL_SAFE` escalation after 3 strikes in 5 s; OOD items routed to `drop_unknown` via confidence threshold |
| Core — Structured architecture | Three-layer Sense–Think–Act split: `sorting_robot.py` / `behaviour_tree.py` / `inference.py` |
| Track B — Navigation | A\* waypoint planning + figure-8 patrol loop (`pathfinding.py`, `_do_patrol`, `_do_deliver`) |
| Track B — Obstacle avoidance | Distance-sensor `_do_avoid` (<0.18 m) + inflated-obstacle A\* grid |
| Track B — Perception-driven decision | CNN category drives drop-zone lookup in `CATEGORY_TO_ZONE` (`behaviour_tree.py`) |
| Advanced — Perception (Vision-Based) | `SortingCNN` (3 conv + 2 FC) with multi-frame confidence-weighted voting, temperature-scaled softmax, outlier-exposure unknown class, three-tier runtime fallback (PyTorch → NumPy → colour histogram) |
