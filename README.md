# Sorting Warehouse

Griffith University 3003ICT - Programming for Robotics - Assessment 1 (Track B).

A vision-based autonomous sorting-warehouse robot, built in Webots R2025a.
A single wheeled robot patrols the warehouse, classifies cargo with an on-board
CNN, plans paths with A*, picks up items via a Supervisor teleport, and
delivers each item to the matching drop zone.

## Architecture (Week 4 slide 17 - Behaviour Architecture Layering)

    Layer 1: Hardware control   -> controllers/sorting_robot/sorting_robot.py
    Layer 2: Reactive logic     -> controllers/sorting_robot/behaviour_tree.py
    Layer 3: AI enhancement     -> controllers/sorting_robot/inference.py
                                   controllers/sorting_robot/model.py

The behaviour tree is a priority selector over eight states: PATROL,
APPROACH_TARGET, CLASSIFY, PICKUP, PLAN_DELIVERY, DELIVER, AVOID, FAIL_SAFE.
The FSM structure (enum, enter_state helper, state_start_ms dwell timing)
is a deliberate Python port of the Week 4 Workshop Smart-Security-Gate
template.

## Quick start

Prerequisites: Webots R2025a and Python 3.10 or newer.

1. Open `worlds/sorting_warehouse.wbt` in Webots.
2. The controller is already wired up (`controller "sorting_robot"` on
   the `DEF SORTING_ROBOT` node). Press Play.
3. On the first run, no `model.pt` exists yet, so the classifier falls
   back to its colour-histogram baseline. The robot will still
   demonstrate the full pipeline - it's just using a dumb classifier.

## Installing the Python dependencies

From a terminal, inside this project directory:

    python -m venv .venv
    .\.venv\Scripts\activate        # Windows
    pip install -r requirements.txt

Then point Webots at that Python interpreter (Tools -> Preferences ->
General -> Python command) at `.venv\Scripts\python.exe`.

## Training the CNN (optional - not required for the demo)

1. Open `worlds/sorting_warehouse.wbt`, temporarily change the
   `SORTING_ROBOT` controller field from `"sorting_robot"` to
   `"data_collector"` and save. Press Play and let it run for 2-3
   simulated minutes. Frames get saved into
   `controllers/data_collector/data/<category>/`.
2. Restore the controller field to `"sorting_robot"`.
3. Train:

       cd controllers\sorting_robot
       python train.py --data ..\data_collector\data --epochs 20

   This writes `model.pt` next to `train.py`. The next time you run
   the Webots simulation, `inference.py` will pick it up automatically
   and use the CNN instead of the colour-histogram fallback.

## Files

    worlds/
        sorting_warehouse.wbt       Webots R2025a world - arena, cargo,
                                    drop zones, obstacles, robot

    controllers/sorting_robot/      Main controller
        sorting_robot.py            Layer 1 - hardware wrapper + main loop
        behaviour_tree.py           Layer 2 - BT + FSM (Week 4 pattern)
        pathfinding.py              A* with f = g + h (Week 6)
        inference.py                CNN wrapper + colour-histogram fallback
        model.py                    PyTorch CNN definition
        train.py                    Offline trainer

    controllers/data_collector/
        data_collector.py           Auto-labelled training data collector
                                    (uses Camera Recognition as oracle)

    requirements.txt                Python deps (numpy required, torch optional)

## Rubric alignment (quick map for the report)

    Technical complexity (30%)  -> CNN + A* + BT + supervisor pick-and-place
    Architecture (15%)          -> Three-layer diagram from Week 4 slide 17
    Behaviour & control (15%)   -> 8-state BT, Week 4 Workshop code pattern
    Safety / fail-safe (10%)    -> AVOID and FAIL_SAFE states; Week 9 phrasing
    Code quality (10%)          -> Modular split, type hints, defensive startup
    Video (5%)                  -> 3-5 minute Webots run
    Individual (5%)             -> Per-member ownership split (see below)
    Advanced (10%)              -> Hybrid BT+CNN; auto-labelled training set

## Suggested individual work split for the group

    Member A - data collection + offline training + model.pt
    Member B - behaviour_tree.py tuning + demo staging + video
    Member C - pathfinding.py + report
    Member D - sorting_warehouse.wbt dressing + architecture diagram + report

## Inputs / outputs (rubric floor: >=2 each)

Inputs  (7): camera, ds_left, ds_right, gps, compass, left wheel sensor,
             right wheel sensor

Outputs (3): left wheel motor, right wheel motor, supervisor teleport
