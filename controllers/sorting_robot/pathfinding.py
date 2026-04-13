"""
pathfinding.py - A* Path Planner
=================================

Griffith 3003ICT - Programming for Robotics - Assessment 1

Wk06-AutoNavi "Path Planning Process":
    1. Create map
    2. Define start and goal
    3. Run search
    4. Generate path
    5. Follow path

A* cost function:

    f(n) = g(n) + h(n)

where:
    g(n) = actual cost from start to node n
    h(n) = heuristic estimate of cost from n to goal (we use octile distance)

The map is a fixed-resolution occupancy grid. Obstacles are inflated by the
robot's radius so the planner returns paths the robot can actually execute
without clipping walls.
"""

import heapq
import math
from typing import List, Tuple, Iterable


Point = Tuple[float, float]
Cell = Tuple[int, int]


class AStarPlanner:
    """
    Grid-based A* over a known static warehouse map.

    Call:
        planner = AStarPlanner(
            bounds=(-2.0, -1.5, 2.0, 1.5),     # (xmin, ymin, xmax, ymax)
            resolution=0.1,                     # 10 cm cells
            obstacles=[((x1, y1), (x2, y2)), ...],  # axis-aligned boxes
            robot_radius=0.1,
        )
        path = planner.plan(start_xy, goal_xy)  # -> list of (x, y) in metres
    """

    # 8-connected grid moves
    _MOVES = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        resolution: float,
        obstacles: Iterable[Tuple[Point, Point]],
        robot_radius: float = 0.1,
    ):
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.res = resolution
        self.robot_radius = robot_radius
        self.width = int(math.ceil((self.xmax - self.xmin) / self.res))
        self.height = int(math.ceil((self.ymax - self.ymin) / self.res))
        self.occupancy = [[False] * self.height for _ in range(self.width)]
        for (p1, p2) in obstacles:
            self._stamp_obstacle(p1, p2)

    # ---------- Map construction --------------------------------------------

    def _stamp_obstacle(self, p1: Point, p2: Point) -> None:
        """Mark the grid cells covered by an axis-aligned box, inflated by
        the robot radius so returned paths stay clear."""
        x1, y1 = p1
        x2, y2 = p2
        xmin = min(x1, x2) - self.robot_radius
        xmax = max(x1, x2) + self.robot_radius
        ymin = min(y1, y2) - self.robot_radius
        ymax = max(y1, y2) + self.robot_radius
        cxmin, cymin = self._world_to_cell((xmin, ymin))
        cxmax, cymax = self._world_to_cell((xmax, ymax))
        for cx in range(max(0, cxmin), min(self.width, cxmax + 1)):
            for cy in range(max(0, cymin), min(self.height, cymax + 1)):
                self.occupancy[cx][cy] = True

    def stamp_point_obstacle(self, xy: Point, radius: float = 0.08) -> None:
        """Temporarily mark a circular area as blocked (e.g. a cargo item)."""
        r = radius + self.robot_radius
        cx, cy = self._world_to_cell(xy)
        r_cells = int(math.ceil(r / self.res))
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                nx, ny = cx + dx, cy + dy
                if self._in_bounds((nx, ny)):
                    wx, wy = self._cell_to_world((nx, ny))
                    if math.hypot(wx - xy[0], wy - xy[1]) <= r:
                        self.occupancy[nx][ny] = True

    def clear_point_obstacle(self, xy: Point, radius: float = 0.08) -> None:
        """Remove a previously stamped circular obstacle."""
        r = radius + self.robot_radius
        cx, cy = self._world_to_cell(xy)
        r_cells = int(math.ceil(r / self.res))
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                nx, ny = cx + dx, cy + dy
                if self._in_bounds((nx, ny)):
                    wx, wy = self._cell_to_world((nx, ny))
                    if math.hypot(wx - xy[0], wy - xy[1]) <= r:
                        self.occupancy[nx][ny] = False

    def _world_to_cell(self, p: Point) -> Cell:
        cx = int(round((p[0] - self.xmin) / self.res))
        cy = int(round((p[1] - self.ymin) / self.res))
        return cx, cy

    def _cell_to_world(self, c: Cell) -> Point:
        x = self.xmin + c[0] * self.res
        y = self.ymin + c[1] * self.res
        return x, y

    def _in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height

    def _free(self, c: Cell) -> bool:
        return self._in_bounds(c) and not self.occupancy[c[0]][c[1]]

    # ---------- A* search ---------------------------------------------------

    @staticmethod
    def _heuristic(a: Cell, b: Cell) -> float:
        """Octile distance - admissible for 8-connected grids with sqrt(2) diagonals."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    def plan(self, start: Point, goal: Point) -> List[Point]:
        """
        Run A* from start to goal (both in world coordinates, metres).

        Returns a list of (x, y) waypoints including the goal, or an empty
        list if no path exists.
        """
        start_cell = self._world_to_cell(start)
        goal_cell = self._world_to_cell(goal)

        # Nudge start/goal if they landed on an obstacle cell (e.g. because
        # the robot is inside the inflated bounding box of a nearby item).
        start_cell = self._nearest_free_cell(start_cell)
        goal_cell = self._nearest_free_cell(goal_cell)

        if start_cell is None or goal_cell is None:
            return []
        if start_cell == goal_cell:
            return [goal]

        open_heap: List[Tuple[float, Cell]] = []
        heapq.heappush(open_heap, (0.0, start_cell))

        came_from = {start_cell: None}
        g_score = {start_cell: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal_cell:
                return self._reconstruct(came_from, current)
            for dx, dy, step_cost in self._MOVES:
                neighbour = (current[0] + dx, current[1] + dy)
                if not self._free(neighbour):
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    # f(n) = g(n) + h(n)
                    f = tentative_g + self._heuristic(neighbour, goal_cell)
                    heapq.heappush(open_heap, (f, neighbour))
        return []

    def _nearest_free_cell(self, cell: Cell) -> Cell:
        """If `cell` is blocked, spiral outward until we hit a free one."""
        if self._free(cell):
            return cell
        for r in range(1, max(self.width, self.height)):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if self._free(candidate):
                        return candidate
        return None

    def _reconstruct(self, came_from, end: Cell) -> List[Point]:
        path_cells = [end]
        cur = end
        while came_from[cur] is not None:
            cur = came_from[cur]
            path_cells.append(cur)
        path_cells.reverse()
        return [self._cell_to_world(c) for c in path_cells]
