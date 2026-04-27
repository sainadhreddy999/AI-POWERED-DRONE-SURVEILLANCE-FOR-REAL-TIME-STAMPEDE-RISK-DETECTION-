import heapq
import numpy as np

# ---------------- HEURISTIC ---------------- #
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------- TIME-AWARE A* ---------------- #
def astar_with_time(start, goal, grid, reserved):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start, 0))  # (f_score, position, time)

    came_from = {}
    g_score = {(start, 0): 0}

    directions = [(1,0), (-1,0), (0,1), (0,-1), (0,0)]  # include WAIT

    while open_set:
        _, current, t = heapq.heappop(open_set)

        if current == goal:
            path = []
            state = (current, t)
            while state in came_from:
                path.append(state[0])
                state = came_from[state]
            path.append(start)
            return path[::-1]

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            nt = t + 1

            if 0 <= nx < rows and 0 <= ny < cols:

                # obstacle check
                if grid[nx][ny] >= 999:
                    continue

                # collision check (time-space)
                if (nx, ny, nt) in reserved:
                    continue

                neighbor = (nx, ny)
                new_cost = g_score[(current, t)] + grid[nx][ny]

                state = (neighbor, nt)

                if state not in g_score or new_cost < g_score[state]:
                    g_score[state] = new_cost
                    f_score = new_cost + heuristic(neighbor, goal)

                    heapq.heappush(open_set, (f_score, neighbor, nt))
                    came_from[state] = (current, t)

    return []


# ---------------- MULTI AGENT (PRIORITIZED) ---------------- #
def compute_paths(objects, density_grid, frame_h, frame_w, GRID_ROWS, GRID_COLS):

    density_np = np.array(density_grid, dtype=np.float32)

    # cost map
    cost_map = 1 + density_np

    # mark obstacles (very high density = blocked)
    cost_map[density_np > 15] = 999

    exits = [
        (0, 0),
        (0, GRID_COLS - 1),
        (GRID_ROWS - 1, 0),
        (GRID_ROWS - 1, GRID_COLS - 1)
    ]

    cell_h = frame_h // GRID_ROWS
    cell_w = frame_w // GRID_COLS

    # reservation table (x, y, time)
    reserved = set()

    paths = {}

    # sort agents → high density areas first (optional improvement)
    sorted_agents = list(objects.items())

    for obj_id, (cx, cy) in sorted_agents:

        gx = int(cy // cell_h)
        gy = int(cx // cell_w)

        start = (gx, gy)

        best_path = None
        best_len = float('inf')

        for exit_pt in exits:
            path = astar_with_time(start, exit_pt, cost_map, reserved)

            if path and len(path) < best_len:
                best_path = path
                best_len = len(path)

        paths[obj_id] = best_path

        # reserve path (collision avoidance)
        if best_path:
            for t, (x, y) in enumerate(best_path):
                reserved.add((x, y, t))

    return paths