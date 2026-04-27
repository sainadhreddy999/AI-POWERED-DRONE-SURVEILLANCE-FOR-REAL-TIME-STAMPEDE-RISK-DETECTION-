import math
import numpy as np

def compute_velocity(history):
    velocities = {}
    for obj_id, points in history.items():
        if len(points) >= 2:
            (x1, y1), (x2, y2) = points[-2], points[-1]
            velocities[obj_id] = math.hypot(x2 - x1, y2 - y1)
        else:
            velocities[obj_id] = 0
    return velocities

def compute_direction(history):
    directions = {}
    for obj_id, points in history.items():
        if len(points) >= 2:
            (x1, y1), (x2, y2) = points[-2], points[-1]
            directions[obj_id] = math.atan2(y2 - y1, x2 - x1)
        else:
            directions[obj_id] = 0
    return directions

def detect_abnormal_behavior(velocities, directions):
    speeds = list(velocities.values())
    dirs = list(directions.values())

    if len(speeds) == 0:
        return "No Data"

    if np.var(speeds) > 50:
        return "PANIC"
    elif len(dirs) > 1 and (max(dirs) - min(dirs)) > 2:
        return "CONFLICT"
    return "NORMAL"

if __name__ == "__main__":
    # Test mock data
    mock_history = {
        1: [(0, 0), (1, 1), (2, 2)],
        2: [(10, 10), (11, 12)]
    }
    
    velocities = compute_velocity(mock_history)
    directions = compute_direction(mock_history)
    behavior = detect_abnormal_behavior(velocities, directions)
    
    print("Velocities:", velocities)
    print("Directions:", directions)
    print("Behavior Status:", behavior)