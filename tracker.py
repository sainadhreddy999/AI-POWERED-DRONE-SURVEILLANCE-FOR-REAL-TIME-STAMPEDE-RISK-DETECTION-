import math

class Tracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}
        self.history = {}
        self.max_history = 10

    def update(self, detections):
        new_objects = {}

        for box in detections:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            matched_id = None
            min_dist = float('inf')

            for obj_id, (px, py) in self.objects.items():
                dist = math.hypot(cx - px, cy - py)
                if dist < 50 and dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            new_objects[matched_id] = (cx, cy)

            if matched_id not in self.history:
                self.history[matched_id] = []
            self.history[matched_id].append((cx, cy))

        self.objects = new_objects
        return self.objects, self.history