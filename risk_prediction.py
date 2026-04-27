import numpy as np

class RiskPredictor:

    def __init__(self):
        self.history = []

    def update(self, density):
        self.history.append(density)

        if len(self.history) > 30:
            self.history.pop(0)

    def predict(self):
        if len(self.history) < 5:
            return "Insufficient Data"

        trend = np.polyfit(range(len(self.history)), self.history, 1)[0]

        if trend > 0.2:
            return "Future Risk Increasing"
        elif trend < -0.2:
            return "Crowd Dispersing"
        else:
            return "Stable Crowd"