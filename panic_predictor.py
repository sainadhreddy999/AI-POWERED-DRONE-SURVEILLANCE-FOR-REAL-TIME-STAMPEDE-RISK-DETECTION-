import torch
import numpy as np
from lstm_model import PanicLSTM

model = PanicLSTM()
model.eval()

sequence = []

def predict_panic(features):
    global sequence

    sequence.append(features)

    if len(sequence) > 20:
        sequence.pop(0)

    if len(sequence) < 20:
        return 0

    seq = torch.tensor(sequence).float().unsqueeze(0)

    with torch.no_grad():
        prob = model(seq).item()

    return prob
