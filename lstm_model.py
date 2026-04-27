import torch
import torch.nn as nn

class PanicLSTM(nn.Module):

    def __init__(self,input_size=4,hidden=32):
        super().__init__()

        self.lstm = nn.LSTM(input_size,hidden,batch_first=True)
        self.fc = nn.Linear(hidden,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)

        return self.sigmoid(out)