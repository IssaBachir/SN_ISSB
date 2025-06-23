import torch.nn as nn


# Definition du reseau de neurones (ANN)
class DiamondModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=input_size, out_features=15)
        self.layer_2 = nn.Linear(in_features=15, out_features=12)
        self.layer_3 = nn.Linear(in_features=12, out_features=8)
        self.layer_4 = nn.Linear(in_features=8, out_features=5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x
