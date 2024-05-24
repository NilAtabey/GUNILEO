import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn

class LabialCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 5, 3), # 1 is the input (because it's a gray-scale image),
                                # 5 is the output, 3 is the kernel size
            nn.ReLU(),
            nn.Conv2d(5, 10, 3),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(24 * 24 * 10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

    def forward(self, x, debug=False):
        x = self.cnn(x)
        if debug: print(x.shape)
        x = torch.flatten(x, 1)
        if debug: print(x.shape)