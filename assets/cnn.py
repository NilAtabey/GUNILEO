import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn

class LabialCNN(nn.Module, debug=False):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), padding=(1, 2, 2)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

    # Remember to put FALSE
    def forward(self, x, debug):
        x = self.cnn(x)
        if debug: print(f"  Layer's shape: {x.shape}")
        x = torch.flatten(x, 1)
        if debug: print(f"  Layer's shape: {x.shape}")