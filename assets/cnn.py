import torch
from torch import nn

class LabialCNN(nn.Module):
    def __init__(self, debug: bool = False):
        super().__init__()

        self.debug = debug

        self.model = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=1,
                kernel_size=(3, 5, 5),
                padding=(1, 2, 2),
                stride=(1, 2, 2)
            ),
            nn.MaxPool3d(
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2)
            ),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), padding=(1, 2, 2)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Flatten(),    # Left as default, check later if it causes problems
            #nn.LSTM(input_size=1, dropout=0.5, bidirectional=True),
        )
        

    # Remember to put FALSE
    def forward(self, x):
        x = self.model(x)   # Run through the model
        if self.debug: print(f"  Layer's shape: {x.shape}")
        #x = torch.flatten(x, 1)     # Flatten layer
        #if debug: print(f"  Layer's shape: {x.shape}")
        if self.debug: print(f"Summary of the layer: {x.shape}")