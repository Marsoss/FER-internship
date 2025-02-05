import torch.nn as nn

class SotaConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SotaConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x

class SotA(nn.Module):
    def __init__(self, num_channels:int=1, num_classes:int=7, width:int=300, height:int=300):
        super(SotA, self).__init__()
        
        self.conv_layers = nn.Sequential(
            SotaConvBlock(num_channels, 32),
            SotaConvBlock(32,64),
            SotaConvBlock(64,128),
            SotaConvBlock(128,256)
        )
    
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (width//16) * (height//16), 512),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
