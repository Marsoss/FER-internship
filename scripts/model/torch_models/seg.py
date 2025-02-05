import torch.nn as nn
import torch
from .segmentation import segmentation

class ConvVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvVGGBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x    

class VGG_mask(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, init_weights=True):
        super(VGG_mask, self).__init__()
      
        self.block1 = ConvVGGBlock(in_channels, 64, kernel_size=3, padding=1)
        self.block2 = ConvVGGBlock(64, 64, kernel_size=3, padding=1)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block3 = ConvVGGBlock(64, 128, kernel_size=3, padding=1)
        self.block4 = ConvVGGBlock(128, 128, kernel_size=3, padding=1)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block5 = ConvVGGBlock(128, 256, kernel_size=3, padding=1)
        self.block6 = ConvVGGBlock(256, 256, kernel_size=3, padding=1)
        self.block7 = ConvVGGBlock(256, 256, kernel_size=3, padding=1)
        self.block8 = ConvVGGBlock(256, 256, kernel_size=3, padding=1)
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block9 = ConvVGGBlock(256, 512, kernel_size=3, padding=1)
        self.block10 = ConvVGGBlock(512, 512, kernel_size=3, padding=1)
        self.block11 = ConvVGGBlock(512, 512, kernel_size=3, padding=1)
        self.block12 = ConvVGGBlock(512, 512, kernel_size=3, padding=1)
        self.maxp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block13 = ConvVGGBlock(512, 512, kernel_size=3, padding=1)
        self.block14 = ConvVGGBlock(512, 512, kernel_size=3, padding=1)
        self.block15 = ConvVGGBlock(512, 512, kernel_size=3, padding=1)
        self.block16 = ConvVGGBlock(512, 512, kernel_size=3, padding=1)
        self.maxp5 = nn.MaxPool2d(kernel_size=2, stride=2)
    

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 7),
        )

        self.seg1 = segmentation(64, 64, depth=1)
        self.seg2 = segmentation(128, 128, depth=1)
        self.seg3 = segmentation(256, 256, depth=1)
        self.seg4 = segmentation(512, 512, depth=1)
        

    def forward(self, x):

        x = self.block1(x)      
        x = self.block2(x)
        m = self.seg1(x)
        x = x * (1 + m)
        x = self.maxp1(x)
        x = self.block3(x)
        x = self.block4(x)
        m = self.seg2(x)
        x = x * (1 + m)
        x = self.maxp2(x)
        x = self.block5(x)
        x = self.block6(x)        
        x = self.block7(x)
        x = self.block8(x)
        m = self.seg3(x)
        x = x * (1 + m)
        x = self.maxp3(x)
        x = self.block9(x)
        x = self.block10(x)        
        x = self.block11(x)
        x = self.block12(x)
        x = self.maxp4(x)
        x = self.block13(x)
        x = self.block14(x)        
        x = self.block15(x)
        x = self.block16(x)
        m = self.seg4(x)
        x = x * (1 + m)
        x = self.maxp5(x)

  
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg19_bn_mask(in_channels:int=3, num_classes:int=7, progress=True, **kwargs):  

    model = VGG_mask(in_channels, num_classes, progress, **kwargs)      
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model

