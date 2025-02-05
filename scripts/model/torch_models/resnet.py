import torch.nn as nn
import torch
import torchvision.models as models

class ResNet(torch.nn.Module):
    def __init__(self, num_classes:int=7):
        super(ResNet, self).__init__()
        self.resnet =  models.resnet50(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer with a new one
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # forward pass through the layers
        x = self.resnet(x)
        # add your own forward pass here to complete the architecture
        return x