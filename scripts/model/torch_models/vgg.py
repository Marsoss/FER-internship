import torch.nn as nn
import torch
import torchvision.models as models


class VGG(torch.nn.Module):
    def __init__(self, num_classes:int=7):
        super(VGG, self).__init__()
        self.vgg =  models.vgg16(pretrained=True)
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer with a new one
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # forward pass through the layers
        x = self.vgg(x)
        # add your own forward pass here to complete the architecture
        return x
        
