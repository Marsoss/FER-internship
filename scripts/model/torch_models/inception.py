import torch.nn as nn
import torch
import torchvision.models as models

class Inception(torch.nn.Module):
    def __init__(self, num_classes:int=7):
        super(Inception, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        for param in self.inception.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer with a new one
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # forward pass through the layers
        x = self.inception(x)
        # add your own forward pass here to complete the architecture
        return x
