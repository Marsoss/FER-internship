# Utils for model training
import copy
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F

def polar_transform(images):
    # Convert images to polar coordinates
    batch_size, num_channels, height, width = images.size()

    # Generate grid of polar coordinates
    theta = torch.linspace(0, 2 * 3.14159, width).to(images.device)
    radius = torch.linspace(0, 1, height).to(images.device)
    theta, radius = torch.meshgrid(theta, radius)
    theta = theta.expand(batch_size, -1, -1)
    radius = radius.expand(batch_size, -1, -1)

    # Convert polar coordinates to Cartesian coordinates
    x_cartesian = radius * torch.cos(theta)
    y_cartesian = radius * torch.sin(theta)

    # Rescale Cartesian coordinates to image size
    x_cartesian = (x_cartesian + 1) * (width - 1) / 2
    y_cartesian = (y_cartesian + 1) * (height - 1) / 2

    # Interpolate pixel values using bilinear sampling
    grid = torch.stack([x_cartesian, y_cartesian], dim=3)
    return F.grid_sample(images, grid, mode='bilinear', align_corners=False)


def random_transform(num_channels:int=1, img_size=(48,48)):
    if num_channels==1:
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.1,0.1)),
                transforms.ColorJitter(brightness=0.1),
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor()
            ])
    else:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.1,0.1)),
                transforms.ColorJitter(brightness=0.1),
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor()
            ])


class EarlyStopping():
    def __init__(self, patience:int=5, min_delta:int=0, restore_best_weights:bool=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = "init"

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = f"Stopped on {self.counter}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter}/{self.patience}"
        return False

class LearningRateScheduler():
    def __init__(self, patience:int=5, min_delta:int=0, reduction_rate=0.5):
        self.patience = patience
        self.min_delta = min_delta
        self.reduction_rate = reduction_rate
        self.best_loss = None
        self.counter = 0
        self.status = "init"

    def __call__(self, model, optimizer, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            model.save("best.pth")
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                optimizer.param_groups[0]['lr'] *= self.reduction_rate 
                self.counter = 0

        self.status = f"LRS: lr = {optimizer.param_groups[0]['lr']} [{self.counter}/{self.patience}]"
        return optimizer
    

