import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

class FolderSplitter:
    def __init__(self, data_folder, validation_split:float=0.15, batch_size:float=32, random_seed:int=42):
        self.data_folder = data_folder
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.random_seed = random_seed

    def get_data_loader(self, transform)->DataLoader:

        # Create ImageFolder dataset
        data = datasets.ImageFolder(root=self.data_folder, transform=transform)

        # Create DataLoader object
        return DataLoader(data, batch_size=self.batch_size, shuffle=True)


    def split_folders(self, transform):
        # Set random seed
        torch.manual_seed(self.random_seed)
        generator = torch.Generator().manual_seed(self.random_seed)

        # Create ImageFolder dataset
        data = datasets.ImageFolder(root=self.data_folder, transform=transform)

        # Split dataset into train and validation
        train_size = int((1 - self.validation_split) * len(data))
        val_size = len(data) - train_size
        train_data, val_data = random_split(data, [train_size, val_size], generator=generator)

        # Create DataLoader objects
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader
