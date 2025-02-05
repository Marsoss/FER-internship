from pathlib import Path
from PIL import Image
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary


from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from data_loader import FolderSplitter
from training_utils import random_transform

from model import create_model

def get_folder(folder:Path, subfolder_name:str) -> Path:
    new_path = Path(folder, subfolder_name)
    if not new_path.exists():
        raise FileNotFoundError(f'No folder named {subfolder_name} in {Path(folder).name}')
    return new_path
        
def get_image_size(folder_path:Path) -> Tuple[int,int]:
    first_dir = next(folder_path.iterdir())
    if first_dir.is_dir():
        first_image_path =  next(first_dir.iterdir())
        if first_image_path.is_file():
            with Image.open(first_image_path) as img:
                return img.size
    return 0,0

def get_class_names(folder_path:Path) -> List[str]:
    return [folder.name for folder in folder_path.iterdir() if folder.is_dir()]

def print_classes(classes_list:List[str]) -> None:
    for i in range(len(classes_list)):
        print(f'class {i}: {classes_list[i]}')

def get_num_channel(color_mode:str, model_name:str) -> int:
    return 1 if color_mode != 'rgb' and model_name.lower() not in ['vgg', 'inception', 'resnet', 'deeplab'] else 3

class CNNTrainer:
    def __init__(self, data_folder:Path, model_name:str="Default", color_mode:str='rgb') -> None:

        #Folders
        self.__data_folder:Path = data_folder

        try:
            self.__train_folder:Path = get_folder(self.__data_folder, "train")
        except FileNotFoundError as e:
            print(e)
        try:
            self.__test_folder:Path = get_folder(self.__data_folder, "test")
        except FileNotFoundError as e:
            print(e)

        #Classes
        self.__class_names:List[str] = get_class_names(self.__train_folder)
        self.__num_classes:int = len(self.__class_names)

        #Images properties
        self.__image_size:Tuple[int,int] = get_image_size(self.__train_folder)
        self.__image_width:int = self.__image_size[0]
        self.__image_height:int = self.__image_size[1]
        self.__num_channels:int = get_num_channel(color_mode, model_name)
        self.__color_mode = color_mode

        #Model
        self.__model_name:str = model_name
        self.__model:nn.Module
        self.__batch_size:int = 32

        #Data 
        self.__train_loader:DataLoader
        self.__val_loader:DataLoader

        #Device
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Control
        self.__data_is_loaded:bool = False
        self.__training_ready:bool = False
        self.__training_done:bool = False

        #Training stats
        self.__t_loss, self.__t_acc, self.__v_loss, self.__v_acc = [], [], [], []

    def print_workspace(self) -> None:
        print(f'train directory : {self.__train_folder}')
        print(f'test directory : {self.__test_folder}')
        
        if self.__num_classes>1:
            print(f'{self.__num_classes} classes found')
            print_classes(self.__class_names)
        elif self.__num_classes==1:
            print("You can't classify with only 1 class. You need at least 2.")
        else:
            print("No classes or subfolder were found")

        if self.__image_height != 0:
            print(f'image width = {self.__image_width} pixels')
            print(f'image height = {self.__image_height} pixels')
        else:
            print("Image size couldn't be determined")
        print(f'color mode : {self.__color_mode}')
        print(f'number of channels: {self.__num_channels}')
        if self.__model_name:
            print(f'model name : {self.__model_name}')
        else:
            print('default model')

        print(f'compute device: {self.__device}')

    def load_data(self) -> None:
        print("loading data...")

        splitter = FolderSplitter(self.__train_folder, batch_size=self.__batch_size)

        transform = random_transform(num_channels=self.__num_channels, img_size=self.__image_size)
        self.__train_loader, self.__val_loader = splitter.split_folders(transform=transform)

        self.__data_is_loaded = True

    def show_sample(self, num_images:int) -> None:
        if self.__data_is_loaded:
            from matplotlib import pyplot as plt
            from math import sqrt

            data_iterator = self.__data.as_numpy_iterator()
            batch = data_iterator.next()

            if sqrt(num_images)>2:
                num_rows = 3
                num_cols = num_images//3 + (num_images%3!=0) 
                div = 3
            elif 2<num_images<=4:
                num_rows = 2
                num_cols = 2
                div = 2
            elif num_images==2:
                num_rows = 1
                num_cols = 2
                fig, ax = plt.subplots(ncols=num_cols, figsize=(10,10))
                for idx, img in enumerate(batch[0][:num_images]):
                    ax[idx].imshow(img.astype(int), cmap='gray')
                    ax[idx].title.set_text(self.__class_names[batch[1][idx]])
                plt.show()
                return
            else:
                num_cols = 1
                num_rows = 1
                div = 1
                fig, ax = plt.subplots(figsize=(10,10))
                img = batch[0][0]
                ax.imshow(img.astype(int), cmap='gray')
                ax.title.set_text(self.__class_names[batch[1][0]])
                plt.show()
                return
        
            
            fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10,10))
            for idx, img in enumerate(batch[0][:num_images]):
                ax[idx%div, idx//div].imshow(img.astype(int), cmap='gray')
                ax[idx%div, idx//div].title.set_text(self.__class_names[batch[1][idx]])
            plt.show()
        else:
            print("You need to load datas first, use method load_data()")

    def set_model_default(self) -> None:
        from torch_models.default import DefaultModel
        self.__model = DefaultModel(self.__num_classes, self.__num_channels, self.__image_width, self.__image_height).to(self.__device)

    def set_model_emotion(self) -> None:
        from torch_models.emotionCNN import EmotionCNN
        self.__model = EmotionCNN(self.__num_classes, self.__num_channels, self.__image_width, self.__image_height)

    def set_model_vgg(self) -> None:
        from torch_models.vgg import VGG
        self.__model = VGG(self.__num_classes).to(self.__device)

    def set_model_resnet(self) -> None:
        from torch_models.resnet import ResNet
        self.__model = ResNet(self.__num_classes).to(self.__device)
    
    def set_model_inception(self) -> None:
        from torch_models.inception import Inception
        self.__model = Inception(self.__num_classes).to(self.__device)
    
    def set_model_sota(self) -> None:
        from torch_models.sota import SotA
        self.__model = SotA(self.__num_channels, self.__num_classes).to(self.__device)

    def  __set_model(self) -> None:
        self.__model = create_model(self.__model_name, 
                                    self.__image_height, 
                                    self.__image_width, 
                                    self.__num_channels, 
                                    self.__num_classes).to(self.__device)

    def set_model(self, new_model:nn.Module) -> None:
        self.__model = new_model.to(self.__device)

    def load_model(self, model_file) -> None:
        checkpoint = torch.load(model_file)
        state_dict = checkpoint
        
        self.__model.load_state_dict(state_dict)
        self.__training_ready = True

    def prepare_training(self) -> None:
        if self.__data_is_loaded:
            self.__set_model()
            self.__training_ready = True
            print("Ready to train")
        else:
            print("You need to load datas first, use method load_data()")

    def __safe_launch(self):
        if not self.__training_ready:
            return False
        go_train = ''
        while go_train.lower() not in ['y', 'n']:
            go_train = input("It might take hours, are you sure you want to start training? Y/N: ")
            if go_train.lower() == 'n':
                print("/!\ Training cancelled /!\ ")
                return False 
            if go_train.lower() == 'y':
                break
            else:
                print("Your answer must be either 'Y' or 'N' ")
        return True

    def get_parameters(self):
        return self.__model.parameters()


    def train(self, criterion=nn.CrossEntropyLoss(), patience:int=3, optimizer:optim=None, epochs:int=30) -> None:
        if not self.__safe_launch():
            return 
        
        from training_utils import LearningRateScheduler, EarlyStopping
        self.__num_epochs = epochs

        # Optimizer
        if optimizer is None:
            optimizer = optim.Adam(self.__model.parameters(), lr=0.001)
        
        es = EarlyStopping(patience=patience)

        lrs = LearningRateScheduler(patience=patience//2 + 1, reduction_rate=0.25)

        self.__epoch = 0
        done = False
        
        while self.__epoch < self.__num_epochs and not done:
            self.__epoch += 1
            train_loss, train_acc = self.__train_one_epoch(criterion, optimizer)
            val_loss, val_acc = self.__validate_one_epoch(criterion)

            self.__t_acc.append(train_acc.item())
            self.__t_loss.append(train_loss)
            self.__v_acc.append(val_acc.item())
            self.__v_loss.append(val_loss)
            
            done =  es(self.__model, val_loss=val_loss)
            optimizer = lrs(self, optimizer, val_loss)
            print(f"Epoch {self.__epoch}/{self.__num_epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f},  val_acc={val_acc:.4f}, EStop:[{es.status}], {lrs.status}")
            
        self.__training_done = True

    def __train_one_epoch(self, criterion: nn.modules.loss, optimizer: optim) -> Tuple[float, float]:
        self.__model.train()
        train_loss = 0
        train_acc = 0
        progress_bar = tqdm(total=len(self.__train_loader))
        for images, labels in self.__train_loader:

            progress_bar.update(1)

            images = images.to(self.__device)
            labels = labels.to(self.__device)
            
            #labels = nn.functional.one_hot(labels, self.__num_classes).float()
            
            # 1. Optimizer zero grad: resets gradients at 0
            optimizer.zero_grad()
            # 2. Forward pass: computes the value with the current parameters
            outputs = self.__model(images)
            if self.__model_name.lower() == 'inception':
                outputs = outputs.logits
            
            # 3. Calculate the loss: difference estimation with the label
           
            loss = criterion(outputs, labels)

            # 4. Backpropagation on the loss with respect to the parameters of the model
            loss.backward()

            # 5. Gradient descent 
            optimizer.step()
            

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            #preds = nn.functional.one_hot(preds, self.__num_classes).float()
            #print(preds)
            #print(labels)
            train_acc += torch.sum(preds == labels.data)
        progress_bar.close()
        train_loss /= len(self.__train_loader) * self.__batch_size
        train_acc = train_acc / (len(self.__train_loader) * self.__batch_size)

        return train_loss, train_acc

    def __validate_one_epoch(self, criterion) -> Tuple[float, float]:
        self.__model.eval()
        val_loss = 0
        val_acc = 0
        progress_bar = tqdm(total=len(self.__val_loader))
        with torch.no_grad():
            for images, labels in self.__val_loader:
                progress_bar.update(1)


                images = images.to(self.__device)
                labels = labels.to(self.__device)
                
                #labels = nn.functional.one_hot(labels, self.__num_classes).float()

                outputs = self.__model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                #preds = nn.functional.one_hot(preds, self.__num_classes).float()
                val_acc += torch.sum(preds == labels.data)
            progress_bar.close()

        val_loss /= len(self.__val_loader) * self.__batch_size
        val_acc = val_acc / (len(self.__val_loader) * self.__batch_size)

        return val_loss, val_acc

    def show_model(self) -> None:
        if self.__training_ready:
            summary(self.__model.to(device=self.__device), (self.__num_channels, self.__image_width, self.__image_height))
        else:
            print('You need to set a model first, use method set_model() or prepare_training()')

    def __show_loss(self, save_name:str):
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(1, self.__epoch+1), np.array(self.__t_loss), label='train loss')
        plt.plot(np.arange(1, self.__epoch+1), np.array(self.__v_loss), label='validation loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.savefig(save_name)
        plt.show()

    def __show_accuracy(self, save_name:str):
        plt.figure(figsize=(10,5))
        plt.plot(np.arange(1, self.__epoch+1), np.array(self.__t_acc), label='training accuracy')
        plt.plot(np.arange(1, self.__epoch+1), np.array(self.__v_acc), label='validation accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.savefig(save_name)
        plt.show()

    def show_plot(self, save_plot_path:Path) -> None:
        if self.__training_done:
            # plot the metrics over time
            self.__show_loss(f"{save_plot_path}_loss.png")
            self.__show_accuracy(f"{save_plot_path}_accuracy.png")
        else:
            print("No plot: You must train the model first")

    def save(self, save_name:str='') -> None:
        if not save_name:
            dataset = Path(self.__data_folder).name
            save_name = f'{self.__model_name}_{dataset}.pth'
        save_path =  Path('models', save_name)
        torch.save(self.__model.state_dict(), save_path)
        print(f'Model saved as { save_path}')


def main_trainer():

    LR = 0.0001
    EPOCHS = 100
    PATIENCE = 5
    
    
    model_name = "unet1"

    dataset_name = "affecnet"

    dataset_path = f"../datasets/{dataset_name}"

    plot_save_name = f"results/{model_name}_{dataset_name}"

    torch.cuda.empty_cache()

    trainer = CNNTrainer(Path(dataset_path), model_name=model_name, color_mode='rgb')

    model_file = f'models\{model_name}_{dataset_name}.pth'

    save_name = f'{model_name}_{dataset_name}.pth'

    num_channels = 1
    num_classes = 8
    width = height = 300

    trainer.print_workspace()
    trainer.load_data()
    trainer.prepare_training()

    trainer.show_model()

    param = trainer.get_parameters()

    optimizer = optim.Adam(param, lr=LR)
    criterion = nn.CrossEntropyLoss()

    trainer.train(patience=PATIENCE, criterion=criterion, optimizer=optimizer, epochs=EPOCHS)
    trainer.save(save_name=save_name)
    trainer.show_plot(Path(plot_save_name))

main_trainer()