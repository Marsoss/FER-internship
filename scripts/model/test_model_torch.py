import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn

from PIL import Image

import random
import os
from typing import List

from metrics import recall_score, precision, conf_matrix, plot_confusion_matrix, f1_score, accuracy

import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt
import itertools

from tqdm import tqdm

from model import create_model

def get_num_channel(color_mode:str, model_name:str) -> int:
    return 1 if color_mode != 'rgb' and model_name.lower() not in ['vgg', 'inception', 'resnet'] else 3

def choose_filename(file_filter) -> str:
    root = tk.Tk()
    root.withdraw()
    current_dir = os.getcwd()
    file_path = filedialog.askopenfilename(initialdir=current_dir, filetypes=file_filter)
    return os.path.relpath(file_path, current_dir)

def create_preprocess_function(height:int, width:int, color_mode:str):
    size = (height, width)
    if color_mode == 'rgb':
        return  transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ])
    else:
        return  transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ])

def get_classes(num_classes:int) -> List[str]:
    if num_classes==7:
        return ['angry','disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    else:
        return ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def num_classes_dataset(dataset:str) -> int:
    return len(os.listdir(dataset))
class CNNTester:
    def __init__(self, model_name:str, dataset:str="FER2013", tr_height:int=48, tr_width:int=48, model_color_mode:str='rgb') -> None:
        
        # Classes
        self.__dataset:str = f'../datasets/{dataset}/test'
        self.__num_classes:int = num_classes_dataset(self.__dataset)

        #Model
        if not model_name:
            self.choose_model()
        else:
            self.__model_name:str = model_name
        self.__batch_size:int = 16
        self.__tr_height:int = tr_height
        self.__tr_width:int =  tr_width
        self.__tr_size = self.__tr_width, self.__tr_height
        self.__num_channels:int = get_num_channel(color_mode=model_color_mode, model_name=model_name)
        self.__model:nn.Module = create_model(model_name=model_name, height=tr_height, width=tr_width, num_channels=self.__num_channels, num_classes=self.__num_classes)
        print(type(self.__model))

        #Data
        if self.__num_classes != self.__model.classifier[-1].out_features: # TODO: Verify classifier[-1] on other models
            raise ValueError("Number of classes in dataset does not match model output")
        self.__class_names:List[str] = get_classes(self.__num_classes)
        
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #One Test
        self.__image_name:str = ''
        self.__emotion:str
        
        self.__preprocess:transforms.Compose =  create_preprocess_function(tr_height, tr_width, model_color_mode)

    def __get_random_img(self) -> str:
        emotion = input(f"Give emotion among {self.__class_names}")
        if emotion not in self.__class_names:
            emotion = random.choice(self.__class_names)

        folder_path = f'{self.__dataset}/{emotion}'

        file_list = os.listdir(folder_path)

        random_file = random.choice(file_list)
        print(f'{folder_path}/{random_file}')
        self.__image_name = f'{folder_path}/{random_file}'
        return emotion

    def predict(self) -> str:
        # Tensor transformation
        image_tensor = self.__preprocess(self.__image) # convert image to tensor
        image_tensor = image_tensor.unsqueeze_(0) # add a batch dimension
        image_tensor = image_tensor[:,:3,:,:]

        # Make a prediction
        output = self.__model(image_tensor)
        print(nn.functional.softmax(output,dim=1))
        _, predicted = torch.max(output, 1)

        return self.__class_names[predicted.item()]
    
    def load_model(self) -> None:
        checkpoint = torch.load(self.__model_name)
        state_dict = checkpoint
        print("loading dict")
        self.__model.load_state_dict(state_dict)

    def get_emotion(self) -> None:
        self.__emotion = input("Give emotion represented by image: ")
                
    def set_image(self, image_path:str, emotion_label:str) -> None:
        self.__image = Image.open(image_path)
        self.__image_name = image_path
        self.__emotion = emotion_label

    def __print_prediction(self):
        predicted_class = self.predict()
        if self.__emotion in self.__class_names:
            print("Emotion expected: ", self.__emotion)
        print("The model predicts: ", predicted_class)

    def random_test(self) -> None:
        self.__get_random_img()
        self.__print_prediction()
        self.__image.show()

    def test(self,  image_name:str='', emotion:str='None', show:bool=False) -> None:
        
        if image_name != '':
            self.set_image(image_name, emotion_label=emotion)
        else:    
            self.choose_image()
        if self.__model_name == '':
            self.choose_model()
        self.__print_prediction()
        if show:
            self.__image.show()

    def set_model_name(self, model_name:str) -> None:
        self.__model_name = model_name

    def choose_image(self) -> None:
        print("choose an image file to test")
        file_filter = [("PNG Files", "*.png"), ("JPG Files", "*.jpg")]
        file_path = choose_filename(file_filter)
        self.get_emotion()
        self.set_image(file_path, self.__emotion)    

    def choose_model(self) -> None:
        print("choose a model to test")
        file_filter = [("PTH Files", "*.pth")]
        file_path = choose_filename(file_filter)
        self.__model_name = file_path    

    def load_data(self) -> None:
        print("loading data...")
        self.__data = datasets.ImageFolder(
            root=self.__dataset,
            transform=self.__preprocess
        )
        self.__test_data_loader = DataLoader(self.__data, batch_size=self.__batch_size, shuffle=False)

    def predict_folder(self) -> None:
        print("computing predictions...")
        self.__model.eval()
        self.__model = self.__model.to(self.__device)

        test_labels = []
        predictions = []
        progress_bar = tqdm(total=len(self.__test_data_loader))
        for image, label in self.__test_data_loader:
            image = image.to(self.__device)
            label = label.to(self.__device)
            progress_bar.update(1)
            image = image[:,:3,:,:]
            outputs = self.__model(image)
            _, predicted = torch.max(outputs, dim=1)
            test_labels.extend(label.tolist())
            predictions.extend(predicted.tolist())
        
        progress_bar.close()
        self.__test_labels = torch.tensor(test_labels)
        self.__predictions = torch.tensor(predictions)

    def accuracy(self) -> float:
        return accuracy(self.__test_labels, self.__predictions)

    def f1_score(self) -> float:
        return f1_score(self.__test_labels, self.__predictions)
    
    def conf_matrix(self) -> np.ndarray:
        return conf_matrix(self.__test_labels, self.__predictions)
    
    def plot_confusion_matrix(self, normalize=False) -> None:
        plot_confusion_matrix(self.__test_labels, self.__predictions, self.__class_names, normalize=normalize)
    
    def precision(self)->float:
        return precision(self.__test_labels, self.__predictions)
    
    def recall_score(self)->float:
        return recall_score(self.__test_labels, self.__predictions)
    

def test_model(tester):
    image_name = 'images\me_happy.jpg'
    tester.test(image_name,emotion='happy')

def test_set(tester, matrix_save_name:str='confusion_matrix.png'):
    tester.load_data()
    tester.load_model()

    tester.predict_folder()
    tester.accuracy()
    tester.f1_score()
    tester.precision()
    tester.recall_score()
    plt.figure()
    tester.plot_confusion_matrix(normalize=True)
    plt.show()
    

def main_test():
    model_name = 'models/unet1_FER2013.pth'

    tester = CNNTester(model_name=model_name,  
                    tr_height=48, 
                    tr_width=48,
                    model_color_mode='grayscale')
    
    tester.load_model()
    
    test_set(tester, 'results/Unet1_matrix.png')

if __name__=='__main__':
    main_test()