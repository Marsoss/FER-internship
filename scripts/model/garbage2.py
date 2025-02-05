import torch 

def get_device():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print(f"    {i}:", torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device

#get_device()

import os
import shutil
import random
import math

def split_dataset(source_folder, train_folder, test_folder, split_ratio=0.85):
    # Create train and test directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get a list of class subfolders in the source folder
    class_subfolders = [subfolder for subfolder in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, subfolder))]

    for subfolder in class_subfolders:
        # Get the list of files in the current class subfolder
        files = os.listdir(os.path.join(source_folder, subfolder))
        random.shuffle(files)

        # Calculate the split index based on the split_ratio
        split_index = math.ceil(len(files) * split_ratio)

        # Split the files into train and test sets
        train_files = files[:split_index]
        test_files = files[split_index:]

        # Create class subfolders in train and test folders
        train_subfolder_path = os.path.join(train_folder, subfolder)
        test_subfolder_path = os.path.join(test_folder, subfolder)

        os.makedirs(train_subfolder_path, exist_ok=True)
        os.makedirs(test_subfolder_path, exist_ok=True)

        # Copy files to the respective train and test subfolders
        for train_file in train_files:
            src_path = os.path.join(source_folder, subfolder, train_file)
            dst_path = os.path.join(train_subfolder_path, train_file)
            shutil.copy(src_path, dst_path)

        for test_file in test_files:
            src_path = os.path.join(source_folder, subfolder, test_file)
            dst_path = os.path.join(test_subfolder_path, test_file)
            shutil.copy(src_path, dst_path)

def split():
    # Modify these paths according to your dataset structure
    source_folder = "../datasets/affectnet"
    train_folder = "../datasets/affecnet/train"
    test_folder = "../datasets/affecnet/test"

    split_ratio = 0.85  # Change this if you want a different split ratio

    split_dataset(source_folder, train_folder, test_folder, split_ratio)

import matplotlib.pyplot as plt

x=[128,256,512,1024,2048,4096]
x2 = [7,8,9,10,11,12]
y=[0.549,0.597,0.64,0.649,0.649,0.643]
plt.plot(x2,y)
plt.xlabel('log2(Nb of Neurons)')
plt.ylabel('Accuracy')
plt.title("Accuracy of Default model according to the size of the dense layer")
plt.show()