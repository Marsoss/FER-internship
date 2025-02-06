import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from metrics import accuracy, recall_score, f1_score, plot_confusion_matrix, conf_matrix, precision

import numpy as np

import joblib

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize the images to a fixed size
    transforms.Grayscale(), # Convert images to grayscale
    transforms.ToTensor()  # Convert images to tensors
])

class ClassifierTrainer:
    def __init__(self, classifier=None, train_folder_path:str=None, ) -> None:
        
        if train_folder_path is not None:
            self.set_train_data(train_folder_path=train_folder_path)

        if classifier is not None:
            self.__classifier = classifier        
            

    def set_train_data(self,  train_folder_path:str):
        # Create the ImageFolder dataset
        self.__train_dataset = ImageFolder(root=train_folder_path, transform=transform)

        # Create the dataloader
        self.__train_dataloader = torch.utils.data.DataLoader(self.__train_dataset, batch_size=32, shuffle=True, num_workers=4)


        # Extract the features and labels from the dataloader
        self.__features = []
        self.__labels = []
        for images, target in self.__train_dataloader:
            self.__features.append(images.numpy())
            self.__labels.append(target.numpy())

        # Convert the features and labels to numpy arrays
        self.__features = np.concatenate(self.__features, axis=0)
        self.__features = self.__features.reshape(self.__features.shape[0], -1)
        self.__labels = np.concatenate(self.__labels, axis=0)


    def train(self, classifier=None) -> None:
        if classifier is not None:
            self.__classifier = classifier
        if self.__classifier is not None:
            self.__classifier.fit(self.__features, self.__labels)

    def train_many(self, classifier_list):
        for classifier, save_name in classifier_list:
            print(save_name)
            self.train(classifier)
            self.save(save_name=save_name)
    
    def save(self, save_name:str='classifier.pkl') -> None:
        joblib.dump(self.__classifier, save_name)
    
    def load_classifier(self, classifier:str) -> None:
        self.__classifier = joblib.load(classifier)

class ClassifierTester:
    def __init__(self, test_folder_path:str=None, classifier:str=None) -> None:
        
        if test_folder_path is not None:
            self.set_test_data(test_folder_path=test_folder_path)

        if classifier is not None:
            self.load_classifier(classifier)
            
        self.__class_names = ['angry','disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def set_test_data(self,  test_folder_path:str):
        # Create the ImageFolder dataset
        self.__test_dataset = ImageFolder(root=test_folder_path, transform=transform)

        # Create the dataloader
        self.__test_dataloader = torch.utils.data.DataLoader(self.__test_dataset, batch_size=32, shuffle=True, num_workers=4)


        # Extract the features and labels from the dataloader
        self.__features = []
        self.__labels = []
        for images, target in self.__test_dataloader:
            self.__features.append(images.numpy())
            self.__labels.append(target.numpy())

        # Convert the features and labels to numpy arrays
        self.__features = np.concatenate(self.__features, axis=0)
        self.__features = self.__features.reshape(self.__features.shape[0], -1)
        self.__labels = np.concatenate(self.__labels, axis=0)

    def load_classifier(self, classifier:str):
        self.__classifier = joblib.load(classifier)

    def predict(self):
        self.__predictions = self.__classifier.predict(self.__features)

    def accuracy(self) -> float:
        return accuracy(self.__labels, self.__predictions)

    def f1_score(self) -> float:
        return f1_score(self.__labels, self.__predictions)
    
    def conf_matrix(self) -> np.ndarray:
        return conf_matrix(self.__labels, self.__predictions)
    
    def plot_confusion_matrix(self, normalize=False) -> None:
        plot_confusion_matrix(self.__labels, self.__predictions, self.__class_names, normalize=normalize)
    
    def precision(self)->float:
        return precision(self.__labels, self.__predictions)
    
    def recall_score(self)->float:
        return recall_score(self.__labels, self.__predictions)
    
    def eval_many(self, classifier_list):
        for _, save_name in classifier_list:
            print(save_name)
            self.load_classifier(save_name)
            self.predict()
            self.accuracy()
            self.precision()
            self.recall_score()
            self.f1_score()
            self.plot_confusion_matrix(normalize=True)
        

def main_classifier():
    dataset = '../datasets/affecnet'
    train_data_path = f'{dataset}/train'
    verbose = True
    save_dir = "models_aff"
    classifiers = [(svm.SVC(decision_function_shape='ovr', verbose=verbose), f"{save_dir}/svm.pkl"),
                    (RandomForestClassifier(n_estimators=100, verbose=verbose),  f"{save_dir}/RandomForest.pkl"),
                    (KNeighborsClassifier(n_neighbors=5), f"{save_dir}/Kneighbors5.pkl"),
                    (DecisionTreeClassifier(), f"{save_dir}/DecisionTree.pkl"),
                    (GaussianNB(), f"{save_dir}/NaiveBayesian.pkl"),
                    (GradientBoostingClassifier(verbose=verbose), f"{save_dir}/GBoosting.pkl")]

    classifiers = classifiers[0:1]
    
    trainer = ClassifierTrainer(train_folder_path=train_data_path)
    print("start training")
    trainer.train_many(classifier_list=classifiers)


    test_data_path = f'{dataset}/test'
    tester = ClassifierTester(test_folder_path=test_data_path)
    print("start evaluation")
    tester.eval_many(classifier_list=classifiers)


main_classifier()