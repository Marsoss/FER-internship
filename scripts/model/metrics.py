import itertools
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt

def accuracy(labels, predictions) -> float:
    # Calculate accuracy
    accuracy = metrics.accuracy_score(labels, predictions)
    print(f"Accuracy: {accuracy}")
    return accuracy

def f1_score(labels, predictions) -> float:
    f1 = metrics.f1_score(labels, predictions, average='macro')
    print("F1 score:", f1)
    return f1

def conf_matrix(labels, predictions) -> np.ndarray:
    conf_matrix = metrics.confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(conf_matrix)
    return conf_matrix
    
def plot_confusion_matrix(labels, predictions, class_names, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues) -> None:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = conf_matrix(labels, predictions)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.plot()
    plt.show()

def precision(labels, predictions) -> float:
    precision = metrics.precision_score(labels, predictions, average='macro')
    print("Precision:", precision)
    return precision

def recall_score(labels, predictions) -> float:
    recall = metrics.recall_score(labels, predictions, average='macro')
    print("Recall:", recall)
    return recall