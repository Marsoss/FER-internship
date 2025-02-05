# Facial Emotion Recognition internship 2A



## Description

This work is the result of a research study about FER (Facial Emotion Recognition). The aim was to compare different machine learning models and potentially improve the state-of-the-art results. You can find a trainer class for Convolutionnal Neural Networks (CNN) and another for traditionnal machine learning classifier as SVM or KNN. They both have their own testing class computing the confusion matrix and other important metrics as precision, recall or F1-score. The dataset used is FER-2013 and preprocessing functions are available to be applied to an entire folder. You will also find a camera file, so you can use the camera of your own device for testing. 

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Installation

Make sure you have every library needed to run this project:

```
python3 -m pip install --upgrade pip
pip install --upgrade pathlib typing Pillow tk
pip install --upgrade numpy matplotlib tqdm
pip install --upgarde torch torchvision sklearn
pip install --upgarde segmentation linformer vit_pytorch 
pip install --upgrade opencv-python

```

## Usage

For CNN training make you modified the python script ``scripts/models/model_trainer_torch.py`` with the model you want to train and the dataset you want your model to train on:
```
python3 scripts/models/model_trainer_torch.py

```

## Support

If you have any questions, feel free to contact at marceau.combet@ecole.ensicaen.fr


## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.
