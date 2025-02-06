# Facial Emotion Recognition internship

## 📌Summary

This work is the result of a research study about **FER** (Facial Emotion Recognition). The aim was to compare different machine learning models and potentially improve the state-of-the-art results. You can find a trainer class for **Convolutionnal Neural Networks** (CNN) and another for traditionnal machine learning classifier as SVM or KNN. They both have their own testing class computing the confusion matrix and other important metrics as precision, recall or F1-score. The dataset used is **FER-2013** and preprocessing functions are available to be applied to an entire folder. You will also find a camera file, so you can use the camera of your own device for testing. 

## ⚙️ Installation

### Advice
Create a dedicated environment and activate it
```bash
python -m venv /path/to/new/virtual/environment
.\environment\Scripts\activate  
```

### Prerequisites
Make sure you have Python installed (preferably Python 3.9+):
```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### Required Libraries
The main dependencies include:
- `torch`, `torchvision`, `torchaudio` (for deep learning models)
- `opencv-python` (for camera use)
- `matplotlib` (for visualization)
- `numpy`, `pandas` (for data handling)
- `tqdm` (for training/testing progression)
- `Pillow` (for image processing)

## 📥 Dataset

You can download the dataset used for training and testing from the following link: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
In the folder containing the project, create a `/datasets` folder where you can extract the downloaded files.

## 🏋️ Training the Model

For CNN training make you modified the python script ``scripts/models/model_trainer_torch.py`` with the model you want to train and the dataset you want your model to train on:

```bash
python model_trainer_torch.py --model unet1 --dataset_path ../datasets/FER2013 --lr 0.001 --epochs 30 --patience 5 --optimizer adam --criterion cross_entropy --show True
```

### Training Parameters:
- `--model` : Name of the model from MODELS in `model.py`
- `--dataset_path` : Path to the dataset
- `--lr` : Learning rate for the optimizer
- `--epochs` : Number of training epochs
- `--patience` : Patience for early stopping
- `--optimizer` : Optimizer type (adam, sgd)
- `--criterion` : Loss criterion (cross_entropy, mse)
- `--show` : If true shows the taining results stored in `/results` directory

The trained model will be saved in the models/ directory.

## 📊 Testing and Metrics
Evaluate the model on a test dataset:

```bash
python test.py --model_path ./models/emotion_model.pth --dataset_path ./data/test
```
### Metrics Reported:

- **Accuracy**: Measures the overall correct predictions.
- **Confusion Matrix**: Shows the classification performance per emotion.
- **Precision, Recall, F1-Score**: Evaluates the model’s effectiveness per class.

## 📷 Real-Time Emotion Recognition using Webcam

To use the model with a webcam:

```bash
python camera.py --model_path ./models/emotion_model.pth
```

Press `q` or `Esc` to exit the webcam window.

## 📌 Contributions

Feel free to submit issues, fork the repo, and create pull requests to improve the project!

## Author

- **Marceau Combet** - *Developer and research assistant*
  - *Contact Information*: [marceau.combet@gmail.com](marceau.combet@gmail.com)
  - *GitHub*: [Marsoss](https://github.com/Marsoss)

## Acknowledgments

This project builds upon work originally conducted during my internship at **NTNU Gjøvik**, where I gained invaluable experience and insights that laid the foundation for this project. I am grateful for the opportunities and support provided by NTNU Gjøvik during my time there.

I would also like to acknowledge **ENSICAEN**, where I pursued my academic studies. The knowledge and skills I acquired at ENSICAEN were instrumental in shaping my approach to this project and contributing to its development.

Special thanks to the mentors who offered guidance, feedback, and encouragement throughout my journey.

## License

Copyright 2023 Marceau Combet

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Third-Party Licenses

This project uses several third-party libraries, each with its respective license:

| Library         | License               | Link |
|----------------|-----------------------|------|
| OpenCV (opencv-python) | Apache License 2.0 | [Apache 2.0](https://opensource.org/licenses/Apache-2.0) |
| PyTorch (torch) | BSD 3-Clause License | [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) |
| TorchVision | BSD 3-Clause License | [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) |
| NumPy | BSD 3-Clause License | [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) |
| Scikit-learn | BSD 3-Clause License | [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) |
| Matplotlib | PSF License (BSD-compatible) | [PSF License](https://matplotlib.org/stable/users/project/license.html) |
| Pillow | PIL License (MIT-compatible) | [PIL License](https://pillow.readthedocs.io/en/stable/license.html) |
| torchsummary | MIT License | [MIT License](https://opensource.org/licenses/MIT) |
| polarTransform | MIT License | [MIT License](https://opensource.org/licenses/MIT) |
| imageio | BSD 2-Clause License | [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) |
| tqdm | MIT License | [MIT License](https://opensource.org/licenses/MIT) |
| linformer | MIT License | [MIT License](https://opensource.org/licenses/MIT) |
| vit_pytorch | MIT License | [MIT License](https://opensource.org/licenses/MIT) |
| Tkinter | PSF License (bundled with Python) | [PSF License](https://docs.python.org/3/license.html) |

For more details, please refer to the respective license links provided above.

