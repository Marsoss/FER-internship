from torch_models.inception import Inception
from torch_models.vgg import VGG
from torch_models.resnet import ResNet
from torch_models.emotionCNN import EmotionCNN
from torch_models.default import DefaultModel
from torch_models.sota import SotA
from torch_models.seg import vgg19_bn_mask
from torch_models.transformer import vision_transformer
from torch_models.unet import unet
from torch_models.deeplab import deeplabv3
from torch_models.resunet import resunet_plus_plus
from torch_models.segnet import segnet
from torch_models.maskrcnn import maskrcnn
import torch.nn as nn
import torch

MODELS = ['inception', 'vgg'      , 'resnet', 'emotion', 'default', 
          'sota'     , 'seg_paper', 'trans' , 'unet1'   , 'deeplab101',
          'deeplab19', 'resunet'  , 'segnet', 'mrcnn']

def get_model(model_name:str) -> str:
    for model in MODELS:
        if model in model_name.lower(): 
            break
    else:
        raise ValueError("Model name unknown or not recognized")
    return model

def create_model(model_name:str, height:int=300, width:int=300, num_channels:int=1, num_classes:int=7) -> nn.Module:
    try:
        model = get_model(model_name=model_name)
        if model == 'inception' : return Inception(num_classes)
        if model == 'vgg' : return VGG(num_classes)
        if model == 'resnet' : return ResNet(num_classes)
        if model == 'emotion' : return EmotionCNN(num_classes, num_channels, width, height)
        if model == 'default' : return  DefaultModel(num_classes, num_channels, width, height)
        if model == 'sota' : return SotA(num_channels, num_classes, width, height)
        if model == 'seg_paper' : return vgg19_bn_mask(in_channels=num_channels, num_classes=num_classes)
        if model == 'trans' : return vision_transformer(image_size=height, num_channels=num_channels, num_classes=num_classes)
        if model == 'unet1' : return unet(in_channels=num_channels, num_classes=num_classes)
        if model == 'deeplab19' : return deeplabv3( num_classes=num_classes, model_name='vgg19')
        if model == 'deeplab101' : return deeplabv3( num_classes=num_classes, model_name='resnet101')
        if model == 'resunet' : return resunet_plus_plus(num_channnel=num_channels, num_classes=num_classes, height=height, width=width)
        if model == 'segnet' : return segnet(num_channnel=num_channels, num_classes=num_classes)
        if model == 'mrcnn' : return maskrcnn()

    except ValueError as e:
        print(e) 
    
def load_model(model_name:str, size:int=96, num_channel:int=1, num_classes:int=7) -> None:
        model = create_model(model_name=model_name, height=size, width=size, num_channels=num_channel, num_classes=num_classes)
        checkpoint = torch.load(model_name)
        state_dict = checkpoint
        print("loading dict...")
        model.load_state_dict(state_dict)
        return model