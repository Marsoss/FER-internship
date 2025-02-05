import torch.nn as nn
import torch
import torchvision.models as models


# DeepLabV3_Resnet101
class DeepLabV3ResNet101(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3ResNet101, self).__init__()
        
        # Load the pre-trained ResNet model
        resnet = models.resnet101(pretrained=True)
        
        # Remove the fully connected layer and average pooling
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Atrous Spatial Pyramid Pooling (ASPP)
        self.aspp = ASPP(resnet.fc.in_features, 256)
        
        # Decoder
        self.decoder = Decoder(6*256, 256)

        # Classifier 
        self.classifier = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x)

        # Classifier

        x = self.classifier(x)
        
        return x
    
class DeepLabV3VGG19(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3VGG19, self).__init__()
        
        # Load the pre-trained VGG model
        vgg = models.vgg19(pretrained=True)
        
        # Remove the fully connected layers and average pooling
        self.encoder = nn.Sequential(*list(vgg.features.children())[:-1])
        
        # Atrous Spatial Pyramid Pooling (ASPP)
        self.aspp = ASPP(512, 256)
        
        # Decoder
        self.decoder = Decoder(6 * 256, 256)

        # Classifier 
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, padding=0)
        
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x)

        # Classifier
        x = self.classifier(x)
        
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        dilations = [1, 6, 12, 18]
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[0], padding=dilations[0])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[1], padding=dilations[1])
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[2], padding=dilations[2])
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[3], padding=dilations[3])
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        
        global_feat = self.global_avg_pool(x)
        global_feat = self.conv1x1(global_feat)
        global_feat = nn.functional.interpolate(global_feat, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat((feat1, feat2, feat3, feat4, feat5, global_feat), dim=1)
        
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.conv2(x)

        return x

def deeplabv3(num_classes:int=7, model_name:str='vgg19'):  

    if model_name.lower() == 'vgg19':
        model = DeepLabV3VGG19(num_classes=num_classes)
    elif model_name.lower() == 'resnet101':
        model = DeepLabV3ResNet101(num_classes=num_classes)
     
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(4*64*8*8, 1024),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1024, 1024),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1024, num_classes),
    )
    return model
