from torchvision import models


def maskrcnn():
    return models.detection.maskrcnn_resnet50_fpn_v2(
        weights=models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        num_classes=6,
    )


