import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Segmentation4(nn.Module):
    def __init__(self, in_channels, out_channels, block=BasicBlock):
        assert in_channels == out_channels
        super(Segmentation4, self).__init__()
        filters = [
            in_channels,
            in_channels * 2,
            in_channels * 4,
            in_channels * 8,
            in_channels * 16,
        ]

        self.downsample1 = nn.Sequential(
            conv1x1(filters[0], filters[1], 1),
            nn.BatchNorm2d(filters[1]),
        )

        self.downsample2 = nn.Sequential(
            conv1x1(filters[1], filters[2], 1),
            nn.BatchNorm2d(filters[2]),
        )

        self.downsample3 = nn.Sequential(
            conv1x1(filters[2], filters[3], 1),
            nn.BatchNorm2d(filters[3]),
        )

        self.downsample4 = nn.Sequential(
            conv1x1(filters[3], filters[4], 1),
            nn.BatchNorm2d(filters[4]),
        )

        """
        self.conv1 = block(filters[0], filters[1], downsample=conv1x1(filters[0], filters[1], 1))
        self.conv2 = block(filters[1], filters[2], downsample=conv1x1(filters[1], filters[2], 1))
        self.conv3 = block(filters[2], filters[3], downsample=conv1x1(filters[2], filters[3], 1))
        """

        self.conv1 = block(filters[0], filters[1], downsample=self.downsample1)
        self.conv2 = block(filters[1], filters[2], downsample=self.downsample2)
        self.conv3 = block(filters[2], filters[3], downsample=self.downsample3)
        self.conv4 = block(filters[3], filters[4], downsample=self.downsample4)

        self.down_pooling = nn.MaxPool2d(kernel_size=2)

        self.downsample5 = nn.Sequential(
            conv1x1(filters[4], filters[3], 1),
            nn.BatchNorm2d(filters[3]),
        )

        self.downsample6 = nn.Sequential(
            conv1x1(filters[3], filters[2], 1),
            nn.BatchNorm2d(filters[2]),
        )

        self.downsample7 = nn.Sequential(
            conv1x1(filters[2], filters[1], 1),
            nn.BatchNorm2d(filters[1]),
        )

        self.downsample8 = nn.Sequential(
            conv1x1(filters[1], filters[0], 1),
            nn.BatchNorm2d(filters[0]),
        )

        """
        self.up_pool4 = up_pooling(filters[3], filters[2])
        self.conv4 = block(filters[3], filters[2], downsample=conv1x1(filters[3], filters[2], 1))
        self.up_pool5 = up_pooling(filters[2], filters[1])
        self.conv5 = block(filters[2], filters[1], downsample=conv1x1(filters[2], filters[1], 1))

        self.conv6 = block(filters[1], filters[0], downsample=conv1x1(filters[1], filters[0], 1))
        """

        self.up_pool5 = up_pooling(filters[4], filters[3])
        self.conv5 = block(filters[4], filters[3], downsample=self.downsample5)
        self.up_pool6 = up_pooling(filters[3], filters[2])
        self.conv6 = block(filters[3], filters[2], downsample=self.downsample6)
        self.up_pool7 = up_pooling(filters[2], filters[1])
        self.conv7 = block(filters[2], filters[1], downsample=self.downsample7)
        self.conv8 = block(filters[1], filters[0], downsample=self.downsample8)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)

        x5 = self.up_pool5(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.conv5(x5)

        x6 = self.up_pool6(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.conv6(x6)

        x7 = self.up_pool7(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.conv7(x7)

        x8 = self.conv8(x7)

        output = torch.softmax(x8, dim=1)
        # output = torch.sigmoid(x8)
        return output


class Segmentation3(nn.Module):
    def __init__(self, in_channels, out_channels, block=BasicBlock):
        assert in_channels == out_channels
        super(Segmentation3, self).__init__()
        filters = [in_channels, in_channels * 2, in_channels * 4, in_channels * 8]

        self.downsample1 = nn.Sequential(
            conv1x1(filters[0], filters[1], 1),
            nn.BatchNorm2d(filters[1]),
        )

        self.downsample2 = nn.Sequential(
            conv1x1(filters[1], filters[2], 1),
            nn.BatchNorm2d(filters[2]),
        )

        self.downsample3 = nn.Sequential(
            conv1x1(filters[2], filters[3], 1),
            nn.BatchNorm2d(filters[3]),
        )

        """
        self.conv1 = block(filters[0], filters[1], downsample=conv1x1(filters[0], filters[1], 1))
        self.conv2 = block(filters[1], filters[2], downsample=conv1x1(filters[1], filters[2], 1))
        self.conv3 = block(filters[2], filters[3], downsample=conv1x1(filters[2], filters[3], 1))
        """

        self.conv1 = block(filters[0], filters[1], downsample=self.downsample1)
        self.conv2 = block(filters[1], filters[2], downsample=self.downsample2)
        self.conv3 = block(filters[2], filters[3], downsample=self.downsample3)

        self.down_pooling = nn.MaxPool2d(kernel_size=2)

        self.downsample4 = nn.Sequential(
            conv1x1(filters[3], filters[2], 1),
            nn.BatchNorm2d(filters[2]),
        )

        self.downsample5 = nn.Sequential(
            conv1x1(filters[2], filters[1], 1),
            nn.BatchNorm2d(filters[1]),
        )

        self.downsample6 = nn.Sequential(
            conv1x1(filters[1], filters[0], 1),
            nn.BatchNorm2d(filters[0]),
        )

        """
        self.up_pool4 = up_pooling(filters[3], filters[2])
        self.conv4 = block(filters[3], filters[2], downsample=conv1x1(filters[3], filters[2], 1))
        self.up_pool5 = up_pooling(filters[2], filters[1])
        self.conv5 = block(filters[2], filters[1], downsample=conv1x1(filters[2], filters[1], 1))

        self.conv6 = block(filters[1], filters[0], downsample=conv1x1(filters[1], filters[0], 1))
        """

        self.up_pool4 = up_pooling(filters[3], filters[2])
        self.conv4 = block(filters[3], filters[2], downsample=self.downsample4)
        self.up_pool5 = up_pooling(filters[2], filters[1])
        self.conv5 = block(filters[2], filters[1], downsample=self.downsample5)

        self.conv6 = block(filters[1], filters[0], downsample=self.downsample6)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)

        x4 = self.up_pool4(x3)
        x4 = torch.cat([x4, x2], dim=1)

        x4 = self.conv4(x4)

        x5 = self.up_pool5(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.conv5(x5)

        x6 = self.conv6(x5)

        output = torch.softmax(x6, dim=1)
        # output = torch.sigmoid(x6)
        return output


class Segmentation2(nn.Module):
    def __init__(self, in_channels, out_channels, block=BasicBlock):
        assert in_channels == out_channels
        super(Segmentation2, self).__init__()
        filters = [in_channels, in_channels * 2, in_channels * 4, in_channels * 8]

        self.downsample1 = nn.Sequential(
            conv1x1(filters[0], filters[1], 1),
            nn.BatchNorm2d(filters[1]),
        )

        self.downsample2 = nn.Sequential(
            conv1x1(filters[1], filters[2], 1),
            nn.BatchNorm2d(filters[2]),
        )

        """
        self.conv1 = block(filters[0], filters[1], downsample=conv1x1(filters[0], filters[1], 1))
        self.conv2 = block(filters[1], filters[2], downsample=conv1x1(filters[1], filters[2], 1))
        """
        self.conv1 = block(filters[0], filters[1], downsample=self.downsample1)
        self.conv2 = block(filters[1], filters[2], downsample=self.downsample2)

        self.down_pooling = nn.MaxPool2d(kernel_size=2)

        self.downsample3 = nn.Sequential(
            conv1x1(filters[2], filters[1], 1),
            nn.BatchNorm2d(filters[1]),
        )

        self.downsample4 = nn.Sequential(
            conv1x1(filters[1], filters[0], 1),
            nn.BatchNorm2d(filters[0]),
        )

        """
        self.up_pool3 = up_pooling(filters[2], filters[1])
        self.conv3 = block(filters[2], filters[1], downsample=conv1x1(filters[2], filters[1], 1))
        self.conv4 = block(filters[1], filters[0], downsample=conv1x1(filters[1], filters[0], 1))
        """
        self.up_pool3 = up_pooling(filters[2], filters[1])
        self.conv3 = block(filters[2], filters[1], downsample=self.downsample3)
        self.conv4 = block(filters[1], filters[0], downsample=self.downsample4)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)

        x3 = self.up_pool3(x2)
        x3 = torch.cat([x3, x1], dim=1)
        x3 = self.conv3(x3)

        x4 = self.conv4(x3)

        output = torch.softmax(x4, dim=1)
        # output = torch.sigmoid(x4)
        return output


class Segmentation1(nn.Module):
    def __init__(self, in_channels, out_channels, block=BasicBlock):
        assert in_channels == out_channels
        super(Segmentation1, self).__init__()
        filters = [in_channels, in_channels * 2, in_channels * 4, in_channels * 8]

        self.downsample1 = nn.Sequential(
            conv1x1(filters[0], filters[1], 1),
            nn.BatchNorm2d(filters[1]),
        )

        self.conv1 = block(filters[0], filters[1], downsample=self.downsample1)

        self.downsample2 = nn.Sequential(
            conv1x1(filters[1], filters[0], 1),
            nn.BatchNorm2d(filters[0]),
        )

        self.conv2 = block(filters[1], filters[0], downsample=self.downsample2)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        output = torch.softmax(x2, dim=1)
        # output = torch.sigmoid(x2)
        return output


def segmentation(in_channels, out_channels, depth, block=BasicBlock):
    if depth == 1:
        return Segmentation1(in_channels, out_channels, block)
    elif depth == 2:
        return Segmentation2(in_channels, out_channels, block)
    elif depth == 3:
        return Segmentation3(in_channels, out_channels, block)
    elif depth == 4:
        return Segmentation4(in_channels, out_channels, block)
    else:
        traceback.print_exc()
        raise Exception("depth need to be from 1-4")