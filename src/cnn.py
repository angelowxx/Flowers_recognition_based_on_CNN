""" File with CNN models. Add your custom CNN model here. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms


class SampleModel(nn.Module):
    """
    A sample PyTorch CNN model
    """

    def __init__(self, input_shape=(3, 64, 64), num_classes=17):
        super(SampleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(3, stride=2)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.
        self.fc1 = nn.Linear(in_features=4500, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x


class HomemadeModel(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), num_classes=17):
        super(HomemadeModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=40, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=80, out_channels=40, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(3, stride=2)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.
        self.fc1 = nn.Linear(in_features=360, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        # x = self.dropout2d(x)
        x = self.conv2(x)
        x = self.pool(x)
        # x = self.dropout2d(x)
        x = self.conv3(x)
        x = self.pool(x)
        # x = self.dropout2d(x)
        x = self.conv4(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x

    def freeze_convolution_layers(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = True
        for param in self.fc1.parameters():
            param.requires_grad = True

    def freeze_linear_layers(self):
        for param in self.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False


class FastCNN(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), num_classes=17):
        super(FastCNN, self).__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=20, kernel_size=9, stride=2
                      , padding=4, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=20),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=40),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=60),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=60, out_channels=110, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=110),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.Linear = nn.Sequential(
            nn.Linear(in_features=110, out_features=45),
            nn.BatchNorm1d(num_features=45),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(in_features=45, out_features=num_classes),
            nn.BatchNorm1d(num_features=num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.normalize(x)
        x = self.CNN(x)

        x = self.global_pool(x).view(x.size(0), -1)
        x = self.Linear(x)

        return x


class LargeCNN(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), num_classes=17):
        super(LargeCNN, self).__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=7),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(5, stride=3),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(5, stride=3),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(5, stride=3),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.5),

        )

        self.Linear = nn.Sequential(
            nn.Linear(in_features=320, out_features=128),
            nn.BatchNorm1d(num_features=56),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(in_features=128, out_features=num_classes),
            nn.BatchNorm1d(num_features=num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.normalize(x)
        x = self.CNN(x)

        x = x.view(x.size(0), -1)
        x = self.Linear(x)

        return x


# Depthwise Separable Convolution Layer
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                    groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise3(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# MobileNet-like Model
class MobileNetLike(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), num_classes=17):
        super(MobileNetLike, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv1 = DepthwiseSeparableConv(32, 32)
        self.conv2 = DepthwiseSeparableConv(32, 64)
        self.conv3 = DepthwiseSeparableConv(64, 64)
        self.conv4 = DepthwiseSeparableConv(64, 128)
        self.conv5 = DepthwiseSeparableConv(128, 128)
        self.conv6 = DepthwiseSeparableConv(128, 256)

        self.dropout = nn.Dropout2d(0.5)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Sequential(
            nn.Linear(256, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(96, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.initial_conv(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.dropout(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
