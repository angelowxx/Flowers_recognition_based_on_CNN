""" File with CNN models. Add your custom CNN model here. """
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class HandmadeModel(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), num_classes=17):
        super(HandmadeModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=20, kernel_size=(3, 3), padding=(1, 1))

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(5, 5), padding=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(7, 7), padding=(3, 3))

        self.conv5 = nn.Conv2d(in_channels=30, out_channels=60, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=(3, 3), padding=(1, 1))
        self.conv7 = nn.Conv2d(in_channels=60, out_channels=70, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(3, stride=2)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.
        #self.fc1 = nn.Linear(in_features=60, out_features=30)
        self.fc2 = nn.Linear(in_features=70, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        #x = self.dropout2d(x)

        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x3 = self.conv4(x)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.pool(x)
        #x = self.dropout2d(x)

        x = self.conv5(x)
        x = self.pool(x)
        #x = self.dropout2d(x)
        x = self.conv6(x)
        x = self.pool(x)
        #x = self.dropout2d(x)
        x = self.conv7(x)
        #x = torch.cat((x, x1), dim=1)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        #x = self.fc1(x)
        #x = self.dropout(x)
        #x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)

        return x

    def freeze_convolution_layers(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = True
        """for param in self.fc1.parameters():
            param.requires_grad = True"""

    def freeze_linear_layers(self):
        for param in self.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = False
        """for param in self.fc1.parameters():
            param.requires_grad = False"""
