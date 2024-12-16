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

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=15, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=6, kernel_size=(3, 3), padding=(3, 3), dilation=3)

        self.conv5 = nn.Conv2d(in_channels=31, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=31, out_channels=15, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        self.conv7 = nn.Conv2d(in_channels=31, out_channels=10, kernel_size=(3, 3), padding=(3, 3), dilation=3)

        self.conv8 = nn.Conv2d(in_channels=45, out_channels=65, kernel_size=(3, 3), padding=(1, 1))
        self.conv9 = nn.Conv2d(in_channels=65, out_channels=90, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(3, stride=2)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.
        #self.fc1 = nn.Linear(in_features=60, out_features=30)
        self.fc2 = nn.Linear(in_features=90, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)

        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
                       self.conv6, self.conv7, self.conv8, self.conv9, self.fc2]
        self.mode = len(self.layers)-1

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        #x = self.dropout2d(x)

        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x3 = self.conv4(x)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.pool(x)
        # x = self.dropout2d(x)

        x1 = self.conv5(x)
        x2 = self.conv6(x)
        x3 = self.conv7(x)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.pool(x)
        # x = self.dropout2d(x)


        x = self.conv8(x)
        x = self.pool(x)
        # x = self.dropout2d(x)
        x = self.conv9(x)
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

    def freeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self.mode = len(self.layers)-1

    def step(self):
        for param in self.layers[self.mode].parameters():
            param.requires_grad = False
        self.mode = (self.mode+1) % len(self.layers)
        for param in self.layers[self.mode].parameters():
            param.requires_grad = True

    def cancel_dropout(self):
        self.dropout2d = nn.Identity()

