import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN_128x128(nn.Module):
    def __init__(self,input_channel: int,num_classes: int):
        '''
        Convolutional Neural Network for classification task.\n
        Parameters
        ----------
            input_channel (int): number of channel in input. (RGB=3, grayscale=1)
            num_classes (int): number of classes in the dataset.
        '''
        super(CNN_128x128,self).__init__()
        self.input_ch = input_channel
        self.num_classes = num_classes
        self.channels = [16,32,64]

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.input_ch,out_channels=self.channels[0],kernel_size=(7),stride=(2), padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=(5),stride=(2), padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.channels[1],out_channels=self.channels[2],kernel_size=(3),stride=(2), padding=1)

        self.batchnorm1 = nn.BatchNorm2d(self.channels[0])
        self.batchnorm2 = nn.BatchNorm2d(self.channels[1])
        self.batchnorm3 = nn.BatchNorm2d(self.channels[2])

        self.drop1 = nn.Dropout(p=0.2)

        # Flatten layer (from ConvLayer to fully-connected)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256,64)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, self.num_classes)
        

    def forward(self,x):
        # CNN phase
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.drop1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.drop1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.drop1(x)
        
        x = self.flat(x)    # flat the data to get a vector for FC layers

        # FC phase
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x



        