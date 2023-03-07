import torch
import torch.nn as nn
import torch.nn.functional as F



class NN_128x128(nn.Module):
    def __init__(self,input_channel: int,num_classes: int, data_size=128 ):
        '''
        Convolutional Neural Network for classification task.\n
        Parameters
        ----------
            input_channel (int): number of channel in input. (RGB=3, grayscale=1)
            num_classes (int): number of classes in the dataset.
        '''
        super(NN_128x128,self).__init__()
        self.input_ch = input_channel
        self.num_classes = num_classes

        #print(data_size)

        # Flatten layer (from ConvLayer to fully-connected)
        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(data_size,64)
        self.fc2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64,self.num_classes)
        

    def forward(self,x):
        # FC phase
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x
