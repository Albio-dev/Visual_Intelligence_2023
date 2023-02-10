import cv2, glob, numpy
from torch.utils.data import DataLoader
from utils import CustomDataset

def loadData(path, folders):

    data = []
    labels = []

    for index, foldername in enumerate(folders):
        new_data = [numpy.asarray(cv2.imread(file)) for file in glob.glob(f'{path}/{foldername}/*.jpg')]
        data += new_data
        labels += [index]*len(new_data)
        
    return numpy.asarray(data), labels


def batcher(x_train, x_test, y_train, y_test, batch_size = 64):
    # Create Dataloader with batch size = 64
    train_dataset = CustomDataset(x_train,y_train)    # we use a custom dataset defined in utils.py file
    test_dataset = CustomDataset(x_test,y_test)       # we use a custom dataset defined in utils.py file

    trainset = DataLoader(train_dataset,batch_size=batch_size,drop_last=True)    # construct the trainset with subjects divided in mini-batch
    testset = DataLoader(test_dataset,batch_size=batch_size,drop_last=True)      # construct the testset with subjects divided in mini-batch

    return trainset, testset


def scatter_mem(batch_size,device,scatter,trainset,cpu_device):
    scatters = [None]*len(trainset)

    for i, data_tr in enumerate(trainset):
        x,y = data_tr                        # unlist the data from the train set
        x = x.view(batch_size,3,128,128).float().to(device)     # change the size for the input data - convert to float type
        x = scatter(x).mean(axis=(3, 4)).movedim(1, 2).to(device)
        scatters[i] = x.to(cpu_device)

    return scatters