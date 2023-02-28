import cv2, glob, numpy, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
from colorsys import hls_to_rgb
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

def loadData(path, folders):

    data = []
    labels = []

    for index, foldername in enumerate(folders):
        new_data = [numpy.asarray(cv2.imread(file, cv2.IMREAD_UNCHANGED)) for file in glob.glob(f'{path}/{foldername}/*.jpg')]
        data += new_data
        labels += [index]*len(new_data)
        
    return numpy.asarray(data), labels

def get_data_split(test_perc, data_path, lab_classes, data = None):
    random_state = 42
    shuffle = True
    if data is None:
        return train_test_split(*loadData(data_path, lab_classes), test_size=test_perc, random_state=random_state, shuffle=shuffle)
    else:
        return train_test_split(*data, test_size=test_perc, random_state=random_state, shuffle=shuffle)

def load_scatter(path):
    raw_data = loadmat(f'{path}/scatter.mat')['datafeatures']
    data = raw_data[0]
    labels = np.array([lab[0][0] for lab in raw_data[1]])
    return data, labels

def scatter_mem(batch_size, device, scatter, dataset, channels):
    scatters = []
    labels = []

    #dataset_device = torch.tensor(dataset).to(device).float().contiguous()
    dataset_device = DataLoader(dataset[0],batch_size=batch_size,drop_last=True)
    labels_batches = DataLoader(dataset[1],batch_size=batch_size,drop_last=True)

    for x, y in zip(dataset_device, labels_batches):
        # change the size for the input data - convert to float type
        if channels != 1:
            x = x.movedim(3, 1).to(device).float().contiguous()
        else:
            x = x.to(device).float().contiguous()
        #print(f'Scattering input shape: {x.shape}')
        x = scatter(x)        
        #print(f'Scattering output shape: {x.shape}')
        if channels != 1:
            x = x.movedim(1, 2).mean(axis=(3, 4))#.movedim(1, 2)# scatter the data and average the values    
        else:
            x = x.mean(axis=(2, 3))
        #print(f'Scattering final shape: {x.shape}')
        x = x.cpu().detach()#.reshape(batch_size, -1)
        scatters += x
        labels += y.cpu()


    return scatters, labels

class metrics:

    def __init__(self, y_true, y_pred, lab_classes) -> None:
        self.classes = lab_classes
        _, self.y_pred = y_pred.max(1)
        self.y_true = y_true

        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.precision = precision_score(self.y_true, self.y_pred)
        self.recall = recall_score(self.y_true, self.y_pred)
        self.f1 = f1_score(self.y_true, self.y_pred)

        self.confmat = confusion_matrix(self.y_true, self.y_pred, labels=list(range(0,len(self.classes))))
        fpr, tpr, threshold = roc_curve(self.y_true, self.y_pred)
        self.roc = (fpr, tpr)

    def __str__(self) -> str:
        return f'Accuracy:\t\t{self.accuracy}\nPrecision:\t\t{self.precision}\nRecall:\t\t\t{self.recall}\nF1:\t\t\t\t{self.f1}'
    
    def printMetrics(self, type = None):
        if type is not None:
            print(f'{type} metrics: \n{self}')
        else:
            print(f'Metrics: \n{self}')
        
    def getMetrics(self, type = None):
        if type is not None:
            return f'{type} metrics: \n{self}'
        else:
            return f'Metrics: \n{self}'


    def rocDisplay(self):
        return RocCurveDisplay(*self.roc)

    def confMatDisplay(self):
        return ConfusionMatrixDisplay(self.confmat, display_labels=self.classes)

    def precisionRecallDisplay(self):
        return PrecisionRecallDisplay(self.precision, self.recall)

    
def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0/(1.0 + abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.data[idx]
        sample = [data,label]
        return sample
    

def batcher(x_train, x_test, y_train, y_test, batch_size = 64):
    # Create Dataloader with batch size = 64
    train_dataset = CustomDataset(x_train,y_train)    # we use a custom dataset defined in utils.py file
    test_dataset = CustomDataset(x_test,y_test)       # we use a custom dataset defined in utils.py file

    trainset = DataLoader(train_dataset,batch_size=batch_size,drop_last=True)    # construct the trainset with subjects divided in mini-batch
    testset = DataLoader(test_dataset,batch_size=batch_size,drop_last=True)      # construct the testset with subjects divided in mini-batch

    return trainset, testset

def load_settings(filename = 'parameters.yaml'):
    import yaml
    with open(filename) as f:
        settings = yaml.load(f, Loader=yaml.loader.FullLoader)
    return settings