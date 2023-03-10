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
import random
import matplotlib.pyplot as plt
import os

def loadData(path, folders, training_data_size=500):

    data = []
    labels = []

    for index, foldername in enumerate(folders):
        new_data = [numpy.asarray(cv2.imread(file, cv2.IMREAD_UNCHANGED)) for file in glob.glob(f'{path}/{foldername}/*.jpg')]
        data += new_data
        labels += [index]*len(new_data)
        
    random.seed(42)
    temp = random.sample(list(zip(numpy.asarray(data).astype(numpy.float32), labels)), training_data_size)

    for label in folders:
        if not os.path.exists(f'{path}/temp/{label}'):
            os.makedirs(f'{path}/temp/{label}')

    for num, i in enumerate(temp):
        cv2.imwrite(f'{path}/temp/{folders[i[1]]}/{num}.png', i[0])
        pass
    
    return [i[0] for i in temp], [i[1] for i in temp]

def get_data_split(test_perc, data_path, lab_classes, training_data_size, data = None):
    random_state = 42
    shuffle = True
    if data is None:
        return train_test_split(*loadData(data_path, lab_classes, training_data_size=training_data_size), test_size=test_perc, random_state=random_state, shuffle=shuffle)
    else:
        return train_test_split(*data, test_size=test_perc, random_state=random_state, shuffle=shuffle)

def load_scatter(path):
    raw_data = loadmat(f'{path}/scatter.mat')['datas']
    data = raw_data[0][0]
    if raw_data[0][1][0].shape == (1,):
        labels = np.array([lab[0] for lab in raw_data[0][1]])
    else:
        labels = np.array([lab[0][0] for lab in raw_data[0][1][0]])
    return data, labels

def matlab_scatter(channels, data, J, qualityFactors, rotations):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    images = data[0]
    labels = data[1]
    qfact1 = qualityFactors[0]
    qfact2 = qualityFactors[1]
    scatter = eng.scattering_function(channels, images,labels, float(J), float(qfact1),float(qfact2), float(rotations), nargout = 2)
    eng.quit()
    return scatter
    

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

def display_stats_graphs(stats_CNN, stats_NN, epochs, save_path=None):
    loss_limit = max(max(list(stats_CNN.values())[0]), max(list(stats_NN.values())[0]))
    fig, axs = plt.subplots(2, 2)
    axs[0][0].set_xlabel('Epochs')
    axs[0][0].set_ylabel('Loss')
    axs[0][0].set_xticks(range(epochs))
    axs[0][0].set_ylim(0, loss_limit)
    axs[0][0].set_title('loss in train for CNN')
    axs[0][0].plot(range(epochs), list(stats_CNN.values())[0])
    
    axs[0][1].set_xlabel('Epochs')
    axs[0][1].set_ylabel('Loss')
    axs[0][1].set_xticks(range(epochs))
    axs[0][1].set_ylim(0, loss_limit)
    axs[0][1].set_title('loss in train for scattering NN')
    axs[0][1].plot(range(epochs), list(stats_NN.values())[0])

    acc_worst = min(min(list(stats_CNN.values())[1]), min(list(stats_NN.values())[1]))
    axs[1][0].set_xlabel('Epochs')
    axs[1][0].set_ylabel('Accuracy')
    axs[1][0].set_xticks(range(epochs))
    axs[1][0].set_ylim(acc_worst, 1)
    axs[1][0].set_title('Accuracy in training for CNN')
    axs[1][0].plot(range(epochs), list(stats_CNN.values())[1])

    axs[1][1].set_xlabel('Epochs')
    axs[1][1].set_ylabel('Accuracy')
    axs[1][1].set_xticks(range(epochs))
    axs[1][1].set_ylim(acc_worst, 1)
    axs[1][1].set_title('Accuracy in training for scattering NN')
    axs[1][1].plot(range(epochs), list(stats_NN.values())[1])

    if save_path is not None:
        fig.savefig(save_path,dpi=300)

    fig.show()
    
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

def get_folder_index(path):
    return str(len(os.listdir(path)) + 1)
