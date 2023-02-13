import numpy as np
import torch, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from CNN_128x128 import CNN_128x128
from utils import compute_metrics
import matplotlib.pyplot as plt
import seaborn as sns

import utils_our

# Set device where to run the model. GPU if available, otherwise cpu (very slow with deep learning models)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# test
import yaml
with open('parameters.yaml', 'r') as f:
    settings = yaml.load(f, Loader=yaml.loader.FullLoader)
print(settings)

data_path = settings['data_path']
model_train_path = settings['model_train_path']
if not os.path.exists(model_train_path):                 # create a directory where to save the best model
    os.makedirs(model_train_path)
test_perc = settings['test_perc']
batch_size = settings['batch_size']
learning_rate = settings['learning_rate']
momentum = settings['momentum']
num_epochs = settings['num_epochs']  
lab_classes = settings['lab_classes']

### MODEL VARIABLES ###
# Define useful variables
# number of classes in the dataset
n_classes = len(lab_classes)                


### DATA LOADING ###
# Split in train and test set
#trainset, testset = utils_our.batcher(batch_size = batch_size, *train_test_split(*utils_our.loadData(data_path, lab_classes), test_size=test_perc))  

def train(trainset):    
    
    best_acc = 0.0
    # Variables to store the results
    losses = []
    acc_train = []
    pred_label_train = torch.empty((0)).to(device)    # .to(device) to move the data/model on GPU or CPU (default)
    true_label_train = torch.empty((0)).to(device)

    ### CREATE MODEL ###

    # Model
    model = CNN_128x128(input_channel=3,num_classes=n_classes).to(device)

    # Optimizer
    optim = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()


    ### FIT MODEL ###
    for epoch in range(num_epochs):
        # Train step
        model.train()                                                   # tells to the model you are in training mode (batchnorm and dropout layers work)
        for data_tr in trainset:
            optim.zero_grad()

            x,y = data_tr                        # unlist the data from the train set
            x = x.view(batch_size,3,128,128).float().to(device)     # change the size for the input data - convert to float type
            y = y.to(device)
            y_pred = model(x)                                        # run the model
            loss = criterion(y_pred,y)                               # compute loss
            _,pred = y_pred.max(1)                                      # get the index == class of the output along the rows (each sample)
            pred_label_train = torch.cat((pred_label_train,pred),dim=0)
            true_label_train = torch.cat((true_label_train,y),dim=0)
            loss.backward()                                             # compute backpropagation
            optim.step()                                                # parameter update

        losses.append(loss.cpu().detach().numpy())
        acc_t = accuracy_score(true_label_train.cpu(),pred_label_train.cpu())
        acc_train.append(acc_t)
        print("Epoch: {}/{}, loss = {:.4f} - acc = {:.4f}".format(epoch + 1, num_epochs, loss, acc_t))
        if acc_t > best_acc:                                                            # save the best model (the highest accuracy in validation)
            torch.save(model.state_dict(), model_train_path + 'CNN_128x128_best_model_trained.pt')
            best_acc = acc_t

        # Reinitialize the variables to compute accuracy
        pred_label_train = torch.empty((0)).to(device)
        true_label_train = torch.empty((0)).to(device)
    
    return model

def test(testset):
    ### TEST MODEL ###
    model_test = CNN_128x128(input_channel=3,num_classes=n_classes).to(device)                # Initialize a new model
    model_test.load_state_dict(torch.load(model_train_path+'CNN_128x128_best_model_trained.pt'))   # Load the model

    pred_label_test = torch.empty((0,n_classes)).to(device)
    true_label_test = torch.empty((0)).to(device)

    with torch.no_grad():
        for data in testset:
            X_te, y_te = data
            X_te = X_te.view(batch_size,3,128,128).float().to(device)
            y_te = y_te.to(device)
            output_test = model_test(X_te)
            pred_label_test = torch.cat((pred_label_test,output_test),dim=0)
            true_label_test = torch.cat((true_label_test,y_te),dim=0)

    return utils_our.metrics(true_label_test.cpu(), pred_label_test.cpu(), lab_classes), model_test


def getData():
    # Split in train and test set
    return utils_our.batcher(batch_size = batch_size, *train_test_split(*utils_our.loadData(data_path, lab_classes), test_size=test_perc))


def isTrained():
    return os.path.isfile(model_train_path+'CNN_128x128_best_model_trained.pt')


if __name__ == "__main__":
    trainset, testset = getData()
    train(trainset=trainset)
    metrics = test(testset=testset)

    metrics.confMatDisplay().plot()
    plt.show()


