import numpy as np
import torch, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from CNN_128x128 import CNN_128x128
from NN_128x128 import NN_128x128
from utils import compute_metrics
import matplotlib.pyplot as plt
import sys

import utils_our
import kymatio.torch as kt

# test

### Parameters ###
data_path = './Data'
model_train_path = './train_checkpoint'
if not os.path.exists(model_train_path):                 # create a directory where to save the best model
    os.makedirs(model_train_path)

test_perc = .3

# How many samples are used per-iteration
batch_size = 64

# Learning rate to scale how much new weighs are evaluated
learning_rate = 0.01

# Scale for past experience to not be perturbated by new ones
momentum = 0.5

# The number of times the model is trained on the entire training dataset.
num_epochs = 80   

## Scatter parameters ##
J = 2
imageSize = (128, 128)
order = 2

# Classes in the dataset             
lab_classes = ['dog','flower']

# Set device where to run the model. GPU if available, otherwise cpu (very slow with deep learning models)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Available device: ', device)

### DATA LOADING ###
# Split in train and test set
trainset, testset = utils_our.batcher(batch_size = batch_size, *train_test_split(*utils_our.loadData(data_path, lab_classes), test_size=test_perc))


### SCATTERING DATA ###

scatter = kt.Scattering2D(J, shape = imageSize, max_order = order)
scatter = scatter.to(device)

print(f'Calculating scattering coefficients of data in {len(trainset)} batches')
scatters = utils_our.scatter_mem(batch_size,device,scatter,trainset)

### MODEL VARIABLES ###
# Define useful variables
best_acc = 0.0
n_classes = len(lab_classes)                # number of classes in the dataset

# Variables to store the results
losses = []
acc_train = []
pred_label_train = torch.empty((0)).to(device)    # .to(device) to move the data/model on GPU or CPU (default)
true_label_train = torch.empty((0)).to(device)


### CREATE MODEL ###

# Model
model = NN_128x128(input_channel=3, num_classes=n_classes).to(device)

# Optimizer
optim = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)

# Loss function
criterion = torch.nn.CrossEntropyLoss()



if scatters is None:
    print('Error during scatter_mem!')
    sys.exit()

### FIT MODEL ###
for epoch in range(num_epochs):
    # Train step
    model.train()                                                   # tells to the model you are in training mode (batchnorm and dropout layers work)
    for i, data_tr in enumerate(trainset):
        optim.zero_grad()

        x,y = data_tr                        # unlist the data from the train set        
        x = scatters[i].to(device)
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
        torch.save(model.state_dict(), model_train_path + 'NN_128x128_best_model_trained.pt')
        best_acc = acc_t

    # Reinitialize the variables to compute accuracy
    pred_label_train = torch.empty((0)).to(device)
    true_label_train = torch.empty((0)).to(device)

### TEST MODEL ###
model_test = NN_128x128(input_channel=3,num_classes=n_classes).to(device)                # Initialize a new model
model_test.load_state_dict(torch.load(model_train_path+'NN_128x128_best_model_trained.pt'))   # Load the model

pred_label_test = torch.empty((0,n_classes)).to(device)
true_label_test = torch.empty((0)).to(device)

with torch.no_grad():
  for data in testset:
    X_te, y_te = data
    X_te = X_te.view(batch_size,3,128,128).float().to(device)
    y_te = y_te.to(device)
    X_te = scatter(X_te).mean(axis=(3, 4)).to(device)
    output_test = model_test(X_te)
    pred_label_test = torch.cat((pred_label_test,output_test),dim=0)
    true_label_test = torch.cat((true_label_test,y_te),dim=0)

compute_metrics(y_true=true_label_test,y_pred=pred_label_test,lab_classes=lab_classes)    # function to compute the metrics (accuracy and confusion matrix)


# Plot the results
plt.figure(figsize=(8,5))
plt.plot(list(range(num_epochs)), losses)
plt.title("Learning curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(list(range(num_epochs)), acc_train)
plt.title("Accuracy curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()