from torch.utils.data import DataLoader
import glob
import cv2, os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from CNN_128x128 import CNN_128x128
from utils import CustomDataset, compute_metrics
import matplotlib.pyplot as plt


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
num_epochs = 30   



# number of classes in the dataset             
lab_classes = ['Dog','Flower']

### DATA LOADING ###



# Load dogs
dogs = [cv2.imread(file) for file in glob.glob(f'{data_path}/dog/*.jpg')]
# Create labels
dogs_labels = [0]*len(dogs)

# Load flowers
flowers = [cv2.imread(file) for file in glob.glob(f'{data_path}/flower/*.jpg')]
# Create labels
flowers_labels = [1]*len(flowers)

labels = dogs_labels + flowers_labels

# Split in train and test set
x_train, x_test, y_train, y_test = train_test_split(dogs+flowers, labels, test_size=test_perc)




# Set device where to run the model. GPU if available, otherwise cpu (very slow with deep learning models)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# Create Dataloader with batch size = 64
train_dataset = CustomDataset(x_train,y_train)    # we use a custom dataset defined in utils.py file
test_dataset = CustomDataset(x_test,y_test)       # we use a custom dataset defined in utils.py file


trainset = DataLoader(train_dataset,batch_size=batch_size,drop_last=True)    # construct the trainset with subjects divided in mini-batch
testset = DataLoader(test_dataset,batch_size=batch_size,drop_last=True)      # construct the testset with subjects divided in mini-batch



### HYPERPARAMETERS ###
# Define useful variables
best_acc = 0.0
n_classes = len(np.unique(labels))                # number of classes in the dataset


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