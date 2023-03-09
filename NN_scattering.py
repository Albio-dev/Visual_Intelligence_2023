import numpy as np
import torch, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from NN_128x128 import NN_128x128
import matplotlib.pyplot as plt
import sys
from scipy.fft import fft2
import utils_our
import kymatio.torch as kt
from kymatio.scattering2d.filter_bank import filter_bank

# Set device where to run the model. GPU if available, otherwise cpu (very slow with deep learning models)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Available device: ', device)

### DATA LOADING ###

def train(trainset, data_size, learning_rate, momentum, num_epochs, lab_classes, model_train_path, channels):
    ### MODEL VARIABLES ###
    # Define useful variables
    best_acc = 0.0

    # Variables to store the results
    losses = []
    acc_train = []
    pred_label_train = torch.empty((0)).to(device)    # .to(device) to move the data/model on GPU or CPU (default)
    true_label_train = torch.empty((0)).to(device)


    ### CREATE MODEL ###

    # Model
    model = NN_128x128(input_channel=channels, num_classes=len(lab_classes), data_size = data_size ).to(device)

    # Optimizer
    optim = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    model.train()      

    ### FIT MODEL ###
    for epoch in range(num_epochs):
        # Train step                                  # tells to the model you are in training mode (batchnorm and dropout layers work)
        for data_tr in trainset:
            optim.zero_grad()

            x,y = data_tr                        # unlist the data from the train set  
            x = x.to(device)      
            y = y.to(device)

            y_pred = model(x)                                        # run the model
            loss = criterion(y_pred,y)                               # compute loss
            _,pred = y_pred.max(1)                                      # get the index == class of the output along the rows (each sample)
            pred_label_train = torch.cat((pred_label_train,pred),dim=0)
            true_label_train = torch.cat((true_label_train,y),dim=0)
            loss.backward()                                             # compute backpropagation
            optim.step()                        # parameter update

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


    return {'loss': losses, 'accuracy': acc_train}

def test(testset, data_size, lab_classes, model_train_path, channels):
    ### TEST MODEL ###
    model_test = NN_128x128(input_channel=channels,num_classes=len(lab_classes) , data_size=data_size).to(device)                # Initialize a new model
    model_test.load_state_dict(torch.load(model_train_path+'NN_128x128_best_model_trained.pt'))   # Load the model

    pred_label_test = torch.empty((0,len(lab_classes))).to(device)
    true_label_test = torch.empty((0)).to(device)

    model_test.eval()
    with torch.no_grad():
        for data in testset:
            X_te, y_te = data
            X_te = X_te.to(device)
            y_te = y_te.to(device)
            output_test = model_test(X_te)
            pred_label_test = torch.cat((pred_label_test,output_test),dim=0)
            true_label_test = torch.cat((true_label_test,y_te),dim=0)

    return utils_our.metrics(true_label_test.cpu(), pred_label_test.cpu(), lab_classes)#compute_metrics(y_true=true_label_test,y_pred=pred_label_test,lab_classes=lab_classes)    # function to compute the metrics (accuracy and confusion matrix)


def isTrained(model_train_path):
    return os.path.isfile(model_train_path+'NN_128x128_best_model_trained.pt')


def getData(batch_size, test_perc, data_path, lab_classes, J, num_rotations, imageSize, order, channels, train_scale = 1):
    # Split in train and test set
    #trainset, testset = utils_our.batcher(batch_size = batch_size, *train_test_split(*utils_our.loadData(data_path, lab_classes), test_size=test_perc))
    dataset = utils_our.loadData(path = data_path, folders = lab_classes)

    ### SCATTERING DATA ###
    #scatter = kt.Scattering2D(J, shape = imageSize, max_order = order, L=num_rotations)#, backend='torch_skcuda')
    #scatter = scatter.to(device)
    
    #print(f'Calculating scattering coefficients of data in {len(trainset)} batches of {batch_size} elements each for training')
    print('Calculating scattering coefficients of data')
    #scatters = utils_our.scatter_mem(batch_size,device,scatter,dataset, channels)
    #scatters = utils_our.load_scatter(data_path)
    s = data_path.split("/")
    sub_color = s[2]
    scatter = utils_our.matlab_scatter(sub_color, dataset, J, [3, 1], num_rotations)
    scatters = utils_our.load_scatter(data_path)
        
    if scatters is None:
        print('Error during scatter_mem!')
        sys.exit()
    print('Scattering coefficients calculated')

    xtrain, xtest, ytrain, ytest = utils_our.get_data_split(data = scatters, test_perc=test_perc, lab_classes=lab_classes, data_path=data_path)
    xtrain = xtrain[:int(len(xtrain)*train_scale)]
    ytrain = ytrain[:int(len(ytrain)*train_scale)]
    return *utils_our.batcher(xtrain, xtest, ytrain, ytest,batch_size= batch_size), np.prod(scatters[0][0].shape), scatter[1]

    #return *utils_our.batcher(*utils_our.get_data_split(data = scatters, test_perc=test_perc, lab_classes=lab_classes, data_path=data_path), batch_size = batch_size), np.prod(scatters[0][0].shape)

def printScatterInfo(scatter, print_func = print, graphs = False):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    _,_, scatnet = eng.filterbank(scatter, nargout=3)
    
    print_func('\t'.join([f'filterbank level {i}: {len(scatnet[i])} wavelets' for i in range(len(scatnet))]))
    print_func(f'Coefficient size: {eng.coefficientSize(scatter, nargout=1)}')
    print_func(f'Number of wavelet decompositions: {sum(np.asarray(eng.paths(scatter, nargout=2)[1]))}')

    if graphs:
        fig = plt.figure()
        fig.suptitle('Scatter filters')
        for index, points in enumerate(scatnet):            
            plt.scatter([x[0] for x in points], [x[1] for x in points])

        fig.legend([f"Filterbank level {i}" for i in range(len(scatnet))])
        fig.show()


def showPassBandScatterFilters(imageSize=(128, 128), J=3, num_rotations=8):
    filters_set, J, rotations = getFilterBank(imageSize=imageSize, J=J, num_rotations=num_rotations)

    fig, axs = plt.subplots(J, rotations, sharex=True, sharey=True)
    fig.set_figheight(6)
    fig.set_figwidth(6)
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    i = 0
    for filter in filters_set['psi']:
        f = filter["levels"][0]
        filter_c = fft2(f)
        filter_c = np.fft.fftshift(filter_c)
        axs[i // rotations, i % rotations].imshow(utils_our.colorize(filter_c))
        axs[i // rotations, i % rotations].axis('off')
        axs[i // rotations, i % rotations].set_title("j = {} \n theta={}".format(i // rotations, i % rotations))
        i = i+1
    
    plt.suptitle(("Wavelets for each scale j and angle theta used."
                  "\nColor saturation and color hue respectively denote complex "
                  "magnitude and complex phase."), fontsize=13)
    plt.show()


def showLowPassScatterFilters(imageSize=(128, 128), J=3, num_rotations=8):
    filters_set, _, _ = getFilterBank(imageSize=imageSize, J=J, num_rotations=num_rotations)

    plt.figure()
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    plt.axis('off')
    plt.set_cmap('gray_r')

    f = filters_set['phi']["levels"][0]

    filter_c = fft2(f)
    filter_c = np.fft.fftshift(filter_c)
    plt.suptitle(("The corresponding low-pass filter, also known as scaling "
                  "function.\nColor saturation and color hue respectively denote "
                  "complex magnitude and complex phase"), fontsize=13)
    filter_c = np.abs(filter_c)
    plt.imshow(filter_c)

    plt.show()


def getFilterBank(imageSize=(128, 128), J=3, num_rotations=8):
    return filter_bank(imageSize[0], imageSize[1], J, L=num_rotations), J, num_rotations


if __name__ == "__main__":
    settings = utils_our.load_settings()
    
    trainset, testset, data_size, scatter = getData(batch_size=settings['batch_size'], test_perc=settings['test_perc'], data_path=settings['data_path'], lab_classes=settings['lab_classes'], J=settings['J'], num_rotations=settings['n_rotations'], imageSize=settings['imageSize'], order=settings['order'], channels=settings['channels'])
    
    print(f'Scatter data size: {data_size}')
    stats_NN = train(trainset, data_size = data_size, learning_rate=settings['learning_rate'], num_epochs=settings['num_epochs'], lab_classes=settings['lab_classes'], momentum=settings['momentum'], model_train_path=settings['model_train_path'], channels=settings['channels'])
    metrics = test(testset,data_size, lab_classes=settings['lab_classes'], model_train_path=settings['model_train_path'], channels=settings['channels'])       
    
    metrics.confMatDisplay().plot()
    print(metrics)
    plt.show()
    #printScatterInfo(scatter, print)
    #utils_our.display_stats_graphs(stats_CNN= None, stats_NN= stats_NN ,epochs= settings['num_epochs'])
    #printScatterInfo(scatter, print)
    #utils_our.display_stats_graphs(stats_CNN= None, stats_NN= stats_NN ,epochs= settings['num_epochs'])