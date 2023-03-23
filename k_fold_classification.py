from lib import train_test
from lib.models.CNN_128x128 import CNN_128x128
from lib.models.NN_128x128 import NN_128x128
from legacy import utils_our
from lib.metrics import metrics as metrics
from lib import scatter_helper
from lib.data_handler import data_handler
from lib.cnn_explorer import explorer
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import torch
import numpy as np
import os

# Set device where to run the model. GPU if available, otherwise cpu (very slow with deep learning models)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device: ', device)
def classify(display = False):

    settings = utils_our.load_settings()


    # Scatter creation
    scatter_params = utils_our.load_settings('scatter_parameters.yaml')
    
    scatter = scatter_helper.scatter(imageSize=settings['imageSize'], mode = 1, scatter_params=scatter_params)

    #Create KFold
    kf = KFold(n_splits=5)

    # Data loading
    data_path = settings['data_path']
    classes = settings['lab_classes']
    batch_size = settings['batch_size']
    test_perc = settings['test_perc']
    handler = data_handler(data_path, classes, batch_size, test_perc)
    handler.loadData(samples=settings['num_samples'])
    epoch_val = settings['epoch_val']
    results_path = settings['results_path']
    current_results_path = f"{results_path}{handler.get_folder_index(results_path)}"

    if not os.path.isdir(current_results_path):
        os.makedirs(current_results_path)


    # Get CNN dataset
    x_train, x_test, y_train, y_test = handler.get_data_split()
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    _, testset = handler.batcher()

    # Get NN dataset
    scatter_dataset = data_handler(data_path, classes, batch_size, test_perc, data = (scatter.scatter(handler.get_data()[0]), handler.get_data()[1]))
    x_train_scatter, x_test_scatter, _, _ = scatter_dataset.get_data_split()
    _, testset_scatter = scatter_dataset.batcher()

    for i, (train_index,test_index) in enumerate(kf.split(x_train)):
        x_train_par, x_val, y_train_par, y_val = x_train[train_index], x_train[test_index], y_train[train_index], y_train[test_index]
        trainset, valset = handler.batcher(data=[x_train_par, x_val, y_train_par, y_val])

        # get NN dataset
        x_train_scatter_par, x_scatter_val, _, _ = x_train_scatter[train_index], x_train_scatter[test_index], y_train[train_index], y_train[test_index]
        trainset_scatter, valset_scatter = handler.batcher(data=[x_train_scatter_par, x_scatter_val, y_train_par, y_val])


        # Model parameters
        classes = settings['lab_classes']
        channels = settings['channels']

        # Model creation
        CNN = CNN_128x128(input_channel=channels, 
                        num_classes=len(classes))
        NN = NN_128x128(input_channel=channels, num_classes=len(classes), data_size = np.prod(list(trainset_scatter)[0][0][0].shape))
        
        # Optimizer parameters
        learning_rate = settings['learning_rate']
        momentum = settings['momentum']

        # Training parameters
        num_epochs = settings['num_epochs']
        NN_best_path = settings['model_train_path']+'NN_128x128_best_model_trained.pt'
        CNN_best_path = settings['model_train_path']+'CNN_128x128_best_model_trained.pt'

        # Call the function in temp.py
        CNN_train_data = train_test.train(model = CNN, train_data=trainset, val_data = valset, num_epochs=num_epochs, best_model_path=CNN_best_path, device=device, optimizer_parameters=(learning_rate, momentum),epoch_val= epoch_val)
        NN_train_data = train_test.train(model = NN, train_data=trainset_scatter,val_data = valset_scatter, num_epochs=num_epochs, best_model_path=NN_best_path, device=device, optimizer_parameters=(learning_rate, momentum),epoch_val= epoch_val)

        # Load best models
        #NN.load_state_dict(torch.load(NN_best_path))
        CNN.load_state_dict(torch.load(CNN_best_path))

        # Plot training data
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Training infos')

        
        metrics.plotTraining(data = CNN_train_data, axs=axs[0][:])
        metrics.plotTraining(data = NN_train_data, axs=axs[1][:])
        # Decide scale
        max_loss = max(max(CNN_train_data['loss']), max(CNN_train_data['loss_val']))
        min_acc = min(min(CNN_train_data['accuracy']), min(CNN_train_data['accuracy_val']))
        # Apply scale to graphs
        axs[0][0].set_ylim(0, max_loss)
        axs[0][1].set_ylim(min_acc, 1)
        axs[1][0].set_ylim(0, max_loss)
        axs[1][1].set_ylim(min_acc, 1)
        fig.show()
        
        fig.savefig(f"{current_results_path}/training_infos_{i}.png", dpi=300)
        


        
    # Test models
    CNN_metrics = metrics(*train_test.test(model=CNN, test_data=testset, device=device), classes)
    NN_metrics = metrics(*train_test.test(model=NN, test_data=testset_scatter, device=device), classes)

    # Print testing results
    CNN_metrics.printMetrics('CNN')
    NN_metrics.printMetrics("NN")

    # Plot confusion matrices
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Confusion matrices')
    CNN_metrics.confMatDisplay().plot(ax = axs[0])
    axs[0].set_title('CNN')
    NN_metrics.confMatDisplay().plot(ax = axs[1])
    axs[1].set_title('NN')
    fig.show()
    fig.savefig(f"{current_results_path}/conf_mat.png", dpi=300)




    #cnn_inspect = explorer(CNN)        
    #fig = cnn_inspect.show_filters(current_results_path)
    #fig.show()
    #fig.savefig(f"{current_results_path}/CNN_filters.png", dpi=300)
    
    
    file = open(f"{current_results_path}/info.txt", 'w')
    file.write(f"{settings}\n{CNN_metrics.getMetrics(type='CNN')}") #\n{NN_metrics.getMetrics(type='NN')
    file.write(f'{scatter}')
    file.close()
    
    input()



if __name__ == '__main__':
    classify(True)