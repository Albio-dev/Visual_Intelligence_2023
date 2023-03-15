from lib import train_test
from lib.models.CNN_128x128 import CNN_128x128
from lib.models.NN_128x128 import NN_128x128
from lib import utils_our
from lib.metrics import metrics as metrics
from lib import scatter_helper
import matplotlib.pylab as plt

import torch
import numpy as np

# Set device where to run the model. GPU if available, otherwise cpu (very slow with deep learning models)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device: ', device)
def classify(display = False):
    settings = utils_our.load_settings()

    # Scatter creation
    scatter_params = utils_our.load_settings('scatter_parameters.yaml')
    
    scatter = scatter_helper.scatter(imageSize=settings['imageSize'], mode = 1, scatter_params=scatter_params)

    # Data loading
    # TODO: load data using specific helper functions
    from lib.data_handler import data_handler
    data_path = settings['data_path']
    classes = settings['lab_classes']
    batch_size = settings['batch_size']
    test_perc = settings['test_perc']
    data_handler = data_handler(data_path, classes, batch_size, test_perc)
    data_handler.loadData(samples=200)
    trainset, testset = data_handler.batcher()

    # Getting scattering coefficients
    data, labels = data_handler.get_data()
    scatter_trainset, scatter_testset = data_handler.batcher(data = (scatter.scatter(data), labels))

    # Model parameters
    classes = settings['lab_classes']
    channels = settings['channels']

    # Model creation
    CNN = CNN_128x128(input_channel=channels, 
                      num_classes=len(classes))
    NN = NN_128x128(input_channel=channels, 
                    num_classes=len(classes), 
                    data_size = np.prod(list(scatter_trainset)[0][0][0].shape))
    
    # Optimizer parameters
    learning_rate = settings['learning_rate']
    momentum = settings['momentum']

    # Training parameters
    num_epochs = settings['num_epochs']
    NN_best_path = settings['model_train_path']+'NN_128x128_best_model_trained.pt'
    CNN_best_path = settings['model_train_path']+'CNN_128x128_best_model_trained.pt'

    # Call the function in temp.py
    CNN_train_data = train_test.train(model = CNN, train_data=trainset, num_epochs=num_epochs, best_model_path=CNN_best_path, device=device, optimizer_parameters=(learning_rate, momentum))
    NN_train_data = train_test.train(model = NN, train_data=scatter_trainset, num_epochs=num_epochs, best_model_path=NN_best_path, device=device, optimizer_parameters=(learning_rate, momentum))

    NN.load_state_dict(torch.load(NN_best_path))
    CNN.load_state_dict(torch.load(CNN_best_path))

    CNN_metrics = metrics(*train_test.test(model=CNN, test_data=testset, device=device), classes)
    NN_metrics = metrics(*train_test.test(model=NN, test_data=scatter_testset, device=device), classes)

    print(f'CNN: {CNN_metrics}')
    print(f'NN: {NN_metrics}')

    if display:
        
        # Plot training data
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Scatter filters')
        metrics.plotTraining(data = CNN_train_data, axs=axs[0][:])
        metrics.plotTraining(data = NN_train_data, axs=axs[1][:])
        max_loss = max(max(CNN_train_data['loss']), max(NN_train_data['loss']))
        min_acc = min(min(CNN_train_data['accuracy']), min(NN_train_data['accuracy']))
        axs[0][0].set_ylim(0, max_loss)
        axs[0][1].set_ylim(min_acc, 1)
        axs[1][0].set_ylim(0, max_loss)
        axs[1][1].set_ylim(min_acc, 1)
        fig.show()

        # Plot confusion matrices
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Confusion matrices')
        CNN_metrics.confMatDisplay().plot(ax = axs[0])
        axs[0].set_title('CNN')
        NN_metrics.confMatDisplay().plot(ax = axs[1])
        axs[1].set_title('NN')
        fig.show()
        
        input()



if __name__ == '__main__':
    classify()