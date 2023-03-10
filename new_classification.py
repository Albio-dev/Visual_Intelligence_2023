from lib import train_test
from lib.models import NN_128x128, CNN_128x128
from lib import utils_our
from lib.metrics import metrics as metrics

import torch

# Set device where to run the model. GPU if available, otherwise cpu (very slow with deep learning models)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device: ', device)
def classify(display = False):
    settings = utils_our.load_settings()

    # Data loading
    # TODO: load data using specific helper functions

    # Model parameters
    classes = settings['lab_classes']
    channels = settings['channels']

    # Model creation
    CNN = CNN_128x128(input_channel=channels, 
                      num_classes=len(classes)).to(device)
    NN = NN_128x128(input_channel=channels, 
                    num_classes=len(classes), 
                    data_size = data_size).to(device)
    
    # Optimizer parameters
    learning_rate = settings['learning_rate']
    momentum = settings['momentum']

    # Training parameters
    num_epochs = settings['num_epochs']
    NN_best_path = settings['model_train_path']+'NN_128x128_best_model_trained.pt'
    CNN_best_path = settings['model_train_path']+'CNN_128x128_best_model_trained.pt'

    # Call the function in temp.py
    train_test.train(model = CNN, train_data=trainset, num_epochs=num_epochs, best_model_path=CNN_best_path, device=device, optimizer_parameters=(learning_rate, momentum))
    train_test.train(model = NN, train_data=trainset, num_epochs=num_epochs, best_model_path=NN_best_path, device=device, optimizer_parameters=(learning_rate, momentum))

    best_NN = NN.load_state_dict(torch.load(NN_best_path))
    best_CNN = CNN.load_state_dict(torch.load(CNN_best_path))

    CNN_metrics = metrics(*train_test.test(model=best_CNN, test_data=testset, device=device), classes)
    NN_metrics = metrics(*train_test.test(model=best_NN, test_data=testset, device=device), classes)

    if display:
