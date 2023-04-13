import random
from lib import train_test
from lib.models.CNN_128x128 import CNN_128x128
from lib.models.NN_128x128 import NN_128x128
from legacy import utils_our
from lib.metrics import metrics as metrics
from lib import scatter_helper
from lib.data_handler import data_handler
from lib.cnn_explorer import explorer
from lib.scripts import make_settings

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

import lib.scripts.custom_augment as T

# Set device where to run the model. GPU if available, otherwise cpu (very slow with deep learning models)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device: ', device)
def classify(display = False):
    make_settings.writefile()
    settings = utils_our.load_settings()


    # Scatter creation
    scatter_params = utils_our.load_settings('scatter_parameters.yaml')
    
    scatter = scatter_helper.scatter(imageSize=settings['imageSize'], mode = 1, scatter_params=scatter_params)

    #Create KFold
    folds = settings['num_k_folds']
    kf = KFold(n_splits=folds)

    # Data loading
    data_path = settings['data_path']
    classes = settings['lab_classes']
    batch_size = settings['batch_size']
    test_perc = settings['test_perc']
    handler = data_handler(data_path, classes, batch_size, test_perc)
    handler.loadData(samples=settings['num_samples'])
    handler.to(device)
    results_path = settings['results_path']
    current_results_path = f"{results_path}{handler.get_folder_index(results_path)}"

    print(current_results_path)

    if not os.path.isdir(current_results_path):
        os.makedirs(current_results_path)

    n_augmentations = settings['augmentations']
    if n_augmentations > 1:
        handler.augment(augmentations=n_augmentations)

    # Get CNN dataset
    x_train, x_test, y_train, y_test = handler.get_data_split()
    #y_train = np.asarray(y_train)
    #y_test = np.asarray(y_test)
    _, testset = handler.batcher()

    
    # Plot training data
    training_fig, training_axs = plt.subplots(2, 2, figsize=(15, 10))
    training_fig.suptitle('Training infos')
    max_loss = 0
    min_acc = 500
    acc_cnn = []
    acc_nn = []

    '''
    policy = T.AutoAugmentPolicy.CUSTOM_POLICY
    augmenter = T.AutoAugment(policy).to(device)
    augmentation_amount = settings['augmentations']
    '''

    for i, (train_index,test_index) in enumerate(kf.split(x_train)):
        print(f"K-fold cycle {i+1}/{folds}")
        x_train_par, x_val, y_train_par, y_val = x_train[train_index], x_train[test_index], y_train[train_index], y_train[test_index]

        print(f"Train data size: {len(x_train_par)}")
        print(f"Validation data size: {len(x_val)}")
        '''
        if augmentation_amount > 0:
            aug_x_train_par = []
            #aug_x_val = []

            aug_y_train_par = []
            #aug_y_val = []

            for x, y in zip(x_train_par, y_train_par):
                augmentation_list = [torch.squeeze(augmenter(torch.unsqueeze(x, dim=0))) for _ in range(augmentation_amount)]
                augmentation_list.insert(0, x)
                aug_x_train_par += augmentation_list
                aug_y_train_par += [y] * (1 + augmentation_amount)

            #for x, y in zip(x_val, y_val):
            #    aug_x_val += [torch.squeeze(augmenter(torch.unsqueeze(x, dim=0))) for _ in range(augmentation_amount)]
            #    aug_y_val += [y] * augmentation_amount

            aug_train_lists = list(zip(aug_x_train_par, aug_y_train_par))
            random.shuffle(aug_train_lists)
            aug_x_train_par, aug_y_train_par = zip(*aug_train_lists)
            
            x_train_par = torch.stack(aug_x_train_par)
            y_train_par = torch.stack(aug_y_train_par)
            #x_val = torch.stack(aug_x_val)
            #y_val = torch.stack(aug_y_val)

        print(f"Augmented train data size: {len(x_train_par)}")
        print(f"Augmented validation data size: {len(x_val)}")
        
        aug_x_train_par = []
        aug_x_val = []
        aug_y_train_par = []
        aug_y_val = []
        
        for x, y in zip(x_train_par, y_train_par):
            aug_x_train_par += [torch.squeeze(augmenter(torch.unsqueeze((x), dim=0))).to(device) for _ in range(augmentation_amount)]
            aug_y_train_par += [y] * augmentation_amount
        
        for x, y in zip(x_val, y_val):
            aug_x_val += [torch.squeeze(augmenter(torch.unsqueeze((x), dim=0))).to(device) for _ in range(augmentation_amount)]
            aug_y_val += [y] * augmentation_amount

        aug_x_train_par = torch.stack(aug_x_train_par).to(device)
        aug_x_val = torch.stack(aug_x_val).to(device)
        aug_y_train_par = torch.stack(aug_y_train_par).to(device)
        aug_y_val = torch.stack(aug_y_val).to(device)'''
        

        #trainset, valset = handler.batcher(data=[aug_x_train_par, aug_x_val, aug_y_train_par, aug_y_val])        
        trainset, valset = handler.batcher(data=[x_train_par, x_val, y_train_par, y_val])
        
        
        # Model parameters
        classes = settings['lab_classes']
        channels = settings['channels']

        # Model creation
        CNN = CNN_128x128(input_channel=channels, num_classes=len(classes))
        
        # Optimizer parameters
        optimizer = settings['optimizer']
        learning_rate = settings['learning_rate']
        momentum = settings['momentum']
        weight_decay = settings['weight_decay']

        # Training parameters
        num_epochs = settings['num_epochs']
        NN_best_path = settings['model_train_path'] + 'NN_128x128_best_model_trained.pt'
        CNN_best_path = settings['model_train_path'] + 'CNN_128x128_best_model_trained.pt'
        epoch_val = settings['epoch_val']

        # Call the function in temp.py
        CNN_train_data = train_test.train(model = CNN, train_data=trainset, val_data = valset, num_epochs=num_epochs, best_model_path=CNN_best_path+str(i), device=device, 
                                          optimizer=optimizer, optimizer_parameters=(learning_rate, momentum, weight_decay),epoch_val= epoch_val)
        
        metrics.plotTraining(data = CNN_train_data, axs=training_axs[0][:], title = 'CNN', iteration=i, epochs_per_validation=epoch_val)
        
        # Decide scale
        max_loss = min(max(max_loss, max(CNN_train_data['loss']), max(CNN_train_data['loss_val'])), 1)#, max(NN_train_data['loss']), max(NN_train_data['loss_val']))
        min_acc = min(min_acc, min(CNN_train_data['accuracy']), min(CNN_train_data['accuracy_val']))#, min(NN_train_data['accuracy']), min(NN_train_data['accuracy_val']))

        acc_cnn.append(CNN_train_data['accuracy'])

    # Apply scale to graphs
    training_axs[0][0].set_ylim(0, max_loss)
    training_axs[0][1].set_ylim(min_acc, 1)
    training_axs[1][0].set_ylim(0, max_loss)
    training_axs[1][1].set_ylim(min_acc, 1)

    #training_fig.show()    
    training_fig.savefig(f"{current_results_path}/training_infos_{i}.png", dpi=300)
        
    # Load best models
    CNN.load_state_dict(torch.load(CNN_best_path + str(acc_cnn.index(max(acc_cnn)))))

    # Test models
    CNN_metrics = metrics(*train_test.test(model=CNN, test_data=testset, device=device), classes)
    
    # Print testing results
    CNN_metrics.printMetrics('CNN')
    
    # Plot confusion matrices
    fig, axs = plt.subplots(1, 1)
    fig.suptitle('Confusion matrices')
    CNN_metrics.confMatDisplay().plot(ax = axs)
    axs.set_title('CNN')
    fig.savefig(f"{current_results_path}/conf_mat.png", dpi=300)

    cnn_inspect = explorer(CNN)        
    fig = cnn_inspect.show_filters(current_results_path)
    #fig.show()
    fig.savefig(f"{current_results_path}/CNN_filters.png", dpi=300)    

    file = open(f"{current_results_path}/info.txt", 'w')
    file.write(f"{settings}\n{CNN_metrics.getMetrics(type='CNN')}\n")
    file.close()
  
    print("Done")



if __name__ == '__main__':
    classify(True)