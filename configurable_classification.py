from lib.cnn_explorer import explorer
from lib.metrics import metrics
from lib.models.CNN_128x128 import CNN_128x128
from lib.models.NN_128x128 import NN_128x128
from lib.scripts import make_settings
from legacy import utils_our
from lib.data_handler import data_handler
from lib import train_test
from lib import scatter_helper

import torch
from sklearn.model_selection import KFold
import os
import matplotlib.pyplot as plt
import numpy as np

# Set device where to run the model. GPU if available, otherwise cpu (very slow with deep learning models)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

def classify(display = False, cnn = True, nn = True):

    # Update settings
    make_settings.writefile()
    settings = utils_our.load_settings()

    # Create folds if required
    folds = settings['num_k_folds']
    if folds > 1:
        kf = KFold(n_splits=folds)

    # Data loading
    data_path = settings['data_path']
    classes = settings['lab_classes']
    batch_size = settings['batch_size']
    test_perc = settings['test_perc']
    handler = data_handler(data_path, classes, batch_size, test_perc)
    # Subsample dataset
    handler.loadData(samples=settings['num_samples'])
    handler.to(device)

    # Prepare for saving new results
    results_path = settings['results_path']
    current_results_path = f"{results_path}{handler.get_folder_index(results_path)}"
    if not os.path.isdir(current_results_path):
        os.makedirs(current_results_path)

    # Clean meta data
    if not os.path.isdir(settings['model_train_path']):
        os.makedirs(settings['model_train_path'])
    else:            
        for i in os.listdir(settings['model_train_path']):
            os.remove(os.path.join(settings['model_train_path'], i))

    # Split train and test data
    x_train, x_test, y_train, y_test = handler.get_data_split()

    # Optional training data augmentation
    n_augmentations = settings['augmentations']
    if n_augmentations > 0:
        x_train, y_train = handler.augment(augmentations=n_augmentations, data = (x_train, y_train))
    
    # Prepare graphs for results display
    training_fig, training_axs = plt.subplots(2, cnn+nn, figsize=(15, 10))
    training_fig.suptitle('Training infos')
    max_loss = 0
    min_acc = float('inf')
    acc_cnn = []
    acc_nn = []

    # Plot confusion matrices
    confmat_fig, axs = plt.subplots(1, cnn+nn, figsize=(15, 10))
    confmat_fig.suptitle('Confusion matrices')

    
    # CNN training
    if cnn:    

        print(f'CNN training')
        _, testset = handler.batcher(data=(x_train, x_test, y_train, y_test))

        # Model parameters
        classes = settings['lab_classes']
        channels = settings['channels']
        
        # Optimizer parameters
        optimizer = settings['optimizer']
        learning_rate = settings['learning_rate']
        momentum = settings['momentum']
        weight_decay = settings['weight_decay']

        # Training parameters
        num_epochs = settings['num_epochs']
        CNN_best_path = settings['model_train_path'] + 'CNN_128x128_best_model_trained.pt'
        epoch_val = settings['epoch_val']

        if folds > 1:

            # Split data over folds
            for i, (train_index, val_index) in enumerate(kf.split(x_train)):

                # Extract fold data
                x_train_par, x_val, y_train_par, y_val = x_train[train_index], x_train[val_index], y_train[train_index], y_train[val_index]

                # Batch up fold data
                trainset, valset = handler.batcher(data=[x_train_par, x_val, y_train_par, y_val])
                
                print(f"K-fold cycle {i+1}/{folds}")
                print(f"Train data size: {len(x_train_par)}")
                print(f"Validation data size: {len(x_val)}")            

                # Model creation
                CNN = CNN_128x128(input_channel=channels, num_classes=len(classes))

                # Call the training function
                CNN_train_data = train_test.train(model = CNN, train_data=trainset, val_data = valset, num_epochs=num_epochs, best_model_path=CNN_best_path+str(i), device=device, 
                                                optimizer=optimizer, optimizer_parameters=(learning_rate, momentum, weight_decay),epoch_val= epoch_val)
                
                acc_cnn.append(CNN_train_data['accuracy'])
                
                if nn:
                    metrics.plotTraining(data = CNN_train_data, axs=training_axs[0][:], title = 'CNN', iteration=i, epochs_per_validation=epoch_val)
                else:
                    metrics.plotTraining(data = CNN_train_data, axs=training_axs, title = 'CNN', iteration=i, epochs_per_validation=epoch_val)
                
                # Save scale for graphs
                max_loss = min(max(max_loss, max(CNN_train_data['loss']), max(CNN_train_data['loss_val'])), 1)
                min_acc = min(min_acc, min(CNN_train_data['accuracy']), min(CNN_train_data['accuracy_val']))

        else:
            x_train, x_val, y_train, y_val = handler.get_data_split(data = (x_train, y_train), test_perc = 0.2)
            trainset, valset = handler.batcher(data=[x_train, x_val, y_train, y_val])

            print(f"Train data size: {len(x_train)}")
            print(f"Validation data size: {len(x_val)}")    

            # Model creation
            CNN = CNN_128x128(input_channel=channels, num_classes=len(classes))

            # Call the function in temp.py
            CNN_train_data = train_test.train(model = CNN, train_data=trainset, val_data = valset, num_epochs=num_epochs, best_model_path=CNN_best_path+'0', device=device, 
                                            optimizer=optimizer, optimizer_parameters=(learning_rate, momentum, weight_decay),epoch_val= epoch_val)
            
            acc_cnn.append(CNN_train_data['accuracy'])

            if nn:
                metrics.plotTraining(data = CNN_train_data, axs=training_axs[0][:], title = 'CNN', epochs_per_validation=epoch_val)
            else:
                metrics.plotTraining(data = CNN_train_data, axs=training_axs, title = 'CNN', epochs_per_validation=epoch_val)
            
            # Decide scale
            max_loss = min(max(max_loss, max(CNN_train_data['loss']), max(CNN_train_data['loss_val'])), 1)#, max(NN_train_data['loss']), max(NN_train_data['loss_val']))
            min_acc = min(min_acc, min(CNN_train_data['accuracy']), min(CNN_train_data['accuracy_val']))#, min(NN_train_data['accuracy']), min(NN_train_data['accuracy_val']))

            
        # Load best models
        CNN.load_state_dict(torch.load(CNN_best_path + str(acc_cnn.index(max(acc_cnn)))))

        # Test models
        CNN_metrics = metrics(*train_test.test(model=CNN, test_data=testset, device=device), classes)
        

        if not nn:
            CNN_metrics.confMatDisplay().plot(ax = axs)
            axs.set_title('CNN')
        else:
            CNN_metrics.confMatDisplay().plot(ax = axs[0])
            axs[0].set_title('CNN')
        
        cnn_inspect = explorer(CNN)        
        filters_fig = cnn_inspect.show_filters(current_results_path)
    




    # NN training
    if nn:
        # Model parameters
        classes = settings['lab_classes']
        channels = settings['channels']
        
        # Training parameters
        num_epochs = settings['num_epochs']
        NN_best_path = settings['model_train_path'] + 'NN_128x128_best_model_trained.pt'
        epoch_val = settings['epoch_val']

        # Optimizer parameters        
        optimizer = settings['optimizer']
        learning_rate = settings['learning_rate']
        momentum = settings['momentum']
        weight_decay = settings['weight_decay']

        # Create scatter objects
        scatter_params = utils_our.load_settings('scatter_parameters.yaml')    
        scatter = scatter_helper.scatter(imageSize=settings['imageSize'], mode = 1, scatter_params=scatter_params)
        
        # get NN dataset   
        scatter_x_train = scatter.scatter(x_train)
        scatter_x_test = scatter.scatter(x_test)
        _, testset_scatter = handler.batcher(data=[scatter_x_train, scatter_x_test, y_train, y_test])

        if folds > 1:

            # Split data over folds
            for i, (train_index, val_index) in enumerate(kf.split(scatter_x_train)):
                x_train_par, x_val, y_train_par, y_val = scatter_x_train[train_index], scatter_x_train[val_index], y_train[train_index], y_train[val_index]

                # Batch up fold data
                trainset_scatter, valset_scatter = handler.batcher(data=[x_train_par, x_val, y_train_par, y_val])

                print(f"K-fold cycle {i+1}/{folds}")
                print(f"Train data size: {len(x_train_par)}")
                print(f"Validation data size: {len(x_val)}")   
                

                NN = NN_128x128(input_channel=channels, 
                            num_classes=len(classes), 
                            data_size = np.prod(list(trainset_scatter)[0][0][0].shape))
                
                NN_train_data = train_test.train(model = NN, train_data=trainset_scatter,val_data = valset_scatter, num_epochs=num_epochs, best_model_path=NN_best_path+str(i), device=device, optimizer=optimizer, optimizer_parameters=(learning_rate, momentum, weight_decay),epoch_val= epoch_val)
                acc_nn.append(NN_train_data['accuracy'])
                
                if cnn:
                    metrics.plotTraining(data = NN_train_data, axs=training_axs[1][:], title = 'NN', iteration=i, epochs_per_validation=epoch_val)
                else:
                    metrics.plotTraining(data = NN_train_data, axs=training_axs, title = 'NN', iteration=i, epochs_per_validation=epoch_val)
                
                # Save scale for graphs
                max_loss = min(max(max_loss, max(NN_train_data['loss']), max(NN_train_data['loss_val'])), 1)
                min_acc = min(min_acc, min(NN_train_data['accuracy']), min(NN_train_data['accuracy_val']))

        else:
            scatter_x_train, scatter_x_val, y_train, y_val = handler.get_data_split(data = (scatter_x_train, y_train), test_perc = 0.2)
            trainset_scatter, valset_scatter = handler.batcher(data=[scatter_x_train, scatter_x_val, y_train, y_val])

            print(f"Train data size: {len(scatter_x_train)}")
            print(f"Validation data size: {len(scatter_x_val)}")


            NN = NN_128x128(input_channel=channels, 
                            num_classes=len(classes), 
                            data_size = np.prod(list(trainset_scatter)[0][0][0].shape))
            NN_train_data = train_test.train(model = NN, train_data=trainset_scatter,val_data = valset_scatter, num_epochs=num_epochs, best_model_path=NN_best_path+'0', device=device, optimizer=optimizer, optimizer_parameters=(learning_rate, momentum, weight_decay),epoch_val= epoch_val)
            acc_nn.append(NN_train_data['accuracy'])

            if cnn:
                metrics.plotTraining(data = NN_train_data, axs=training_axs[1][:], title = 'NN', epochs_per_validation=epoch_val)
            else:
                metrics.plotTraining(data = NN_train_data, axs=training_axs, title = 'NN', epochs_per_validation=epoch_val)
            
            # Decide scale
            max_loss = min(max(max_loss, max(NN_train_data['loss']), max(NN_train_data['loss_val'])), 1)
            min_acc = min(min_acc, min(NN_train_data['accuracy']), min(NN_train_data['accuracy_val']))


        
        # Load best models
        NN.load_state_dict(torch.load(NN_best_path + str(acc_nn.index(max(acc_nn)))))
        NN_metrics = metrics(*train_test.test(model=NN, test_data=testset_scatter, device=device), classes)

        if not cnn:
            NN_metrics.confMatDisplay().plot(ax = axs)
            axs.set_title('NN')
        else:
            NN_metrics.confMatDisplay().plot(ax = axs[1])
            axs[1].set_title('NN')







    # Apply scale to graphs
    if cnn and not nn or nn and not cnn:
        training_axs[0].set_ylim(0, max_loss)
        training_axs[1].set_ylim(min_acc, 1)

    if cnn and nn:
        training_axs[0][0].set_ylim(0, max_loss)
        training_axs[0][1].set_ylim(min_acc, 1)
        training_axs[1][0].set_ylim(0, max_loss)
        training_axs[1][1].set_ylim(min_acc, 1)

    training_fig.savefig(f"{current_results_path}/training_infos.png", dpi=300)
    confmat_fig.savefig(f"{current_results_path}/conf_mat.png", dpi=300)
    if cnn:        
        filters_fig.savefig(f"{current_results_path}/CNN_filters.png", dpi=300) 

    file = open(f"{current_results_path}/info.txt", 'w')
    file.write(f"{settings}\n")
    if cnn:
        file.write(f"{CNN_metrics.getMetrics(type='CNN')}\n")
    if nn: 
        file.write(f"{NN_metrics.getMetrics(type='NN')}\n")
    file.close()
    
    print("Done")
    
    # Display training results
    if display:
        training_fig.show()
        confmat_fig.show()
        if cnn:
            filters_fig.show()
        #input()
        plt.show()



if __name__ == '__main__':
    classify(display=True, cnn=True, nn=False)