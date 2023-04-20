import yaml

generic = {
    "data_path" : './Data/gray',
    'channels' : 1,
    "model_train_path": './train_checkpoint/',

    # Classes in the dataset             
    "lab_classes" : ['amiloide','not_amiloide'],
    "results_path" : './results/',

    # How many samples are used per-iteration
    "batch_size" : 16,
    # Quantity of dataset used for the testing
    "test_perc" : .2,
    # Size of the input images
    "imageSize" : (100, 100),
    #The number of samples (images) used
    "num_samples" : 398,  #max 2774 (with flowers) # max 4100 with whales (3 class)
    # How many training epochs for every validation
    "epoch_val": 1,
    #The number of folds of KFold. 1 to disable
    "num_k_folds": 3,
    # Number of augmented images to produce. 0 to disable
    'augmentations': 0,
    #The weight decay used for the optimizer
    "weight_decay": 0.01,
    #The number of the optimizer that we want to use: 0- SGD, 1- Adam
    "optimizer": 0
}

model_hyperparameters = {
    # Learning rate to scale how much new weighs are evaluated
    "learning_rate": 0.0005,
    # Scale for past experience to not be perturbated by new ones
    "momentum" : 0.9,
    # The number of times the model is trained on the entire training dataset.
    "num_epochs" : 200
}

scattering_parameters = {
    # Invariance scale
    "J" : 6,
    # Order of scattering
    "order" : 2,
    # Number of rotations
    "num_rotations" : [2, 2],
    # Quality factors
    "quality_factors": [1,1]
}

def writefile():
    with open('parameters.yaml', 'w') as f:
        f.write(yaml.dump(generic))
        f.write(yaml.dump(model_hyperparameters))

    with open('scatter_parameters.yaml', 'w') as f:
        f.write(yaml.dump(scattering_parameters))

writefile()
    
def edit_parameter(parameter, value):
    if parameter in generic:
        generic[parameter] = value
    elif parameter in model_hyperparameters:
        model_hyperparameters[parameter] = value
    elif parameter in scattering_parameters:
        scattering_parameters[parameter] = value
    else:
        print("Parameter not found")
        return
    writefile()


