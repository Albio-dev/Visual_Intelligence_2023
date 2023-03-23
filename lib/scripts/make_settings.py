import yaml

generic = {
    "data_path" : './Data/gray',
    'channels' : 1,
    "model_train_path": './train_checkpoint/',

    # Classes in the dataset             
    "lab_classes" : ['flower','dog'],
    
    "results_path" : './results/',

    # How many samples are used per-iteration
    "batch_size" : 64,
    # Quantity of dataset used for the testing
    "test_perc" : .2,
    # Size of the input images
    "imageSize" : (128, 128),
    #The number of samples (images) used
    "num_samples" : 500,  #max 2774 (with flowers)
    # How many training epochs for every validation
    "epoch_val": 10


}

model_hyperparameters = {
    # Learning rate to scale how much new weighs are evaluated
    "learning_rate": 0.001,
    # Scale for past experience to not be perturbated by new ones
    "momentum" : 0.5,
    # The number of times the model is trained on the entire training dataset.
    "num_epochs" : 100
}

scattering_parameters = {
    # Invariance scale
    "J" : 6,
    # Order of scattering
    "order" : 2,
    # Number of rotations
    "num_rotations" : [8, 8],
    # Quality factors
    "quality_factors": [2, 1]
}

def writefile():
    with open('parameters.yaml', 'w') as f:
        f.write(yaml.dump(generic))
        f.write(yaml.dump(model_hyperparameters))

    with open('scatter_parameters.yaml', 'w') as f:
        f.write(yaml.dump(scattering_parameters))

writefile()
    
def setScatteringParameters(J, order, imageSize, n_rotations):
    scattering_parameters['J'] = J
    scattering_parameters['order'] = order
    scattering_parameters['imageSize'] = imageSize
    scattering_parameters['n_rotations'] = n_rotations

    writefile()    

def setGenericParameters(data_path, model_train_path, lab_classes, batch_size, test_perc, channels):
    generic['data_path'] = data_path
    generic['model_train_path'] = model_train_path
    generic['lab_classes'] = lab_classes
    generic['batch_size'] = batch_size
    generic['test_perc'] = test_perc
    generic['channels'] = channels

    writefile()

def setModelHyperparameters(learning_rate, momentum, num_epochs):
    model_hyperparameters['learning_rate'] = learning_rate
    model_hyperparameters['momentum'] = momentum
    model_hyperparameters['num_epochs'] = num_epochs

    writefile()

