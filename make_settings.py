import yaml

generic = {
    "data_path" : './Data',
    "model_train_path": './train_checkpoint/',

    # Classes in the dataset             
    "lab_classes" : ['dog','flower'],

    # How many samples are used per-iteration
    "batch_size" : 64,
    # Quantity of dataset used for the testing
    "test_perc" : .3
}

model_hyperparameters = {
    # Learning rate to scale how much new weighs are evaluated
    "learning_rate": 0.01,
    # Scale for past experience to not be perturbated by new ones
    "momentum" : 0.5,
    # The number of times the model is trained on the entire training dataset.
    "num_epochs" : 80    
}

scattering_parameters = {
    # TODO: ??
    "J" : 4,
    # Order of scattering
    "order" : 2,
    # Size of the input images
    "imageSize" : (128, 128),
    # Number of rotations
    "n_rotations" : 8
}

with open('parameters.yaml', 'w') as f:
    f.write(yaml.dump(generic))
    f.write(yaml.dump(model_hyperparameters))
    f.write(yaml.dump(scattering_parameters))
    

