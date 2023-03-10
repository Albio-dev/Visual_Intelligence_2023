import lib.scripts.make_settings as make_settings
import classification

nums_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for num_epochs in nums_epochs:
    make_settings.setModelHyperparameters(0.01, 0.5, num_epochs)
    classification.classification_task(display = False)