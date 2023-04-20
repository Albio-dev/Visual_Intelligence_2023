import configurable_classification
import lib.scripts.make_settings
import matlab.engine

parameters_to_modify = {
    'batch_size': [4, 8, 16, 128, 256]
}

for parameter in parameters_to_modify:
    for value in parameters_to_modify[parameter]:
        print('Classification model with {} = {}'.format(parameter, value))
        lib.scripts.make_settings.edit_parameter(parameter, value)
        configurable_classification.classify(cnn=False)
        #matlab.engine.quit_matlab()