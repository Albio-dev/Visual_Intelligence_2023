import make_settings
import classification

test_percs = [0.1, 0.2, 0.3, 0.4, 0.5]

for test_perc in test_percs:
    make_settings.setGenericParameters('./Data', './train_checkpoint/', ['dog','flower'], 64, test_perc)
    classification.classification_task(display = False)