import lib.scripts.make_settings as make_settings
import legacy.classification as classification

test_percs = [0.1, 0.2, 0.3, 0.4, 0.5]


for test_perc in test_percs:
    make_settings.setGenericParameters('./Data', './train_checkpoint/', ['dog','flower'], 128, test_perc, channels=3)
    classification.classification_task(display = False)

for test_perc in test_percs:
    make_settings.setGenericParameters('./Data', './train_checkpoint/', ['dog_gray','flower_gray'], 128, test_perc, channels=1)
    classification.classification_task(display = False)
