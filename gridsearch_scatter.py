import make_settings
import classification

js = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
orders = [2]
n_rotations = [8]

for j in js:
    for order in orders:
        for n_rotation in n_rotations:
            make_settings.setScatteringParameters(j, order, (128, 128), n_rotation)
            classification.classification_task(display = False)