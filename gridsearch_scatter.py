import make_settings
import classification

js = [2, 3, 4, 5, 6, 7]
orders = [1, 2]
n_rotations = [4, 6, 8, 10]

for j in js:
    for order in orders:
        for n_rotation in n_rotations:
            make_settings.setScatteringParameters(j, order, (128, 128), n_rotation)
            classification.classification_task(display = False)