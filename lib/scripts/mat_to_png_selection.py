import os, cv2
import scipy.io
import scipy.misc


input_path = 'Data\\new_dataset'
input_classes = ['pos', 'neg']

output_path = 'Data\\gray'
output_classes = ['pos', 'neg']
allowed_lists = ['CN_neg.txt', 'MCI+AD_pos.txt']

for c in range(len(output_classes)):

    with open(os.path.join(input_path, allowed_lists[c]), 'r') as f:
        allowed = f.read().splitlines()
    index = 0

    if not os.path.exists(os.path.join(output_path, output_classes[c])):
        os.makedirs(os.path.join(output_path, output_classes[c]))

    for i in range(len(input_classes)):
        p = os.path.join(input_path,input_classes[i])

        for file in (os.listdir(p)):
            if '_'.join(file.split('_')[2:5])[:-4] in allowed:

                # Read .mat files and convert to .png
                mat = scipy.io.loadmat(f'{os.path.join(p,file)}')
                cv2.imwrite(f'{os.path.join(output_path, output_classes[c], str(index))}.png', mat['corr_ms'])

                index += 1



