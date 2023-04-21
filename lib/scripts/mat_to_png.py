import os, cv2
import scipy.io
import scipy.misc

input_path = 'Data\\new_dataset'
input_classes = ['pos', 'neg']

output_path = 'Data\\gray'
output_classes = ['amiloide', 'not_amiloide']

for c in range(len(input_classes)):
    
        p = os.path.join(input_path,input_classes[c])

        if not os.path.exists(os.path.join(output_path, output_classes[c])):
            os.makedirs(os.path.join(output_path, output_classes[c]))

        for i, file in enumerate(os.listdir(p)):

            # Read .mat files and convert to .png
            mat = scipy.io.loadmat(f'{os.path.join(p,file)}')
            cv2.imwrite(f'{os.path.join(output_path, output_classes[c], str(i))}.png', mat['corr_ms'])



            

