import os, cv2
import scipy.io
import scipy.misc

classes = ['amiloide', 'not_amiloide']

path = 'Data\\gray'

for c in classes:
    
        p = os.path.join(path,c)
    
        for i, file in enumerate(os.listdir(p)):

            # Read .mat files and convert to .png
            mat = scipy.io.loadmat(f'{os.path.join(p,file)}')
            cv2.imwrite(f'{os.path.join(p,str(i))}.png', mat['corr_ms']*255)

            

