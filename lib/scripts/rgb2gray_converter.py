import os
from PIL import Image

path = '../../Data/'

# Load the data
for filename in os.listdir(path+'rgb/cat_cifar'):
    img = Image.open(path+'rgb/cat_cifar/'+filename)
    img = img.convert('L')
    img.save(path+'gray/cat_cifar/'+filename)
    