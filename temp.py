import os
from PIL import Image

path = './Data'

# Load the data
for filename in os.listdir(path+'/flower'):
    img = Image.open(path+'/flower/'+filename)
    img = img.convert('L')
    img.save(path+'/flower_gray/'+filename)
    