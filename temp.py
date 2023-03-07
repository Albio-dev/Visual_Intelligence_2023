import os
from PIL import Image

path = './Data'

# Load the data
for filename in os.listdir(path+'/rgb/whale'):
    img = Image.open(path+'/rgb/whale/'+filename)
    img = img.convert('L')
    img.save(path+'/gray/whale/'+filename)
    