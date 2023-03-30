import shutil, os

root_folder = "Data/gray"
folders = ["flower", "dog"]

for folder in folders:
    path = f'{root_folder}/{folder}'
    files = os.listdir(path)

    for i, file in enumerate(files):
        if file.endswith(".jpg"):
            shutil.copy(path+'/'+ f'{file}', path+'/new/'+ f"{i}.jpg")