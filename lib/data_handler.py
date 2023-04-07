from matplotlib import pyplot as plt
import numpy, cv2, glob, random, os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.transforms.autoaugment as autoaugment
from lib import custom_augment as T

class data_handler:

    def __init__(self, data_path, classes, batch_size, test_perc, channels = 1, samples = 1, data = None):
        self.seed = 42
        random.seed(self.seed)

        # Save parameters as class variables
        self.data_path = data_path
        self.classes = classes
        self.batch_size = batch_size
        self.test_perc = test_perc
        self.channels = channels

        # Load data from folders
        if data is None:
            self.data, self.labels = self.loadData(samples = samples)
            
        else:
            self.data, self.labels = data

        self.transforms = None

    

    # Load data from folders and eventually properly subsample equally on each class
    def loadData(self, samples = None):
        data = {}
        labels = {}

        # Read data from every folder
        for index, foldername in enumerate(self.classes):
            data[foldername] = [numpy.asarray(cv2.imread(file, cv2.IMREAD_UNCHANGED)) for file in glob.glob(f'{self.data_path}/{foldername}/*.*')]
            labels[foldername] = [index]*len(data[foldername])

        # Save full dataset as class variables
        self.raw_data = data
        self.raw_labels = labels

        # Extract samples in single list
        self.data = numpy.asarray([j for i in data.values() for j in i])
        self.labels = numpy.asarray([j for i in labels.values() for j in i])

        # If subsamples are required, return a random subsample, balanced on classes
        if samples is not None:
            
            new_data = {}
            new_labels = {}

            # Subdivide equally on classes
            samples_per_class = samples // len(self.classes)

            # Extract samples for every class
            for class_name in self.classes:

                # Create indeces for random subsample
                indeces = random.sample(range(len(data[class_name])), samples_per_class)

                # Create new lists with the subsamples
                new_data[class_name] = [data[class_name][i] for i in indeces]
                new_labels[class_name] = [labels[class_name][i] for i in indeces]

            self.raw_data = new_data
            self.raw_labels = new_labels
            
            # Extract samples as list
            self.data = numpy.asarray([j for i in new_data.values() for j in i])
            self.labels = numpy.asarray([j for i in new_labels.values() for j in i])


        # Return the data
        return self.data.reshape(-1, 1, *self.data.shape[1:]), self.labels

    # Write the data to a temporary folder
    def writeTempDataset(self):  
        for label in self.classes:
            if not os.path.exists(f'{self.data_path}/temp/{label}'):
                os.makedirs(f'{self.data_path}/temp/{label}')

            for num, i in enumerate(self.raw_data[label]):
                cv2.imwrite(f'{self.data_path}/temp/{label}/{num}.png', i)
                
    # Clean the temporary folder
    def deleteTempDataset(self):
        for label in self.classes:
            if os.path.exists(f'{self.data_path}/temp/{label}'):
                os.rmdir(f'{self.data_path}/temp/{label}')


    def get_data_split(self, test_perc = None, data = None):
        # Check random state
        shuffle = True

        # Use provided test percentage if provided
        if test_perc is None:
            test_perc = self.test_perc

        # If no data is provided, use the class data
        if data is None:            
            return train_test_split(self.data, self.labels, test_size=test_perc, random_state=self.seed, shuffle=shuffle)
            
        else:
            return train_test_split(data[0], data[1], test_size=test_perc, random_state=self.seed, shuffle=shuffle)

    # Create a custom dataset class
    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.labels = labels
            self.data = data

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            label = self.labels[idx]
            data = self.data[idx]
            sample = [data,label]
            return sample
        
    # Create a batcher for the data
    # Input required in order:
    # x_train, x_test, y_train, y_test
    def batcher(self, batch_size = 64, data = None):


        # If no data is provided, use the class data
        if data is None:
            x_train, x_test, y_train, y_test = self.get_data_split(test_perc = self.test_perc)
        
        # If data is in the form (data, labels)
        elif len(data) == 2:
            x_train, x_test, y_train, y_test = self.get_data_split(test_perc = self.test_perc, data = data)
        
        # If data is in the form (x_train, x_test, y_train, y_test)
        elif len(data) == 4:
            x_train, x_test, y_train, y_test = data

        # Reshape data as channels, height, width
        x_train = x_train.reshape(-1, self.channels, *x_train.shape[1:])
        x_test = x_test.reshape(-1, self.channels, *x_test.shape[1:])

        # Create Dataloader with batch size
        train_dataset = self.CustomDataset(x_train,y_train)    
        test_dataset = self.CustomDataset(x_test,y_test)       

        trainset = DataLoader(train_dataset,batch_size=batch_size,drop_last=False)    # construct the trainset with subjects divided in mini-batch
        testset = DataLoader(test_dataset,batch_size=batch_size,drop_last=False)      # construct the testset with subjects divided in mini-batch

        return trainset, testset
        
    # Getter for list data
    def get_data(self):
        return self.data, self.labels
    
    # Getter for raw data (class dictionaries)
    def get_raw_data(self):
        return self.raw_data, self.raw_labels
    
    
    def get_folder_index(self,path):
        try:
            return max([int(x) for x in os.listdir(path)]) + 1
        except:
            return 0
       
    def to(self, device):
        self.data = torch.tensor(self.data).to(device)
        self.labels = torch.tensor(self.labels).to(device)
        return self
    
    def get_augmentation_transforms(self):
        if True:
            op = random.randint(0, 6)
            if op == 0:
                self.transforms = torch.nn.Sequential(
                    transforms.RandomAffine(degrees=(15, 15)),
                )
            elif op == 1:
                self.transforms = torch.nn.Sequential(
                    transforms.RandomAffine(degrees=(30, 30)),
                )
            elif op == 2:
                self.transforms = torch.nn.Sequential(
                    transforms.RandomAffine(degrees=(45, 45)),
                )
            elif op == 3:
                self.transforms = torch.nn.Sequential(
                    transforms.RandomAffine(degrees=(60, 60)),
                )
            elif op == 4:
                self.transforms = torch.nn.Sequential(
                    transforms.RandomAffine(degrees=(75, 75)),
                )
            else:
                self.transforms = torch.nn.Sequential(
                    transforms.RandomAffine(degrees=(90, 90)),
                )
        return self.transforms


    def augment(self, augmentations = 2, data = None):
        if data is None:
            data = self.data
            labels = self.labels
        else:
            labels = data[1]
            data = data[0]
    
        #augmented_data = torch.squeeze(torch.cat([self.get_augmentation_transforms()(torch.unsqueeze(data, dim=1)) for _ in range(augmentations)]))

        
        policy = T.AutoAugmentPolicy.CUSTOM_POLICY
        augmenter = T.AutoAugment(policy)

        augmented_data = []
        augmented_labels = []

        if augmentations > 0:

            for x, y in zip(data, labels):
                augmented_data += [torch.squeeze(augmenter(torch.unsqueeze(x, dim=0))) for _ in range(augmentations)]
                augmented_labels += [y] * augmentations

            aug_train_lists = list(zip(augmented_data, augmented_labels))
            random.shuffle(aug_train_lists)
            augmented_data, augmented_labels = zip(*aug_train_lists)
            
            augmented_data = torch.stack(augmented_data)
            augmented_labels = torch.stack(augmented_labels)
        else:
            augmented_data = data
            augmented_labels = labels
        
        #labels = torch.repeat_interleave(labels, augmentations)

        plt.imshow(augmented_data[0].cpu())
        plt.savefig('testfig1.jpg')

        plt.imshow(augmented_data[1].cpu())
        plt.savefig('testfig2.jpg')

        plt.imshow(augmented_data[2].cpu())
        plt.savefig('testfig3.jpg')
        
        plt.imshow(augmented_data[3].cpu())
        plt.savefig('testfig4.jpg')
        
        plt.imshow(augmented_data[4].cpu())
        plt.savefig('testfig5.jpg')
        
        plt.imshow(augmented_data[5].cpu())
        plt.savefig('testfig6.jpg')
        
        plt.imshow(augmented_data[6].cpu())
        plt.savefig('testfig7.jpg')
        
        plt.imshow(augmented_data[7].cpu())
        plt.savefig('testfig8.jpg')

        self.data = augmented_data
        self.labels = augmented_labels
        return augmented_data
