import numpy, cv2, glob, random, os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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

    

    # Load data from folders and eventually properly subsample equally on each class
    def loadData(self, samples = None):
        data = {}
        labels = {}

        # Read data from every folder
        for index, foldername in enumerate(self.classes):
            data[foldername] = [numpy.asarray(cv2.imread(file, cv2.IMREAD_UNCHANGED)) for file in glob.glob(f'{self.data_path}/{foldername}/*.jpg')]
            labels[foldername] = [index]*len(data[foldername])

        # Save full dataset as class variables
        self.raw_data = data
        self.raw_labels = labels

        # Extract samples in single list
        self.data = numpy.asarray([j for i in data.values() for j in i])
        self.labels = [j for i in labels.values() for j in i]

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
            self.labels = [j for i in new_labels.values() for j in i]

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
        return str(len(os.listdir(path)) + 1)