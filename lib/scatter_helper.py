import numpy as np

import matplotlib.pyplot as plt

class scatter:

    # Modes for different scattering types
    # 0 - Kymatio
    # 1 - Matlab
    def __init__(self, imageSize, mode, scatter_params):
        # Decide whether to use kymatio or matlab
        self.mode = mode

        # Extract parameters from scatter_params
        J = scatter_params['J']
        num_rotations = scatter_params['num_rotations']


        if mode == 0:
            # TODO: properly link kymatio
            import kymatio as kt
            self.scatterBank = kt.filter_bank(imageSize[0], imageSize[1], J, L=num_rotations)
            self.scatterFunc = kt.Scattering2D(J, shape = imageSize, L=num_rotations)
            #scatters = utils_our.scatter_mem(batch_size,device,scatter,dataset, channels)
            #scatters = utils_our.load_scatter(data_path)
        else:
            # Extract matlab-spacific parameters
            quality_factors = scatter_params['quality_factors']

            # Initialize matlab engine
            import matlab.engine
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(r'lib/matlab')

            # Create scattering function
            self.scatterNet = self.eng.get_scatterNet(2**J, quality_factors, num_rotations, imageSize, nargout=1)
            self.scatterFunc = lambda x: self.eng.scattering(x, self.scatterNet, nargout=1)

            # Get informations about scattering
            _, _, filterbank = self.eng.filterbank(self.scatterNet, nargout=3)
            self.info = self.scatter_info(scatter = self.scatterNet, filterbank = filterbank, wavelets = [len(i) for i in filterbank], coefficients = self.eng.coefficientSize(self.scatterNet, nargout=1))

    # Scatter information class
    class scatter_info:

        def __init__(self, scatter, filterbank, wavelets, coefficients) -> None:
            self.scatter = scatter
            self.filterbank = [np.asarray(x) for x in filterbank]
            self.wavelets = np.asarray(wavelets)
            self.coefficients = np.asarray(coefficients)[0]

        # Return scattering info as string
        def __str__(self) -> str:
            return f'wavelets: {self.wavelets}\ncoefficients: {self.coefficients}'
        
        # Plot wavelets distribution
        def graph_wavelets(self):
            fig = plt.figure()
            fig.suptitle('Scatter filters')
            # Plot points for every layer
            for points in self.filterbank:            
                plt.scatter([x[0] for x in points], [x[1] for x in points])

            fig.legend([f"Filterbank level {i}" for i in range(len(self.filterbank))])
            fig.show()

    
    # Return scattering information class
    def get_info(self):
        return self.info

    # Compute scattering coefficients
    def scatter(self, data):
        return np.asarray(self.scatterFunc(self.eng.uint8(data)))#[np.asarray(x._data) for x in self.scatterFunc(data)]




def getData(batch_size, test_perc, data_path, lab_classes, J, num_rotations, imageSize, order, channels , train_scale = 1):
    # Split in train and test set
    #trainset, testset = utils_our.batcher(batch_size = batch_size, *train_test_split(*utils_our.loadData(data_path, lab_classes), test_size=test_perc))
    dataset = utils_our.loadData(path = data_path, folders = lab_classes)

    ### SCATTERING DATA ###
    #scatter = kt.Scattering2D(J, shape = imageSize, max_order = order, L=num_rotations)#, backend='torch_skcuda')
    #scatter = scatter.to(device)
    
    #print(f'Calculating scattering coefficients of data in {len(trainset)} batches of {batch_size} elements each for training')
    print('Calculating scattering coefficients of data')
    #scatters = utils_our.scatter_mem(batch_size,device,scatter,dataset, channels)
    #scatters = utils_our.load_scatter(data_path)
    s = data_path.split("/")
    sub_color = s[2]
    scatter = utils_our.matlab_scatter(sub_color, dataset, J, [4, 2], num_rotations)
    scatters = utils_our.load_scatter(data_path)
        
    if scatters is None:
        print('Error during scatter_mem!')
        sys.exit()
    print('Scattering coefficients calculated')

    xtrain, xtest, ytrain, ytest = utils_our.get_data_split(data = scatters, test_perc=test_perc, lab_classes=lab_classes, data_path=data_path)
    xtrain = xtrain[:int(len(xtrain)*train_scale)]
    ytrain = ytrain[:int(len(ytrain)*train_scale)]
    return *utils_our.batcher(xtrain, xtest, ytrain, ytest,batch_size= batch_size), np.prod(scatters[0][0].shape), scatter[1]

    #return *utils_our.batcher(*utils_our.get_data_split(data = scatters, test_perc=test_perc, lab_classes=lab_classes, data_path=data_path), batch_size = batch_size), np.prod(scatters[0][0].shape)

'''

s = data_path.split("/")
sub_color = s[2]
scatter = utils_our.matlab_scatter(sub_color, dataset, J, [4, 2], num_rotations)
scatters = utils_our.load_scatter(data_path)


xtrain, xtest, ytrain, ytest = utils_our.get_data_split(data = scatters, test_perc=test_perc, lab_classes=lab_classes, data_path=data_path)
xtrain = xtrain[:int(len(xtrain)*train_scale)]
ytrain = ytrain[:int(len(ytrain)*train_scale)]
return *utils_our.batcher(xtrain, xtest, ytrain, ytest,batch_size= batch_size), np.prod(scatters[0][0].shape), scatter[1]
'''