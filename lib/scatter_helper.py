import numpy as np

import matplotlib.pyplot as plt
import torch

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
        print("Scattering data...")
        print(f'Scattering {data.shape[0]} images of size {data.shape[2]}x{data.shape[3]}')
        device = data.device
        return torch.Tensor(self.scatterFunc(self.eng.uint8(np.asarray(data.cpu())))).to(device)


