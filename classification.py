import CNN
import NN_scattering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utils
from scipy.fft import fft2

from colorsys import hls_to_rgb

import make_settings


def showCNNFilters(model):
    layer = 1
    filter = list(model.children())[layer].weight.data.clone()
    utils.visTensor(filter, ch=0, allkernels=False)
    plt.axis('off')
    plt.ioff()
    plt.show()


def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0/(1.0 + abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c


def showPassBandScatterFilters():
    filters_set, J, order = NN_scattering.getFilterBank()

    fig, axs = plt.subplots(J, order, sharex=True, sharey=True)
    fig.set_figheight(6)
    fig.set_figwidth(6)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    i = 0
    for filter in filters_set['psi']:
        f = filter["levels"][0]
        filter_c = fft2(f)
        filter_c = np.fft.fftshift(filter_c)
        axs[i // order, i % order].imshow(colorize(filter_c))
        axs[i // order, i % order].axis('off')
        axs[i // order, i % order].set_title("$j = {}$ \n $\\theta={}$".format(i // order, i % order))
        i = i+1
    
    fig.suptitle((r"Wavelets for each scales $j$ and angles $\theta$ used."
                  "\nColor saturation and color hue respectively denote complex "
                  "magnitude and complex phase."), fontsize=13)
    plt.show()


def showLowPassScatterFilters():
    filters_set, _, _ = NN_scattering.getFilterBank()

    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.axis('off')
    plt.set_cmap('gray_r')

    f = filters_set['phi']["levels"][0]

    filter_c = fft2(f)
    filter_c = np.fft.fftshift(filter_c)
    plt.suptitle(("The corresponding low-pass filter, also known as scaling "
                  "function.\nColor saturation and color hue respectively denote "
                  "complex magnitude and complex phase"), fontsize=13)
    filter_c = np.abs(filter_c)
    plt.imshow(filter_c)

    plt.show()


def show_confusion_matrix(CNN_metrics,NN_metrics):

    fig, axs = plt.subplots(1, 2)
    CNN_metrics.confMatDisplay().plot(ax=axs[0])
    #sns.heatmap(confmat_CNN, ax=axs[0], annot=True)
    axs[0].set_title('confusion matrix in test for CNN')
    axs[0].set_xlabel('predicted')
    axs[0].set_ylabel('true')
    NN_metrics.confMatDisplay().plot(ax=axs[1])
    #sns.heatmap(confmat_scattering, ax=axs[1], annot=True)
    axs[1].set_title('confusion matrix in test for scattering NN')
    axs[1].set_xlabel('predicted')
    axs[1].set_ylabel('true')
    plt.show()

    
trainset_cnn, testset_cnn = CNN.getData()
trainset_scatter, testset_scatter = NN_scattering.getData()

if not CNN.isTrained():
    CNN.train(trainset_cnn)

if not NN_scattering.isTrained():
    NN_scattering.train(trainset_scatter)

CNN_metrics, CNN_model = CNN.test(testset_cnn)
NN_metrics = NN_scattering.test(testset_scatter)

#show_confusion_matrix(CNN_metrics, NN_metrics)
#showPassBandScatterFilters()
#showLowPassScatterFilters()
showCNNFilters(CNN_model)
