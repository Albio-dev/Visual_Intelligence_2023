
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler("log.txt")
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

def classification_task(display = True):
    import CNN
    import NN_scattering

    import yaml
    with open('parameters.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.loader.FullLoader)

        
    trainset_cnn, testset_cnn = CNN.getData()
    trainset_scatter, testset_scatter, data_size = NN_scattering.getData()

    #if not CNN.isTrained():
    CNN.train(trainset_cnn)

    #if not NN_scattering.isTrained():
    NN_scattering.train(trainset_scatter, data_size)

    CNN_metrics, CNN_model = CNN.test(testset_cnn)
    NN_metrics = NN_scattering.test(testset_scatter, data_size)


    if os.path.exists(f"{settings['model_train_path']}CNN_128x128_best_model_trained.pt"):
        os.remove(f"{settings['model_train_path']}CNN_128x128_best_model_trained.pt")
    if os.path.exists(f"{settings['model_train_path']}NN_128x128_best_model_trained.pt"):
        os.remove(f"{settings['model_train_path']}NN_128x128_best_model_trained.pt")
        


    

    # Write to file settings and metrics
    logger.info(settings)
    logger.info(f"{CNN_metrics.getMetrics(type='CNN')}")
    logger.info(f"{NN_metrics.getMetrics(type='NN')}")

    CNN_metrics.printMetrics("CNN")
    NN_metrics.printMetrics("NN")

    if display == True:
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title('confusion matrix in test for CNN')
        CNN_metrics.confMatDisplay().plot(ax=axs[0])
        axs[1].set_title('confusion matrix in test for scattering NN')
        NN_metrics.confMatDisplay().plot(ax=axs[1])
        plt.show()

        NN_scattering.showPassBandScatterFilters()
        CNN.showCNNFilters(CNN_model)

if __name__ == "__main__":
    classification_task(False)



