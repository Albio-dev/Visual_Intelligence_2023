
import matplotlib.pyplot as plt
import os
import logging
import CNN
import NN_scattering
import make_settings
import utils_our

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler("log.txt")
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

def classification_task(display = True):
    settings = utils_our.load_settings()
        
    trainset_cnn, testset_cnn = CNN.getData(data_path=settings['data_path'], test_perc=settings['test_perc'], batch_size=settings['batch_size'], lab_classes=settings['lab_classes'], channels=settings['channels'], train_scale=1)
    trainset_scatter, testset_scatter, data_size, scatter = NN_scattering.getData(batch_size=settings['batch_size'], test_perc=settings['test_perc'], data_path=settings['data_path'], lab_classes=settings['lab_classes'], J=settings['J'], num_rotations=settings['n_rotations'], imageSize=settings['imageSize'], order=settings['order'], channels=settings['channels'], train_scale=1)

    
    if not CNN.isTrained(model_train_path=settings['model_train_path']):
        _, _, stats_CNN = CNN.train(trainset_cnn, learning_rate=settings['learning_rate'], num_epochs=settings['num_epochs'], batch_size=settings['batch_size'], model_train_path=settings['model_train_path'], lab_classes=settings['lab_classes'], momentum=settings['momentum'], channels=settings['channels'])

    if not NN_scattering.isTrained(model_train_path=settings['model_train_path']):
        stats_NN = NN_scattering.train(trainset_scatter, data_size = data_size, learning_rate=settings['learning_rate'], num_epochs=settings['num_epochs'], lab_classes=settings['lab_classes'], momentum=settings['momentum'], model_train_path=settings['model_train_path'], channels=settings['channels'])

    CNN_metrics, CNN_model = CNN.test(testset_cnn, model_train_path=settings['model_train_path'], lab_classes=settings['lab_classes'], batch_size=settings['batch_size'], channels=settings['channels'])
    NN_metrics = NN_scattering.test(testset_scatter, data_size, lab_classes=settings['lab_classes'], model_train_path=settings['model_train_path'], channels=settings['channels'])

    
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
        
        utils_our.display_stats_graphs(stats_CNN, stats_NN, settings['num_epochs'])

        #NN_scattering.showPassBandScatterFilters(J = settings['J'], num_rotations = settings['n_rotations'], imageSize= settings['imageSize'])
        CNN.showCNNFilters(CNN_model)

        NN_scattering.printScatterInfo(scatter, logger.info, display)
    else:
        NN_scattering.printScatterInfo(scatter, print)

if __name__ == "__main__":
    make_settings.writefile()
    classification_task(True)

def k_run(n):
    for i in range(n):
        classification_task(False)

