import CNN
import NN_scattering
import matplotlib.pyplot as plt
import seaborn as sns

import make_settings

trainset_cnn, testset_cnn = CNN.getData()
trainset_scatter, testset_scatter = NN_scattering.getData()

CNN.train(trainset_cnn)
NN_scattering.train(trainset_scatter)

confmat_CNN = CNN.test(testset_cnn)
confmat_scattering = NN_scattering.test(testset_scatter)

fig, axs = plt.subplots(1, 2)
sns.heatmap(confmat_CNN, ax=axs[0], annot=True)
axs[0].set_title('confusion matrix in test for CNN')
axs[0].set_xlabel('predicted')
axs[0].set_ylabel('true')
sns.heatmap(confmat_scattering, ax=axs[1], annot=True)
axs[1].set_title('confusion matrix in test for scattering NN')
axs[1].set_xlabel('predicted')
axs[1].set_ylabel('true')
fig.show()