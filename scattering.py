import kymatio.torch as kt
import utils_our
import torch
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# Load data
dataPath = './Data'
classes = ['dog', 'flower']
batch_size = 64
test_perc = .3

trainset, testset = utils_our.batcher(batch_size = batch_size, *train_test_split(*utils_our.loadData(dataPath, classes), test_size=test_perc))


J = 2
imageSize = (128, 128)
order = 2

scatter = kt.Scattering2D(J, shape = imageSize, max_order = order)
scatter = scatter.to(device)
print('Created scattering banks')

scatNet = []

for row in trainset:

    x, y = row
    x = x.view(batch_size,3,128,128).float().to(device)

    s_test = scatter(x)
    s_test = s_test.reshape(batch_size, 1, -1)

    print(s_test.device, s_test.shape)
    #scatNet += torch.aslist(s_test)

print(len(scatNet))


print('Executed scattering transform on data')

for i, j in enumerate(s_test):
    torch.save(j, f'./Data/scatters/{i}.pt')