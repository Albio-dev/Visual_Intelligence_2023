import torchvision.utils
import torch
import matplotlib.pyplot as plt

class explorer:

    def __init__(self, model) -> None:
        self.model = model

    def getLayer(self, layer):
        return list(self.model.children())[layer].weight.data.cpu().clone()
    
    

    def visTensor(self, tensor, ax):
        n, c, w, h = tensor.shape

        tensor = tensor[:, 0, :, :].unsqueeze(dim=1)

        grid = torchvision.utils.make_grid(tensor, nrow=n//8, normalize=True, padding=1)
        ax.imshow(grid.cpu().numpy().transpose((1, 2, 0)))

    def show_filters(self,save_path):
        filters = sum([1 if type(i) == torch.nn.modules.conv.Conv2d else 0 for i in list(self.model.children())])
        
        fig, axs = plt.subplots(1, filters)
        for i in range(filters):
            self.visTensor(self.getLayer(i), axs[i])
        
        if save_path is not None:
            fig.savefig(f"{save_path}/CNN_filters.png", dpi=300)
        fig.show()



