import torch
import torchvision
import transforms

m = torchvision.models.resnet18(pretrained=True)

from torchvision.datasets import ImageFolder
train_ds = ImageFolder('../datasets/bm/train', transform=transforms.bird_transform())
valid_ds = ImageFolder('../datasets/bm/valid', transform=transforms.bird_transform())
test_ds = ImageFolder('../datasets/bm/test', transform=transforms.bird_transform())

def embed_dataset(m, dataset):
    m.eval()
    zs, dl = [], torch.utils.data.DataLoader(dataset, batch_size=m.hparams.train_batch_size)
    for x, _ in iter(dl): 
        zs.append(m(x.to(m.device)).cpu())
    return torch.cat(zs)
