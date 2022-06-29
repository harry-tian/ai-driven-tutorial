import pickle, pathlib
import numpy as np
import torch
import torchvision
import lpips
from tqdm import tqdm
from models import transforms



dataset = "bird"
transform = transforms.bird_transform()
batch_size = 10






nets = ["alex","vgg","squeeze"]
# nets = ["vgg","squeeze"]


lpips_path = f"/net/scratch/tianh/ai-driven-tutorial/data/dist/lpips/{dataset}"
data_path = f"/net/scratch/tianh/ai-driven-tutorial/data/datasets/{dataset}"


train_dataset = torchvision.datasets.ImageFolder(f"{data_path}/train", transform=transform)
valid_dataset = torchvision.datasets.ImageFolder(f"{data_path}/valid", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(f"{data_path}/test", transform=transform)
train_batch_size = len(train_dataset) if not batch_size else batch_size 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)

for net in nets:
    loss_fn = lpips.LPIPS(net=net).cuda()

    def get_dist(dist_fn, target, train_loader):
        dist = [dist_fn.forward(target.cuda(), train[0].cuda(), normalize=True).cpu().numpy().ravel() for train in train_loader]
        return np.hstack(dist)

    # for split, dataset in [("train", train_dataset), ("valid", valid_dataset)]:
    for split, dataset in [("test", test_dataset)]:
        with torch.no_grad():
            dists = []
            for x,y in dataset:
                dists.append(get_dist(loss_fn, x, train_loader))
            print(f"{split} finished")

        dists = np.vstack(dists)
        pathlib.Path(lpips_path).mkdir(parents=True, exist_ok=True)
        pickle.dump(dists, open(f"{lpips_path}/lpips.{net}.{split}.pkl", "wb"))