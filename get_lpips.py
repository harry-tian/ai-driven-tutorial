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


# nets = ["alex","vgg","squeeze"]
nets = ["alex"]


lpips_path = f"/net/scratch/tianh/ai-driven-tutorial/data/dist/lpips/{dataset}"
data_path = f"/net/scratch/tianh/ai-driven-tutorial/data/datasets/{dataset}"
train_dataset = torchvision.datasets.ImageFolder(f"{data_path}/train", transform=transform)
valid_dataset = torchvision.datasets.ImageFolder(f"{data_path}/valid", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(f"{data_path}/test", transform=transform)
batch_size = len(train_dataset) if not batch_size else batch_size 

def get_dist(dist_fn, target, dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    dist = [dist_fn.forward(target.cuda(), train[0].cuda(), normalize=True).cpu().numpy().ravel() for train in dataloader]
    return np.hstack(dist)



for net in nets:
    loss_fn = lpips.LPIPS(net=net).cuda()

    with torch.no_grad():
        train_dists = []
        for x,y in tqdm(train_dataset):
            train_dists.append(get_dist(loss_fn, x, train_dataset))
        print(f"train finished")

        test_dists = []
        for x,y in tqdm(test_dataset):
            test_dists.append(get_dist(loss_fn, x, train_dataset))
        print(f"test finished")

    train_dists = np.vstack(train_dists)
    pathlib.Path(lpips_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(train_dists, open(f"{lpips_path}/lpips.{net}.train.pkl", "wb"))


    test_dists = np.vstack(test_dists)
    pathlib.Path(lpips_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(test_dists, open(f"{lpips_path}/lpips.{net}.train_test.pkl", "wb"))


    # for split, dataset in [("train", train_dataset), ("valid", valid_dataset), ("test", test_dataset)]:
    
    # for split, dataset in [("valid", valid_dataset), ("test", test_dataset)]:
    #     with torch.no_grad():
    #         dists = []
    #         for x,y in dataset:
    #             # dists.append(get_dist(loss_fn, x, dataset))
    #             dists.append(get_dist(loss_fn, x, train_dataset))
    #         print(f"{split} finished")

    #     dists = np.vstack(dists)
    #     pathlib.Path(lpips_path).mkdir(parents=True, exist_ok=True)
    #     pickle.dump(dists, open(f"{lpips_path}/lpips.{net}.train_{split}.pkl", "wb"))