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


lpips_path = "/net/scratch/tianh/ai-driven-tutorial/data/dist/lpips"
data_path = "/net/scratch/tianh/ai-driven-tutorial/data/datasets"


train_dataset = torchvision.datasets.ImageFolder(f"{data_path}/{dataset}/train", transform=transform)
valid_dataset = torchvision.datasets.ImageFolder(f"{data_path}/{dataset}/valid", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(f"{data_path}/{dataset}/test", transform=transform)
train_batch_size = len(train_dataset) if not batch_size else batch_size 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)

for net in nets:
    loss_fn = lpips.LPIPS(net=net).cuda()

    def get_dist(a, bs):
        if batch_size:
            dabs = []
            for i, b in enumerate(train_loader):
                a, b = a.cuda(), b[0].cuda()
                dabs.append(loss_fn.forward(a, b, normalize=True).cpu().numpy().ravel())
            dabs = np.hstack(dabs)
        else:
            dl = torch.utils.data.DataLoader(bs, batch_size=len(bs))
            bs = next(iter(dl))[0]
            a, bs = a.cuda(), bs.cuda()
            dabs = loss_fn.forward(a, bs, normalize=True).cpu().numpy().ravel()
        return dabs

    with torch.no_grad():
        train, valid, test = [], [], []
        for i, batch in tqdm(enumerate(train_dataset)):
            train.append(get_dist(batch[0], train_dataset))
        print("train finished")
        for i, batch in tqdm(enumerate(valid_dataset)):
            valid.append(get_dist(batch[0], train_dataset))
        print("valid finished")
        for i, batch in tqdm(enumerate(test_dataset)):
            test.append(get_dist(batch[0], train_dataset))
        print("test finished")

    train = np.vstack(train)
    valid = np.vstack(valid)
    test = np.vstack(test)
    print(train.shape, valid.shape, test.shape)
    pathlib.Path(lpips_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(train, open(f"{lpips_path}/{dataset}/lpips.{net}.train.pkl", "wb"))
    pickle.dump(valid, open(f"{lpips_path}/{dataset}/lpips.{net}.valid.pkl", "wb"))
    pickle.dump(test, open(f"{lpips_path}/{dataset}/lpips.{net}.test.pkl", "wb"))