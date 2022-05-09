import pickle, pathlib
import numpy as np
import torch
import torchvision
import lpips
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="lpips/bm", type=str, required=True)
parser.add_argument("--net", default="alex", type=str, required=True)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--batch_size", default=None, type=int)
args = parser.parse_args()
print(args)

loss_fn = lpips.LPIPS(net=args.net)
if args.gpu: loss_fn = loss_fn.cuda()

train_dataset = torchvision.datasets.ImageFolder("../datasets/bm/train", transform=torchvision.transforms.ToTensor())
valid_dataset = torchvision.datasets.ImageFolder("../datasets/bm/valid", transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder("../datasets/bm/test", transform=torchvision.transforms.ToTensor())
train_batch_size = len(train_dataset) if args.batch_size is None else args.batch_size
valid_batch_size = len(valid_dataset) if args.batch_size is None else args.batch_size
test_batch_size = len(test_dataset) if args.batch_size is None else args.batch_size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)

def get_dist(a, bs):
    if args.batch_size:
        dabs = []
        for i, b in enumerate(train_loader):
            if args.gpu:
                a, b = a.cuda(), b[0].cuda()
            dabs.append(loss_fn.forward(a, b, normalize=True).cpu().numpy().ravel())
        dabs = np.hstack(dabs)
    else:
        dl = torch.utils.data.DataLoader(bs, batch_size=len(bs))
        bs = next(iter(dl))[0]
        if args.gpu:
            a, bs = a.cuda(), bs.cuda()
        dabs = loss_fn.forward(a, bs, normalize=True).cpu().numpy().ravel()
    return dabs

with torch.no_grad():
    dttt, dvtt, dstt = [], [], []
    for i, batch in enumerate(train_dataset):
        if args.batch_size and i % 10 == 0: print("Example i:", i)
        dttt.append(get_dist(batch[0], train_dataset))
    print("train finished")
    for i, batch in enumerate(valid_dataset):
        if args.batch_size and i % 10 == 0: print("Example i:", i)
        dvtt.append(get_dist(batch[0], train_dataset))
    print("valid finished")
    for i, batch in enumerate(test_dataset):
        if args.batch_size and i % 10 == 0: print("Example i:", i)
        dstt.append(get_dist(batch[0], train_dataset))
    print("test finished")

mat_dttt = np.vstack(dttt)
mat_dvtt = np.vstack(dvtt)
mat_dstt = np.vstack(dstt)
print(mat_dttt.shape, mat_dvtt.shape, mat_dstt.shape)
path = "../embeds/" + args.path
pathlib.Path(path).mkdir(parents=True, exist_ok=True)
pickle.dump(mat_dttt, open(f"{path}/lpips.{args.net}.dttt.pkl", "wb"))
pickle.dump(mat_dvtt, open(f"{path}/lpips.{args.net}.dvtt.pkl", "wb"))
pickle.dump(mat_dstt, open(f"{path}/lpips.{args.net}.dstt.pkl", "wb"))