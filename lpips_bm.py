import torch
import torchvision
import lpips
loss_fn = lpips.LPIPS(net='alex')
loss_fn.cuda()
# d = loss_fn.forward()

train_dataset = torchvision.datasets.ImageFolder("data/bm/train", transform=torchvision.transforms.ToTensor())
valid_dataset = torchvision.datasets.ImageFolder("data/bm/valid", transform=torchvision.transforms.ToTensor())
train_N = len(train_dataset)
valid_N = len(valid_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_N)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_N)
train_data = next(iter(train_loader))[0].cuda()
valid_data = next(iter(valid_loader))[0].cuda()

with torch.no_grad():
    d_train, d_valid = [], []
    for i in train_data:
        d_train.append(loss_fn.forward(i, train_data, normalize=True).cpu().numpy().ravel())
        d_valid.append(loss_fn.forward(i, valid_data, normalize=True).cpu().numpy().ravel())

import numpy as np
import pickle
print(np.vstack(d_train).shape, np.vstack(d_valid).shape)
pickle.dump(np.vstack(d_train), open("lpips.bm.train.pkl", "wb"))
pickle.dump(np.vstack(d_valid), open("lpips.bm.valid.pkl", "wb"))