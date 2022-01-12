import numpy as np
import pickle, os
import torch
import torchvision
import lpips
loss_fn = lpips.LPIPS(net='alex')
loss_fn.cuda()
# d = loss_fn.forward()


train_dataset = torchvision.datasets.DatasetFolder("/data/4/all", extensions='npy', loader=np.load, transform=torchvision.transforms.ToTensor())
train_N = len(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_N)
train_data = next(iter(train_loader))[0].cuda()

print(train_data.shape)

with torch.no_grad():
    d_train = [], []
    for i in train_data:
        dist_train = [], []
        for d in range(7):
            a = torch.repeat_interleave(i[d].unsqueeze(0).unsqueeze(0), 3, dim=1)
            b = torch.repeat_interleave(train_data[:,d].unsqueeze(1), 3, dim=1)
            dist_train.append(loss_fn.forward(a, b, normalize=True).cpu().numpy().ravel())
        d_train.append(np.sum(dist_train, axis=0) / 7)

print(np.vstack(d_train).shape)
pickle.dump(np.vstack(d_train), open("lpips.prostatex.train.pkl", "wb"))