# -*- coding: utf-8 -*-
import os, pickle
import argparse
from pathlib import Path
import numpy as np
import warnings
import torch
import utils
import torchvision
warnings.filterwarnings("ignore")


model_name = "TN_bm_triplet"
name = '' 

if model_name == "resnt":
    from resnt_args import RESN
# elif model_name == "triplet_subset":
#     from triplet_net_synth import TripletNet as RESN
# elif model_name == "triplet_net_food":
#     from triplet_net_food import TripletNet as RESN
elif model_name == "TN_bm_triplet":
    from TN_bm import TN_bm as RESN
# else:
#     from triplet_net_1_args import RESN
# else:
#     from triplet_net_2_args import TripletNet as RESN

ckpts = {"resnt":
            'results/resnt/1nxuz6dz/checkpoints/best_model.ckpt',
         "triplet_resn":
            'results/triplet/zyxkca5r/checkpoints/best_model.ckpt',
         "triplet_net":
            "results/triplet/2m2o143t/checkpoints/best_model.ckpt",
         "triplet_bs=32":
            "results/triplet/3682i8vx/checkpoints/best_model.ckpt",
         "triplet_bs=16":
            "results/triplet/ipcc8t03/checkpoints/best_model.ckpt",
         "triplet_bs=8":
            "results/triplet/x5kzln87/checkpoints/best_model.ckpt",
         "triplet_bs=40":
            "results/triplet/11njtozj/checkpoints/best_model.ckpt",
         "triplet_subset":
            "results/triplet/2lpqxv5u/checkpoints/best_model.ckpt",
         "triplet_net_food":
            "results/triplet/2qj4h4zk/checkpoints/epoch=16-valid_loss=0.00.ckpt",
         "TN_bm_triplet":
            "results/triplet_net_bm_triplet/d1cfb3qv/checkpoints/epoch=4-valid_loss=0.00.ckpt"}

ckpt = ckpts[model_name]
train_dir = '/net/scratch/hanliu-shared/data/bm/train'
valid_dir = '/net/scratch/hanliu-shared/data/bm/valid'
# args = argparse.Namespace(
#     train_dir=train_dir, pretrained=True,
#     valid_dir=valid_dir, embed_dim=10, train_batch_size=64)

args = argparse.Namespace(
    split_by="triplet")
model = RESN.load_from_checkpoint(ckpt, **vars(args))
_ = model.eval()

# batch = model.val_dataloader() if split == "valid" else model.train_dataloader()
# data_dir = "/net/scratch/hanliu-shared/data/bm/train"
data_dir = "/net/scratch/hanliu-shared/data/bm/valid"
dataset = torchvision.datasets.ImageFolder(data_dir, transform=utils.bm_train_transform())
dataset = torch.tensor(np.array([data[0].numpy() for data in dataset]))
        
#  print(type(batch))
#  print(batch.shape)
#  quit()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=4, shuffle=False)
embeds = []
for batch_idx, batch in enumerate(dataloader):
   #  inputs = [b[0] for b in batch]
   #  print(len(inputs[0]))
   # labels = [b[1] for b in batch]
   # print(input[0].shape) 
   embeds.append(model.embed(batch))
#  embeds = model.feature_extractor(inputs)
# for layer in model.fc:
#     embeds = [layer(e) for e in embeds]

#  data_dir = valid_dir if split == "valid" else train_dir
#  fids = sorted(os.listdir(data_dir+'/0')) + sorted(os.listdir(data_dir+'/1'))
#  fids = [fid.replace('.jpg', '') for fid in fids]
#  fids = np.asarray(fids)
embeds = np.asarray([e.squeeze().detach().numpy() for e in embeds])[0]
# inputs = np.asarray([i.squeeze().detach().numpy() for i in inputs])[0]
# labels = np.asarray([l.squeeze().detach().numpy() for l in labels])[0]
print(embeds.shape)
#  quit()

path = "../embeds/{}_{}_{}.pkl".format(model_name, "valid", name)
# pickle.dump((fids, inputs, labels, embeds), open(path, "wb"))
pickle.dump(embeds, open(path, "wb"))
print("dumped to {}".format(path))