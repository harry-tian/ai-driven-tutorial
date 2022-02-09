# -*- coding: utf-8 -*-
import os, pickle
import argparse
from pathlib import Path
import numpy as np
import warnings
import torch
warnings.filterwarnings("ignore")


model_name = "triplet_subset"
name = 'emb10.l10' 


if model_name == "resnt":
    from resnt_args import RESN
elif model_name == "triplet_subset":
    from triplet_net_subset import TripletNet as RESN
else:
    from triplet_net_1_args import RESN
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
            "results/triplet/2t12623u/checkpoints/best_model.ckpt"}

ckpt = ckpts[model_name]
train_dir = '/net/scratch/hanliu-shared/data/bm/train'
valid_dir = '/net/scratch/hanliu-shared/data/bm/valid'
args = argparse.Namespace(
    train_dir=train_dir,
    valid_dir=valid_dir, embed_dim=10, train_batch_size=160)
model = RESN.load_from_checkpoint(ckpt, **vars(args))
_ = model.eval()

for split in ["valid"]:
    print(f"generating embeddings for split: {split}")
    # batch = model.val_dataloader() if split == "valid" else model.train_dataloader()
    batch = model.get_valid_dataset()
    batch = list(iter(torch.utils.data.DataLoader(batch, batch_size=len(batch), num_workers=4, shuffle=False)))
    inputs = [b[0] for b in batch]
    # labels = [b[1] for b in batch]

    embeds = [model.feature_extractor(im) for im in inputs]
    # for layer in model.fc:
    #     embeds = [layer(e) for e in embeds]
    # embeds = [model.embed(im) for im in inputs]
    print((embeds[0].detach().numpy()).shape)

    data_dir = valid_dir if split == "valid" else train_dir
    fids = sorted(os.listdir(data_dir+'/0')) + sorted(os.listdir(data_dir+'/1'))
    fids = [fid.replace('.jpg', '') for fid in fids]
    fids = np.asarray(fids)
    embeds = np.asarray([e.squeeze().detach().numpy() for e in embeds])[0]
    inputs = np.asarray([i.squeeze().detach().numpy() for i in inputs])[0]
    # labels = np.asarray([l.squeeze().detach().numpy() for l in labels])[0]

    path = "../embeds/{}_{}_{}.pkl".format(model_name, split, name)
    # pickle.dump((fids, inputs, labels, embeds), open(path, "wb"))
    pickle.dump((fids, inputs, embeds), open(path, "wb"))
    print("Encoded {} findings at ".format(path))