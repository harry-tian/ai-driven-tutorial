from re import sub
from telnetlib import SB

from sklearn import datasets
import utils
import argparse, pickle
from pydoc import locate
import numpy as np
import torchvision, torch, os

bm = {"data_dir":"/net/scratch/tianh-shared/bm",
            "transform": "bm"}

bird = {"data_dir":"/net/scratch/tianh-shared/bird",
            "transform": "xray"}

prostatex = {"data_dir":"/net/scratch/tianh-shared/bird",
            "transform": "xray"}

wv = {"data_dir":"/net/scratch/chacha/data/weevil_vespula",
            "transform": "wv"}

def get_embeds(model_path, args, ckpt, split, data_dir, transform, embed_path):
    model = locate(model_path)
    model = model.load_from_checkpoint(ckpt, **vars(args)).to("cuda")
    model.eval()

    transform = utils.get_transform(transform, aug=False)
    print(split)
    data_dir = os.path.join(data_dir, split)
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    if len(dataset) <= 128:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=4)
        embeds = model.embed(list(iter(dataloader))[0][0].cuda())
        embeds = embeds.cpu().detach().numpy()
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4)
        i = 0
        for x, _ in dataloader:
            z = model.embed(x.cuda())
            pickle.dump(z.cpu().detach().numpy(),open(embed_path.replace(".pkl",f"_{i}.pkl"),"wb"))
            i += 1

        embeds = []
        for j in range(i):
            p = embed_path.replace(".pkl",f"_{j}.pkl")
            embed = pickle.load(open(p,"rb"))
            embeds.append(embed)
            os.remove(p)
        embeds = np.concatenate(embeds)
    
    
    print(f"embeds.shape:{embeds.shape}")
    pickle.dump(embeds, open(embed_path, "wb"))
    print(f"dumped to {embed_path}")


model_path = "resn_args.RESN"

# ckpt = 'baselines/5i3qt0bw'
ckpt = 'baselines/21tiniqd'


# dataset = bird
dataset = wv

# subdir = "bird"
subdir = "wv"
name = "RESN_split_emb10"
split = "test"





args = argparse.Namespace(embed_dim=10)
name = name.replace("split",split)
embed_path = f"../embeds/{subdir}/{name}.pkl"
ckpt = f"/net/scratch/chacha/explain_teach/models/results/{ckpt}/checkpoints/best_model.ckpt" 
get_embeds(model_path, args, ckpt, split, dataset["data_dir"], dataset["transform"], embed_path=embed_path)