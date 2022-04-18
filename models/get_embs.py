from re import sub
from telnetlib import SB

from sklearn import datasets
import utils
import argparse, pickle
from pydoc import locate
import numpy as np
import torchvision, torch, os
from sklearn.metrics.pairwise import euclidean_distances

bm = {"data_dir":"/net/scratch/tianh-shared/bm",
            "transform": "bm"}

bird = {"data_dir":"/net/scratch/tianh-shared/bird",
            "transform": "xray"}

chest_xray = {"data_ddir":"/net/scratch/tianh-shared/chest_xray",
            "data_dir":"/net/scratch/tianh-shared/NIH/4classes/auto_split",
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


# model_path = "RESN.RESN"


# <<<<<<< HEAD
# # ckpt = 'baselines/5i3qt0bw'
# ckpt = 'baselines/21tiniqd'


# # dataset = bird
# dataset = wv

# # subdir = "bird"
# subdir = "wv"
# ckpt = 'resn/u9as2wx6'
model_name = "MTL"
ckpt = 'synthetic_MTL/11hwl4ch' #synthetic_MTL/3sk1rynk' #'chacha-syn-htriplets/2ul5cslk'

model_path_dict ={
    'TN': "TN.TN",
    'MTL': "MTL_RESNTN.MTL_RESNTN",
    'RESN': "RESN.RESN"
}
model_path = model_path_dict[model_name]#"MTL_RESNTN.MTL_RESNTN" #"RESN.RESN"

# dataset = chest_xray
# subdir = "chest_xray"
dataset = wv
subdir = "wv"
splits = ["train","test","valid"]


# model_ckpt = '/net/scratch/chacha/explain_teach/models/results/synthetic_MTL/3sk1rynk/checkpoints/best_model.ckpt.'



args = argparse.Namespace(embed_dim=10)
# <<<<<<< HEAD
# ckpt = f"/net/scratch/chacha/explain_teach/models/results/{ckpt}/checkpoints/best_model.ckpt" 
# get_embeds(model_path, args, ckpt, split, dataset["data_dir"], dataset["transform"], embed_path=embed_path)
# =======
ckpt = f"results/{ckpt}/checkpoints/best_model.ckpt" 
for split in splits:
    # name = name.replace("split",split)
    name = f"{model_name}_{split}_emb10"
    embed_path = f"../embeds/{subdir}/{name}_lambda_1.pkl"

    get_embeds(model_path, args, ckpt, split, dataset["data_dir"], dataset["transform"], embed_path)
# >>>>>>> 03f67bcb1ecf5f92198e5eb45357b8e23d90d7fa
