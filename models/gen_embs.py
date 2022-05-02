from re import sub
from telnetlib import SB

from sklearn import datasets
import argparse, pickle
from pydoc import locate
import numpy as np
import torchvision, torch, os
import transforms

######### global variables ###########

bm = {"data_dir":"/net/scratch/tianh-shared/bm",
            "transform": "resn"}

bird = {"data_dir":"/net/scratch/tianh-shared/bird",
            "transform": "bird"}

chest_xray = {"data_dir":"/net/scratch/tianh-shared/PC/3classes",
            "transform": "resn"}

prostatex = {"data_dir":"/net/scratch/tianh-shared/bird",
            "transform": "xray"}

wv = {"data_dir":"/net/scratch/chacha/data/weevil_vespula",
            "transform": "wv"}

model_path_dict ={
    'TN': "TN.TN",
    'MTL': "MTL.MTL",
    'RESN': "RESN.RESN"
}

max_dataset = 30
def get_embeds(model_path, args, ckpt, split, data_dir, transform, embed_path):
    """generates an embedding given a model, its checkpoint, and a dataset

    Args:
        model_path (str): the model to generate the embedding
        args (dict): model args (placeholder)
        ckpt (str): model checkpoint
        split (str): \in {train, valid, test}
    """
    model = locate(model_path)
    model = model.load_from_checkpoint(ckpt, **vars(args)).to("cuda")
    model.eval()

    transform = transforms.get_transform(transform, aug=False)
    print(split)
    data_dir = os.path.join(data_dir, split)
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    if len(dataset) <= max_dataset:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=4)
        embeds = model(list(iter(dataloader))[0][0].cuda())
        embeds = embeds.cpu().detach().numpy()
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=max_dataset, num_workers=4)
        i = 0
        for x, _ in dataloader:
            z = model(x.cuda())
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


####### Tune variables here #############


model_name = "MTL"
ckpt = 'bm_prolific/azjw6n56' 

model_path = model_path_dict[model_name]

dataset = bm
subdir = "bm/prolific"
splits = ["train","test","valid"]



def main():
    args = argparse.Namespace(embed_dim=10)
    ckpt = f"results/{ckpt}/checkpoints/best_model.ckpt" 
    # ckpt = "/net/scratch/tianh/explain_teach/models/results/bm_prolific/zxkgmdyj/checkpoints/epoch=241-valid_loss=0.00.ckpt"
    for split in splits:
        name = f"{model_name}_{split}_emb10"
        embed_path = f"../embeds/{subdir}/{name}.pkl"

        get_embeds(model_path, args, ckpt, split, dataset["data_dir"], dataset["transform"], embed_path)

if __name__ == "__main__":
    main()
