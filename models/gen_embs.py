import argparse, pickle
from pydoc import locate
import numpy as np
import torchvision, torch, os, pathlib
import transforms

######### global variables ###########

DATASETS = {}

DATASETS["bm"] = {"data_dir":"../datasets/bm",
            "transform": "resn"}

DATASETS["bird"] = {"data_dir":"/net/scratch/tianh-shared/bird",
            "transform": "bird"}

DATASETS["chest_xray"] = {"data_dir":"/net/scratch/tianh-shared/PC/3classes",
            "transform": "resn"}

DATASETS["prostatex"] = {"data_dir":"/net/scratch/tianh-shared/bird",
            "transform": "xray"}

DATASETS["wv"] = {"data_dir":"../datasets/weevil_vespula",
            "transform": "wv"}

model_path_dict ={
    'TN': "TN.TN",
    'MTL': "MTL.MTL",
    'RESN': "RESN.RESN",
    'MTL_han': "MTL_han.MTL"
}

max_dataset = 30
def get_embeds(model_path, ckpt, split, data_dir, transform, embed_path):
    """generates an embedding given a model, its checkpoint, and a dataset

    Args:
        model_path (str): the model to generate the embedding
        ckpt (str): model checkpoint
        split (str): \in {train, valid, test}
    """
    model = locate(model_path)
    model = model.load_from_checkpoint(ckpt).to("cuda")
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


# model_name = "RESN"
# wandb_group = "wv_2d"
# wandb_run = "2wql6bo5"
# ckpt_infix =  "/".join([wandb_group, wandb_run])
# suffix = "emb50"

# dataset = DATASETS['wv']
# subdir = "wv_2d/pretrained"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="RESN", type=str, required=True)
parser.add_argument("--wandb_group", default="wv_2d", type=str, required=True)
parser.add_argument("--wandb_name", default="", type=str, required=False)
parser.add_argument("--wandb_run", default="2wql6bo5", type=str, required=True)
parser.add_argument("--suffix", default="emb50", type=str, required=True)
parser.add_argument("--dataset", default="wv", type=str, required=True)
parser.add_argument("--subdir", default="wv_2d/pretrained", type=str, required=True)

def main():
    args = parser.parse_args()
    splits = ["train","test","valid"]
    model_name = args.model_name
    model_path = model_path_dict[model_name]
    suffix = args.suffix
    ckpt_infix = "/".join([args.wandb_group, args.wandb_run])
    ckpt = f"checkpoints/{ckpt_infix}/checkpoints/best_model.ckpt"
    dataset = DATASETS[args.dataset]
    subdir = "/".join([args.subdir, args.wandb_group, args.wandb_name])
    pathlib.Path("../embeds/" + subdir).mkdir(parents=True, exist_ok=True)
    print(args)
    for split in splits:
        name = f"{model_name}_{split}_{suffix}"
        embed_path = f"../embeds/{subdir}/{name}.pkl"
        get_embeds(model_path, ckpt, split, dataset["data_dir"], dataset["transform"], embed_path)

if __name__ == "__main__":
    main()
