import argparse, pickle
from pydoc import locate
import numpy as np
import torchvision, torch, os, pathlib
from torchvision import  models
import transforms

######### global variables ###########

DATASETS = {}

DATASETS["bm"] = {"data_dir":"/net/scratch/tianh-shared/bm",
            "transform": "resn"}

DATASETS["bird"] = {"data_dir":"/net/scratch/tianh-shared/bird",
            "transform": "bird"}

DATASETS["chest_xray"] = {"data_dir":"/net/scratch/tianh-shared/PC/3classes",
            "transform": "resn"}

DATASETS["prostatex"] = {"data_dir":"/net/scratch/tianh-shared/bird",
            "transform": "xray"}

DATASETS["wv_2d"] = {"data_dir":"../datasets/weevil_vespula",
            "transform": "wv"}

DATASETS["wv_3d"] = {"data_dir":"../datasets/wv_3d",
            "transform": "wv_3d"}

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

    feature_extractor = models.resnet18(pretrained=True).to("cuda")

    # model = locate(model_path)
    # model = model.load_from_checkpoint(ckpt).to("cuda")
    # model.eval()

    transform = transforms.get_transform("bird", aug=False)
    data_dir = os.path.join(data_dir, split)
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)

    print(f"generating embeds: ----- {split} ------")
    if len(dataset) <= max_dataset:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=4)
        embeds = feature_extractor(list(iter(dataloader))[0][0].cuda())
        embeds = embeds.cpu().detach().numpy()
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=max_dataset, num_workers=4)
        i = 0
        for x, _ in dataloader:
            z = feature_extractor(x.cuda())
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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="wv", type=str, required=True)
parser.add_argument("--subdir", default="wv_2d/pretrained", type=str, required=True)
# parser.add_argument("--wandb_group", default="wv_2d", type=str, required=True)
# parser.add_argument("--wandb_name", default="", type=str, required=False)
# parser.add_argument("--wandb_run", default="2wql6bo5", type=str, required=True)
parser.add_argument("--suffix", default="emb50", type=str, required=True)

def main():
    args = parser.parse_args()
    splits = ["train","valid","test"]
    # model_name = args.model_name
    # model_path = model_path_dict[model_name]
    suffix = args.suffix
    # ckpt_infix = "/".join([args.wandb_group, args.wandb_run])
    # ckpt = f"checkpoints/{ckpt_infix}/checkpoints/best_model.ckpt"
    dataset = DATASETS[args.dataset]
    subdir = args.subdir
    pathlib.Path("../embeds/" + subdir).mkdir(parents=True, exist_ok=True)
    print(args)
    for split in splits:
        name = f"ResNet18_{split}_{suffix}"
        embed_path = f"../embeds/{subdir}/{name}.pkl"
        get_embeds("", "", split, dataset["data_dir"], dataset["transform"], embed_path)

if __name__ == "__main__":
    main()
