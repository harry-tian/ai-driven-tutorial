from re import sub
from telnetlib import SB
import utils
import argparse, pickle
from pydoc import locate
import numpy as np

def get_embeds(model_path, args, ckpt, split, train_idx=None, valid_idx=None, embed_path=None):
    model = locate(model_path)
    model = model.load_from_checkpoint(ckpt, **vars(args)).to("cuda")
    model.eval()
    train_dataset, valid_dataset = utils.get_bm_datasets(train_idx=train_idx, valid_idx=valid_idx)

    if split == "train":
        dataset = train_dataset.cuda()
    elif split == "val" or split == "valid":
        dataset = valid_dataset.cuda()
    else:
        print("???")
        quit()
    
    embeds = model.embed(dataset)
    # embeds = model.feature_extractor(dataset)
    # for layer in model.fc:
    #     embeds = layer(embeds)
    
    embeds = embeds.cpu().detach().numpy()
    print(f"embeds.shape:{embeds.shape}")

    if not embed_path:
        embed_path = f"{model_path}_{split}.pkl"
    pickle.dump(embeds, open(embed_path, "wb"))
    print(f"dumped to {embed_path}")

    return embeds

model_path = "resn_args.RESN"

args = argparse.Namespace(embed_dim=10)
ckpt = 'baselines/3r1dhwq2'



subdir = "bm"
name = "RESN_split_emb10"
split = "valid"





name = name.replace("split",split)
embed_path = f"../embeds/{subdir}/{name}.pkl"
ckpt = f"results/{ckpt}/checkpoints/best_model.ckpt" 
get_embeds(model_path, args, ckpt, split, embed_path=embed_path)