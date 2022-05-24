import embed_evals as evals
import pickle, os
import torchvision
import pandas as pd
import numpy as np
from omegaconf import OmegaConf as oc
from tqdm import tqdm



RESN_embed_dir = "wv_3d_blob_2_RESN"
dataset = "wv_3d_blob"
seeds = 3





RESN_embed_dir = os.path.join("../embeds/", RESN_embed_dir)
metric = ["NINO_ds_acc","rNINO_ds_acc","NIFO_ds_acc"]
out = "RESN_evals.csv"
dim = 512


for seed in range(seeds):
    RESN_train = [pickle.load(open(f"{RESN_embed_dir}/RESN_train_d{dim}_seed{seed}.pkl","rb")) for seed in range(seeds)]
    RESN_test = [pickle.load(open(f"{RESN_embed_dir}/RESN_test_d{dim}_seed{seed}.pkl","rb")) for seed in range(seeds)]
    RESN_preds = [pickle.load(open(f"{RESN_embed_dir}/RESN_preds_d{dim}_seed{seed}.pkl","rb")) for seed in range(seeds)]
syn_x_train = pickle.load(open("../datasets/wv_3d_blob_2/train_features.pkl","rb"))
syn_x_test = pickle.load(open("../datasets/wv_3d_blob_2/test_features.pkl","rb"))
y_train = np.array([x[1] for x in torchvision.datasets.ImageFolder("../datasets/wv_3d_blob_2/train")])
y_test = np.array([x[1] for x in torchvision.datasets.ImageFolder("../datasets/wv_3d_blob_2/test")])





align_results = pd.DataFrame()
align_dir = "../models/configs/wv_3d_blob/triplets/aligns"
align_yamls = [os.path.join(align_dir,f) for f in os.listdir(align_dir)]

for yaml in tqdm(align_yamls):
    args = oc.load(yaml)
    weights = args["weights"]
    group = args["wandb_group"]
    # if "unfiltered" in group: continue

    for seed, (z_train,z_test,y_preds) in enumerate(zip(RESN_train,RESN_test,RESN_preds)):
        results = {"group":group, "seed":seed, "weights":weights}
        syn_evals = evals.syn_evals(z_train, y_train, z_test, y_test, y_preds, syn_x_train, syn_x_test, weights, None, k=1)
        results.update({m: syn_evals[m] for m in metric})
                
        results = pd.DataFrame({k:[v] for k,v in results.items()})
        align_results = pd.concat([align_results, results])

align_results.to_csv(out,index=False)

