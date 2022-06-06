import embed_evals as evals
import pickle, os, sys, pathlib
import torchvision
import pandas as pd
import numpy as np
from omegaconf import OmegaConf as oc
from tqdm import tqdm

sys.path.insert(0, '/net/scratch/tianh/explain_teach')
from utils import utils


data = "wv_linear_sm"


out = f"../results/syn_results/{data}"
align_dir = f"../models/configs/{data}/triplets/aligns"

seeds = 3
RESN_embed_dir = data + "_RESN"
syn_x_train = pickle.load(open(f"../data/datasets/{data}/train_features.pkl","rb"))
syn_x_test = pickle.load(open(f"../data/datasets/{data}/test_features.pkl","rb"))
y_train = np.array([x[1] for x in torchvision.datasets.ImageFolder(f"../data/datasets/{data}/train")])
y_test = np.array([x[1] for x in torchvision.datasets.ImageFolder(f"../data/datasets/{data}/test")])




RESN_embed_dir = os.path.join("../data/embeds/", RESN_embed_dir)
metric = ["NINO_ds_acc","rNINO_ds_acc","NIFO_ds_acc"]
align_yamls = [os.path.join(align_dir,f) for f in os.listdir(align_dir)]

for dim in [50,512]:
    align_results = pd.DataFrame()
    for yaml in tqdm(align_yamls):
        args = oc.load(yaml)
        weights = args["weights"]
        group = args["wandb_group"]

        for seed in range(seeds):
            RESN_train = pickle.load(open(f"{RESN_embed_dir}/RESN_train_d{dim}_seed{seed}.pkl","rb")) 
            RESN_test = pickle.load(open(f"{RESN_embed_dir}/RESN_test_d{dim}_seed{seed}.pkl","rb")) 
            RESN_preds = pickle.load(open(f"{RESN_embed_dir}/RESN_preds_d{dim}_seed{seed}.pkl","rb")) 
            results = {"wandb_group":group, "seed":seed, "weights":weights, "name":"RESN"}
            syn_evals = evals.syn_evals(RESN_train, y_train, RESN_test, y_test, RESN_preds, syn_x_train, syn_x_test, weights, None, k=1)
            results.update({m: syn_evals[m] for m in metric})
                    
            results = pd.DataFrame({k:[v] for k,v in results.items()})
            align_results = pd.concat([align_results, results])

    align_results["align"] = [utils.find_float(x)[0] for x in align_results["wandb_group"]]

    eval_cols = ['NINO_ds_acc',  'NIFO_ds_acc', 'NI_h2h_d50', 'NO_h2h_d50', 'NI_h2h_d512', 'NO_h2h_d512', 'test_triplet_acc','test_clf_acc', 'test_1nn_acc', 'rNINO_ds_acc',]
    config_cols = ["seed", "weights", "filtered", "embed_dim", "lamda", "wandb_group", "wandb_project"]

    for col in config_cols + eval_cols:
        if col not in align_results.columns: align_results[col] = [None]*len(align_results)

    dir = f"out/d={dim}"
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    align_results.to_csv(f"{dir}/RESN.csv",index=False)

