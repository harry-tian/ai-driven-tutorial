import embed_evals as evals
import pickle, os
import torchvision
import pandas as pd
import numpy as np
from omegaconf import OmegaConf as oc
from tqdm import tqdm



RESN_embed_dir = "../embeds/wv_3d_alignments/RESN"
for seed in range(1):
    RESN_train = [pickle.load(open(f"{RESN_embed_dir}/RESN_train_seed{seed}.pkl","rb")) for seed in range(1)]
    RESN_test = [pickle.load(open(f"{RESN_embed_dir}/RESN_test_seed{seed}.pkl","rb")) for seed in range(1)]
syn_x_train = pickle.load(open("../datasets/wv_3d/train_features.pkl","rb"))
syn_x_test = pickle.load(open("../datasets/wv_3d/test_features.pkl","rb"))
y_train = np.array([x[1] for x in torchvision.datasets.ImageFolder("../datasets/wv_3d/train")])
y_test = np.array([x[1] for x in torchvision.datasets.ImageFolder("../datasets/wv_3d/test")])

metric = ["NINO_ds_acc","rNINO_ds_acc","NIFO_ds_acc"]

align_results = pd.DataFrame()
align_dir = "../models/configs/wv_3d/align_triplets"
align_yamls = [os.path.join(align_dir,f) for f in os.listdir(align_dir)]
for k in [1,3,5]:
    for yaml in tqdm(align_yamls):
        args = oc.load(yaml)
        weights = args["weights"]
        group = args["wandb_group"]
        
        results = []
        for z_train,z_test in tqdm(zip(RESN_train,RESN_test)):
            data = evals.resn_evals(z_train, y_train, z_test, y_test, syn_x_train, syn_x_test, weights, None, k=k)
            data = {f"{m}_k={k}": [data[m]] for m in metric}
            results.append(pd.DataFrame(data=data))
        results = pd.concat(results)
        results.insert(0, "group", [group]*1)
        align_results = pd.concat([align_results, results])

align_results.to_csv("aligns.csv",index=False)


# filtered_results = pd.DataFrame()
# filtered_dir = "../models/configs/wv_3d/filtered_triplets"
# filtered_yamls = [os.path.join(filtered_dir,f) for f in os.listdir(filtered_dir)]
# for yaml in tqdm(filtered_yamls):
#     args = oc.load(yaml)
#     weights = args["weights"]
#     group = args["wandb_group"]
    
#     results = []
#     for z_train,z_test in tqdm(zip(RESN_train,RESN_test)):
#         data = evals.resn_evals(z_train, y_train, z_test, y_test, syn_x_train, syn_x_test, weights, None, k=1)
#         data = {m: [data[m]] for m in metric}
#         results.append(pd.DataFrame(data=data))
#     results = pd.concat(results)
#     results.insert(0, "group", [group]*1)
#     filtered_results = pd.concat([filtered_results, results])

# filtered_results.to_csv("filtereds.csv",index=False)


# noisy_results = pd.DataFrame()
# noisy_dir = "../models/configs/wv_3d/noisy_triplets"
# noisy_yamls = [os.path.join(noisy_dir,f) for f in os.listdir(noisy_dir)]
# for yaml in tqdm(noisy_yamls):
#     args = oc.load(yaml)
#     weights = args["weights"]
#     group = args["wandb_group"]
    
#     results = []
#     for z_train,z_test in tqdm(zip(RESN_train,RESN_test)):
#         data = evals.resn_evals(z_train, y_train, z_test, y_test, syn_x_train, syn_x_test, weights, None, k=1)
#         data = {m: [data[m]] for m in metric}
#         results.append(pd.DataFrame(data=data))
#     results = pd.concat(results)
#     results.insert(0, "group", [group]*1)
#     noisy_results = pd.concat([noisy_results, results])

# noisy_results.to_csv("noisys.csv",index=False)