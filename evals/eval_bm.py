import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette()
from sklearn.metrics.pairwise import euclidean_distances as euc_dist
from eval_ds import *
from embed_evals import syn_evals, get_NI

import argparse
parser = parser = argparse.ArgumentParser()
parser.add_argument("-w", "--wandb_csv", default=None, type=str)
parser.add_argument("-p", "--wandb_project", default=None, type=str)
parser.add_argument("-g", "--wandb_group", default=None, type=str)
args = parser.parse_args()
print(args)

import pathlib
if args.wandb_csv is not None:
    wandb_csv = args.wandb_csv
    wandb_csv_path = pathlib.Path(wandb_csv)
    file_path = wandb_csv.split("/")[-1]
    project, group = file_path.split(".")[:2]
    results_path = str(wandb_csv_path.parents[1]) + '/'
elif args.wandb_project is not None and args.wandb_group is not None:
    project = args.wandb_project
    group = args.wandb_group
    results_path = 'results/'
    wandb_csv = results_path + 'wandb/' + '.'.join([project, group, 'csv'])
else: 
    print("Error: no argument passed!")
    exit()

results_csv = results_path + '.'.join([project, group, 'csv'])
print(results_path, results_csv)

from torchvision.datasets import ImageFolder
train_ds = ImageFolder('../datasets/bm/train')
valid_ds = ImageFolder('../datasets/bm/valid')
test_ds = ImageFolder('../datasets/bm/test')
y_train = np.array([y for _, y in train_ds])
y_valid = np.array([y for _, y in valid_ds])
y_test = np.array([y for _, y in test_ds])
ytvs = y_train, y_valid, y_test

def load_all_embs(path, models, dim, arch=None, seeds=None):
    emb, folds = {}, ['train', 'valid', 'test']
    for model in models:
        if model not in emb: emb[model] = {}
        model_path = '/'.join([path, model, arch]) if arch else '/'.join([path, model])
        if seeds is not None:
            for seed in seeds:
                if model not in emb[model]: emb[model][seed] = {}
                for fold in folds:
                    if fold not in emb[model][seed]: emb[model][seed][fold] = {}
                    emb[model][seed][fold] = pickle.load(
                        open(f'{model_path}_{fold}_emb{dim}_s{seed}.pkl', 'rb'))
        else:
            for fold in folds:
                if fold not in emb[model]: emb[model][fold] = {}
                emb[model][fold] = pickle.load(
                    open(f'{model_path}_{fold}_emb{dim}.pkl', 'rb'))
    return emb

def get_dst_from_embs(embs):
    train, test = embs['train'], embs['test']
    return euc_dist(test, train)

def get_dss_from_embs(embs):
    train, test = embs['test'], embs['test']
    return euc_dist(test, train)

df_wandb = pd.read_csv(wandb_csv)
df = df_wandb.copy()
df['model'] = df['Name'].map(lambda x: x.split('s')[0])
df['perf'] = df['test_clf_acc'] + df['test_triplet_acc']
df = df.loc[df.groupby('model')['perf'].idxmax()]
df_wandb_best = df.copy()
best = {k:int(v) for k, v in df.Name.str.split('s')}
agent = 's'.join(['MTL0', str(best['MTL0'])])
print("best seeds:", best)

embeds_path = '../embeds/bm/prolific/' + '/'.join([project, group])
models = [f'MTL{l}' for l in [0, 0.2, 0.5, 0.8, 1]]
dim = df_wandb.loc[0]['_content.embed_dim']
seeds = sorted(df_wandb['_content.seed'].unique())
print("all seeds:", seeds)
embs = load_all_embs(embeds_path, models, dim, 'MTL_han', seeds)
dsts = {}
for model in models:
    for seed in seeds:
        name = 's'.join([model, str(seed)])
        dsts[name] = get_dst_from_embs(embs[model][seed])
b_embs = {m: embs[m][best[m]] for m in embs}

syns = {k: v for k, v in dsts.items() if 'MTL0s' in k}
# syns['lpips.alex'] = pickle.load(open('../embeds/lpips/bm/lpips.alex.dstt.pkl', 'rb'))
# syns['lpips.vgg'] = pickle.load(open('../embeds/lpips/bm/lpips.vgg.dstt.pkl', 'rb'))
# b_syns = {s: syns[s] for s in [agent, 'lpips.alex', 'lpips.vgg']}
print("syn agents:", syns.keys(), len(syns))

model, seed = 'MTL1', best[model]
z_train, z_test = embs[model][seed]['train'], embs[model][seed]['test']
resn_nis = get_NI(z_train, y_train, z_test, y_test)

id_columns = ['agent', 'name', 'model', 'seed']
ts_columns = ['test_clf_acc', 'test_1nn_acc', 'test_triplet_acc']
ds_columns = ['NINO_ds_acc', 'NIFO_ds_acc', 'rNINO_ds_acc']
er_columns = ['NINO_ds_err', 'NIFO_ds_err', 'rNINO_ds_err']
ni_columns = ['NIs', 'NI_acc']
all_columns = id_columns + ts_columns + ds_columns + er_columns + ni_columns
results = pd.DataFrame(columns=all_columns)
for syn in syns:
    print(f"Evaluating models with agent: {syn}")
    for model in models:
        for seed in seeds:
            z_train, z_test = embs[model][seed]['train'], embs[model][seed]['test']
            evals = syn_evals(z_train, y_train, z_test, y_test, None, None, None, None, dist=syns[syn])
            nn_mat = np.vstack([np.arange(len(y_test)), evals['NIs'], resn_nis]).T
            evals['NI_acc'] = (get_ds_choice(syns[syn], nn_mat) == 0).mean()
            name = 's'.join([model, str(seed)])
            evals.update({k: v for k, v in zip(id_columns, [syn, name, model, seed])})
            test_values = df_wandb[df_wandb['Name'] == name][ts_columns].values[0]
            evals.update({k: v for k, v in zip(ts_columns, test_values)})
            results.loc[len(results)] = [evals[k] for k in all_columns]

results.to_csv(results_csv, index=False)
print("Saved results at:", results_csv)