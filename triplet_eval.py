import os, pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances 
import torch
from tqdm import tqdm

np.random.seed(42)
def euc_dist(x, y): return euclidean_distances([x],[y])[0][0]

model = 'resnt'
model = 'triplet_bs=8'
model = 'triplet_subset'

name = 'emb10.l10'
# train_path = '{}/{}_train_{}.pkl'.format("embeds", model, name)
valid_path = '{}/{}_valid_{}.pkl'.format("embeds", model, name)
# f_train, _, y_train, X_train = pickle.load(open(train_path, "rb"))
# f_valid, _, y_valid, X_valid = pickle.load(open(valid_path, "rb"))
f_valid, _, X_valid = pickle.load(open(valid_path, "rb"))

# train_align = []
# train_dist = pickle.load(open("embeds/lpips.bm.train.pkl", "rb"))
# combs = torch.combinations(torch.arange(0, len(train_dist)-1).int(), r=3)
# for i in tqdm(range(100000)):
#     a, p, n = np.random.choice(len(X_train), 3, replace=False)
# # for c in tqdm(combs):
# #     a, p, n = c
#     ap = train_dist[a, p] < train_dist[a, n]
#     rd = euc_dist(X_train[a], X_train[p]) < euc_dist(X_train[a], X_train[n])
#     train_align.append(ap == rd)
# print(np.mean(train_align))

valid_align = []
valid_dist = pickle.load(open("embeds/lpips.bm.valid.pkl", "rb"))
combs = torch.combinations(torch.arange(0, len(X_valid)-1).int(), r=3)
for c in tqdm(combs):
    a, p, n = c
    ap = valid_dist[a, p] < valid_dist[a, n]
    rd = euc_dist(X_valid[a], X_valid[p]) < euc_dist(X_valid[a], X_valid[n])
    valid_align.append(ap == rd)
print(np.mean(valid_align))