import os, pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances 
import torch
import torchvision
from tqdm import tqdm
# pdist = torch.nn.PairwiseDistance()
from itertools import combinations
from scipy import stats
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC, LinearSVC
from torchvision import transforms
import matplotlib.pyplot as plt

np.random.seed(42)
# def euc_dist(x, y): return euclidean_distances([x],[y])[0][0]
def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def bm_eval(train_embeds, valid_embeds):
    assert(train_embeds.shape[0] == 160)
    assert(valid_embeds.shape[0] == 40)

    train_triplets = "data/bm_lpips_triplets/train_triplets.pkl"
    valid_triplets = "data/bm_lpips_triplets/valid_triplets.pkl"
    val2train_triplets = "data/bm_lpips_triplets/val2train_triplets.pkl"
    train_triplets = pickle.load(open(train_triplets, "rb"))
    valid_triplets = pickle.load(open(valid_triplets, "rb"))
    val2train_triplets = pickle.load(open(val2train_triplets,"rb"))
    train_y = [0]*80 + [1]*80
    valid_y = [0]*20 + [1]*20

    train_triplet_acc = get_triplet_acc(train_embeds, train_triplets)
    valid_triplet_acc = get_triplet_acc(valid_embeds, valid_triplets)
    val2train_triplet_acc = get_val2train_triplet_acc(train_embeds, valid_embeds, val2train_triplets)
    knn_acc = get_knn_score(train_embeds, train_y, valid_embeds, valid_y, metric="")
    knn_auc = get_knn_score(train_embeds, train_y, valid_embeds, valid_y, metric="auc")

    print(f"train_triplet_acc: {train_triplet_acc}")
    print(f"valid_triplet_acc: {valid_triplet_acc}")
    print(f"val2train_triplet_acc: {val2train_triplet_acc}")
    print(f"knn_acc: {knn_acc}")
    print(f"knn_auc: {knn_auc}")

    return train_triplet_acc, valid_triplet_acc, val2train_triplet_acc, knn_acc, knn_auc

def get_triplet_acc(embeds, triplets, dist_f=euc_dist):
    """Return triplet accuracy given ground-truth triplets."""
    align = []
    for triplet in tqdm(triplets):
        a, p, n = triplet
        ap = dist_f(embeds[a], embeds[p]) 
        an = dist_f(embeds[a], embeds[n])
        align.append(ap < an)
    acc = np.mean(align)
    return acc

def get_triplet_acc_distM(embeds, dist_matrix, dist_f=euc_dist):
    """Return triplet accuracy given ground-truth distance matrix."""
    triplets = []
    combs = np.array(list(combinations(np.arange(0, len(dist_matrix)), r=3)))
    for c in combs:
        a, p, n = c
        if dist_matrix[a, p] < dist_matrix[a, n]:
            triplets.append([a, p, n])
        else:
            triplets.append([a, n, p])
    return triplet_acc(embeds, triplets, dist_f)

def get_val2train_triplet_acc(train_embeds, val_embeds, val2train_triplets, dist_f=euc_dist):
    align = []
    for triplet in tqdm(val2train_triplets):
        val_a, train_p, train_n = triplet[0], triplet[1], triplet[2]
        ap = dist_f(val_embeds[val_a], train_embeds[train_p]) 
        an = dist_f(val_embeds[val_a], train_embeds[train_n])
        align.append(ap < an)
    acc = np.mean(align)
    return acc

def get_val2train_triplets(val2train_dist_matrix):
    train_len = val2train_dist_matrix.shape[1]
    train_combs = torch.combinations(torch.arange(train_len),2).numpy()
    triplets = []
    for val, dist_to_train in enumerate(val2train_dist_matrix):
        for comb in train_combs:
            p, n = comb[0], comb[1]
            if dist_to_train[p] > dist_to_train[n]:
                triplet = [val, n, p]
            else:
                triplet = [val, p, n]
            triplets.append(triplet)
    return np.array(triplets)

def get_knn_score(x_train, y_train, x_valid, y_valid, k=1, metric="auc", weights="uniform"):
    knc = KNeighborsClassifier(n_neighbors=k, weights=weights)
    knc.fit(x_train, y_train)
    if metric == 'auc':
        probs = knc.predict_proba(x_valid)
        probs = probs[:, 1] if probs.shape[1] > 1 else probs
        score = roc_auc_score(y_valid, probs)
    else:
        score = knc.score(x_valid, y_valid)
    return score


# def get_knn_score(k, data, index, metric="auc", weights="uniform"):
#     x_train, y_train, x_valid, y_valid = data
#     knc = KNeighborsClassifier(n_neighbors=k, weights=weights)
#     knc.fit(x_train[index], y_train[index])
#     if metric == 'auc':
#         probs = knc.predict_proba(x_valid)
#         probs = probs[:, 1] if probs.shape[1] > 1 else probs
#         score = roc_auc_score(y_valid, probs)
#     else:
#         score = knc.score(x_valid, y_valid)
#     return score

