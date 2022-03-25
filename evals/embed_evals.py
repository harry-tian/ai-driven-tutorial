import os, pickle
# import pandas as pd
import numpy as np
import torch
# from tqdm import tqdm
# pdist = torch.nn.PairwiseDistance()
from itertools import combinations
# from scipy import stats
# from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
# from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import euclidean_distances
np.random.seed(42)
def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def bm_eval_human(x_train, x_valid):
    y_train = np.array([0]*80+[1]*80)
    y_valid = np.array([0]*20+[1]*20)
    assert(x_train.shape[0] == 160)
    assert(x_valid.shape[0] == 40)

    train_triplets = "data/bm_triplets/3c2_unique=182/train_triplets.pkl"
    valid_triplets = "data/bm_triplets/3c2_unique=182/valid_triplets.pkl"
    # val2train_triplets = "data/bm_lpips_triplets/val2train_triplets.pkl"
    train_triplets = pickle.load(open(train_triplets, "rb"))
    valid_triplets = pickle.load(open(valid_triplets, "rb"))
    # val2train_triplets = pickle.load(open(val2train_triplets,"rb"))

    train_triplet_acc = get_triplet_acc(x_train, train_triplets)
    valid_triplet_acc = get_triplet_acc(x_valid, valid_triplets)
    # val2train_triplet_acc = get_val2train_triplet_acc(x_train, x_valid, val2train_triplets)
    knn_acc = get_knn_score(x_train, y_train, x_valid, y_valid, metric="")
    knn_auc = get_knn_score(x_train, y_train, x_valid, y_valid, metric="auc")
    human_align = human_1NN_align(x_train, x_valid)

    results = {
        "train_triplet_acc":train_triplet_acc,
        "valid_triplet_acc":valid_triplet_acc,
        "knn_acc":knn_acc,
        "knn_auc":knn_auc,
        # "val2train_triplet_acc":val2train_triplet_acc,
        "human_1NN_align":human_align,
    }
    print(results)

    return results

def human_1NN_align(x_train, x_valid):
    embeds = np.concatenate((np.array(x_train),np.array(x_valid)))
    train_embs = pickle.load(open("embeds/bm/human/TN_train_emb10.pkl","rb"))
    valid_embs = pickle.load(open("embeds/bm/human/TN_valid_emb10.pkl","rb"))
    human_embs = np.concatenate((np.array(train_embs),np.array(valid_embs)))
    assert(len(embeds) == len(human_embs))

    correct = 0
    total = 0
    for i in range(len(human_embs)):
        total += 1
        if get_1nn(human_embs, i) == get_1nn(embeds, i):
            correct += 1
    
    return correct/total

def get_triplet_acc(embeds, triplets, dist_f=euc_dist):
    """Return triplet accuracy given ground-truth triplets."""
    align = []
    for triplet in triplets:
        a, p, n = triplet
        ap = dist_f(embeds[a], embeds[p]) 
        an = dist_f(embeds[a], embeds[n])
        align.append(ap < an)
    acc = np.mean(align)
    return acc

def get_knn_score(x_train, y_train, x_valid, y_valid, 
                k=1, metric="acc", weights="uniform"):
    knc = KNeighborsClassifier(n_neighbors=k, weights=weights)
    knc.fit(x_train, y_train)
    if metric == 'auc':
        probs = knc.predict_proba(x_valid)
        probs = probs[:, 1] if probs.shape[1] > 1 else probs
        score = roc_auc_score(y_valid, probs)
    else:
        score = knc.score(x_valid, y_valid)
    return score

def get_1nn(data, index):
    dist = euclidean_distances(data)
    return np.argsort(dist[index])[1]

####### MISC #############################

def triplet_distM_align(triplets, dist_matrix):
    correct = 0
    total = 0
    for triplet in triplets:
        total += 1
        a,p,n = triplet[0], triplet[1], triplet[2]
        if dist_matrix[a, p] < dist_matrix[a, n]:
            correct += 1
    return correct/total

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
    return get_triplet_acc(embeds, triplets, dist_f)
    
def get_val2train_triplet_acc(train_embeds, val_embeds, val2train_triplets, dist_f=euc_dist):
    align = []
    for triplet in val2train_triplets:
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

def bm_eval_lpips(x_train, y_train, x_valid, y_valid):
    assert(x_train.shape[0] == 160)
    assert(x_valid.shape[0] == 40)

    train_triplets = "data/bm_lpips_triplets/train_triplets.pkl"
    valid_triplets = "data/bm_lpips_triplets/valid_triplets.pkl"
    val2train_triplets = "data/bm_lpips_triplets/val2train_triplets.pkl"
    train_triplets = pickle.load(open(train_triplets, "rb"))
    valid_triplets = pickle.load(open(valid_triplets, "rb"))
    val2train_triplets = pickle.load(open(val2train_triplets,"rb"))

    train_triplet_acc = get_triplet_acc(x_train, train_triplets)
    valid_triplet_acc = get_triplet_acc(x_valid, valid_triplets)
    val2train_triplet_acc = get_val2train_triplet_acc(x_train, x_valid, val2train_triplets)
    knn_acc = get_knn_score(x_train, y_train, x_valid, y_valid, metric="")
    knn_auc = get_knn_score(x_train, y_train, x_valid, y_valid, metric="auc")
    print("lpips triplet evals:")

    print(f"train_triplet_acc: {train_triplet_acc}")
    print(f"valid_triplet_acc: {valid_triplet_acc}")
    print(f"val2train_triplet_acc: {val2train_triplet_acc}")
    print(f"knn_acc: {knn_acc}")
    print(f"knn_auc: {knn_auc}")

    return train_triplet_acc, valid_triplet_acc, val2train_triplet_acc, knn_acc, knn_auc