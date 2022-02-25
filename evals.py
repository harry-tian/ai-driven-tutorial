import os, pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances 
import torch
import torchvision
from tqdm import tqdm
pdist = torch.nn.PairwiseDistance()
from itertools import combinations

np.random.seed(42)
# def euc_dist(x, y): return euclidean_distances([x],[y])[0][0]
def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def triplet_acc(embeds, triplets, dist_f=euc_dist):
    """Return triplet accuracy given ground-truth triplets."""
    align = []
    for triplet in tqdm(triplets):
        a, p, n = triplet
        ap = dist_f(embeds[a], embeds[p]) 
        an = dist_f(embeds[a], embeds[n])
        align.append(ap < an)
    acc = np.mean(align)
    return acc


def triplet_acc_distM(embeds, dist_matrix, dist_f=euc_dist):
    """Return triplet accuracy given ground-truth distance matrix."""
    align = []
    triplets = np.array(list(combinations(np.arange(0, len(dist_matrix)), r=3)))
    for triplet in tqdm(triplets):
        a, p, n = triplet
        ap = dist_matrix[a, p] < dist_matrix[a, n]
        rd = euc_dist(embeds[a], embeds[p]) < euc_dist(embeds[a], embeds[n])
        align.append(ap == rd)
    acc = np.mean(align)
    return acc