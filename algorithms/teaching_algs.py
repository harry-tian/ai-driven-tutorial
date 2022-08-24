
import sys, pickle
# from aix360.algorithms.protodash import ProtodashExplainer
from algorithms import pdash, mmd
import numpy as np
from algorithms.selection import tripet_greedy, nn_greedy
from sklearn_extra.cluster import KMedoids
def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def find_idx(X, target):
    for i, x in enumerate(X):
        if np.array_equal(x, target):
            return i
    return -1

def protodash(X, m, args=None):
    _, prototype_idx, _ = pdash.pdash(X, X, m=m, kernelType="Gaussian")
    return prototype_idx

def mmd_greedy(X, m, args=None):
    return mmd.mmd_greedy(X, np.arange(len(X)), m)

def random(X, m, args=None):
    return np.random.choice(np.arange(len(X)), m, replace=False)

def k_medoids(X, m):
    kmedoids = KMedoids(n_clusters=m, random_state=0).fit(X)
    centers = kmedoids.cluster_centers_
    return np.array([find_idx(X,c) for c in centers])

### contrastive selection
def protodash_contrastive(X, Y, m):
    return select_contrastive(X, Y, m, protodash)

def kmedoids_contrastive(X, Y, m):
    return select_contrastive(X, Y, m, k_medoids)
    
def random_contrastive(X, Y, m):
    return select_contrastive(X, Y, m, random)

def select_contrastive(X, Y, m, alg):
    assert(m%2==0)
    classes = np.unique(Y)
    c0 = np.where(Y==classes[0])[0]
    c1 = np.where(Y==classes[1])[0]
    
    return c0[alg(X[c0], m//2)], c1[alg(X[c1], m//2)]


### grouping policies
def group_min(X, S1, S2):
    assert(len(S1)==len(S2))

    # random search
    total = factorial(len(S1)) * len(S1)
    min_dist = np.inf
    for _ in range(total):
        pairs = group_random(X, S1, S2)
        dist = pair_dist_sum(X, pairs)
        if dist < min_dist:
            min_dist_pair = pairs
    return min_dist_pair

def group_max(X, S1, S2):
    assert(len(S1)==len(S2))

    # random search
    total = factorial(len(S1)) * len(S1)
    max_dist = -np.inf
    for _ in range(total):
        pairs = group_random(X, S1, S2)
        dist = pair_dist_sum(X, pairs)
        if dist > max_dist:
            max_dist_pair = pairs
    return max_dist_pair


def group_random(X, S1, S2):
    assert(len(S1)==len(S2))
    S2 = np.random.choice(S2, len(S2), replace=False)
    pairs = np.array([[s1, s2] for s1,s2 in zip(S1, S2)])
    return pairs

def pair_dist_sum(X, pairs):
        return np.array([euc_dist(X[pair[0]], X[pair[1]]) for pair in pairs]).sum()



def tripetgreedy(X, m, args=None):
    try:
        topk = args["topk"]
    except: 
        topk = 10
    try:
        triplets = args["triplets"]
    except: 
        triplets = np.array(pickle.load(open("data/datasets/bm_triplets/3c2_unique=182/train_triplets.pkl", "rb")))
    try:
        y_train = args["y_train"]
    except: 
        y_train = np.array([0]*80+[1]*80)
    return list(tripet_greedy(X, m, triplets, labels=y_train, topk=topk, verbose=False)[0])

def nngreedy(embeds, m, labels):
    return list(nn_greedy(embeds, m, labels)[0])