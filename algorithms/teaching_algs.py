
import sys, pickle
sys.path.insert(0,'..')
# from aix360.algorithms.protodash import ProtodashExplainer
from algorithms import pdash
import numpy as np
from algorithms.selection import tripet_greedy, nn_greedy
from sklearn_extra.cluster import KMedoids

def find_idx(X, target):
    for i, x in enumerate(X):
        if np.array_equal(x, target):
            return i
    return -1

def protodash(X, m, args=None):
    _, prototype_idx, _ = pdash.pdash(X, X, m=m, kernelType="Gaussian")
    return prototype_idx

def protogreedy(X, m, args=None):
    return pdash.proto_g(X, np.arange(len(X)), m)

def random(X, m, args=None):
    return np.random.choice(np.arange(len(X)), m, replace=False)

def k_medoids(X, m):
    kmedoids = KMedoids(n_clusters=m, random_state=0).fit(X)
    centers = kmedoids.cluster_centers_
    return np.array([find_idx(X,c) for c in centers])



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