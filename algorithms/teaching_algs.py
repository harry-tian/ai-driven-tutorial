
import sys, pickle
sys.path.insert(0,'..')
from aix360.algorithms.protodash import ProtodashExplainer
from algorithms.pdash import proto_g
import numpy as np
from algorithms.selection import tripet_greedy, nn_greedy

def protodash(X, m, args=None):
    kernel = args["kernel"] if args else "Gaussian"
    protodash = ProtodashExplainer()
    _, prototype_idx, _ = protodash.explain(X, X, m=m, kernelType=kernel)
    return prototype_idx

def protogreedy(X, m, args=None):
    return proto_g(X, np.arange(len(X)), m)

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

def random(X, m, args=None):
    return np.random.choice(np.arange(len(X)), m, replace=False)

def nngreedy(embeds, m, labels):
    return list(nn_greedy(embeds, m, labels)[0])