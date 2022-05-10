from itertools import combinations
import random,sys
import numpy as np
sys.path.insert(0, '..')
import evals.embed_evals as evals

def get_noisy_triplets(triplets, p):
    triplets = np.array(triplets)
    noisy_triplets = []
    for t in triplets:
        if np.random.binomial(1, p, 1)[0] > 0:
            noisy_triplets.append([t[0], t[2], t[1]])
        else: noisy_triplets.append(t)
    return noisy_triplets

def filter_train_triplets(triplets, y_train):
    filtered = []
    for t in triplets:
        a,p,n = t
        if not (y_train[a]==y_train[n] and y_train[a]!=y_train[p]): filtered.append(t)
    return np.array(filtered)

def filter_mixed_triplets(triplets, y_train, y_test):
    filtered = []
    for t in triplets:
        a,p,n = t
        if not (y_test[a]==y_train[n] and y_test[a]!=y_train[p]): filtered.append(t)
    return np.array(filtered)

def sample_triplets(indexs, num_triplets, visual_weights, embeds, powers):
    combs = list(combinations(indexs, 3))
    triplets = []
    for trip in combs:
        a,p,n = trip
        triplets.append([a,p,n])
        triplets.append([p,a,n])
        triplets.append([n,p,a])
    selected_triplets = random.sample(triplets, k=num_triplets)
    
    return [calc_triplets(t, visual_weights, embeds, powers) for t in selected_triplets]

def sample_mixed_triplets(idx1, train_idx, num_triplets, visual_weights, test_embeds, train_embeds, powers):
    combs = list(combinations(train_idx, 2))
    triplets = []
    for a in idx1:
        for trip in combs:
            p,n = trip
            triplets.append([a,p,n])
    selected_triplets = random.sample(triplets, k=num_triplets)

    return [calc_triplets2(t, visual_weights, test_embeds, train_embeds, powers) for t in selected_triplets]

def calc_triplets(triplet, visual_weights, embeds, powers):
    a,p,n  = triplet
    point1 = embeds[a]
    point2 = embeds[p]
    point3 = embeds[n]
    d_ap = evals.weightedPdist(point1,point2,visual_weights, powers)
    d_an = evals.weightedPdist(point1,point3,visual_weights, powers)
    if d_ap > d_an: return [a,n,p]
    else: return [a,p,n]

def calc_triplets2(triplet, visual_weights, test_embeds, train_embeds, powers):
    a,p,n  = triplet
    point1 = test_embeds[a]
    point2 = train_embeds[p]
    point3 = train_embeds[n]
    d_ap = evals.weightedPdist(point1,point2,visual_weights, powers)
    d_an = evals.weightedPdist(point1,point3,visual_weights, powers)
    if d_ap > d_an: return [a,n,p]
    else: return [a,p,n]

    