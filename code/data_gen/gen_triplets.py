from itertools import combinations
import random
SEED = 448
random.seed(SEED)
import numpy as np

def sample_triplets(indexs, num_triplets, visual_weights, embeds):
    combs = list(combinations(indexs, 3))
    triplets = []
    for trip in combs:
        a,p,n = trip
        triplets.append([a,p,n])
        triplets.append([p,a,n])
        triplets.append([n,p,a])
    print("total triplets number: ", len(triplets))
    selected_triplets = random.sample(triplets, k=num_triplets)
    
    return [calc_triplets(t, visual_weights, embeds) for t in selected_triplets]

def sample_mixed_triplets(idx1, idx2, num_triplets, visual_weights, embed1, embed2):
    combs = list(combinations(idx2, 2))
    triplets = []
    for a in idx1:
        for trip in combs:
            p,n = trip
            triplets.append([a,p,n])
    print("total triplets number: ", len(triplets))
    selected_triplets = random.sample(triplets, k=num_triplets)

    return [calc_triplets2(t, visual_weights, embed1, embed2) for t in selected_triplets]

def calc_triplets(triplet, visual_weights, embeds):
    a,p,n  = triplet
    point1 = embeds[a]
    point2 = embeds[p]
    point3 = embeds[n]
    d_ap = weightedL2(point1,point2,visual_weights)
    d_an = weightedL2(point1,point3,visual_weights)
    if d_ap > d_an: return [a,n,p]
    else: return [a,p,n]

def calc_triplets2(triplet, visual_weights, embed1, embed2):
    a,p,n  = triplet
    point1 = embed1[a]
    point2 = embed2[p]
    point3 = embed2[n]
    d_ap = weightedL2(point1,point2,visual_weights)
    d_an = weightedL2(point1,point3,visual_weights)
    if d_ap > d_an: return [a,n,p]
    else: return [a,p,n]


def weightedL2(a, b, visual_weights):
    q = a-b
    return np.sqrt((visual_weights*q*q).sum())
    
# def calc_triplets(triplet, df, selected_features, visual_weights):
#     a,p,n  = triplet
#     # d_ap, d_an = dist(a,p,n)
#     point1 = df[(df['index']==a)][selected_features].to_numpy()
#     point2 = df[(df['index']==p)][selected_features].to_numpy()
#     point3 = df[(df['index']==n)][selected_features].to_numpy()
#     d_ap = weightedL2(point1,point2,visual_weights)
#     d_an = weightedL2(point1,point3,visual_weights)
#     if d_ap > d_an:
#         return [a,n,p]
#     else:
#         return [a,p,n]