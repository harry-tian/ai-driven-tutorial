from itertools import combinations
import random,sys
import numpy as np

def sample_triplets(train_idx, num_triplets): 
    triplets = []
    for _ in range(num_triplets):
        triplet = random.sample(list(train_idx),3) 
        if triplet not in triplets: triplets.append(triplet)
    return triplets
    
def sample_mixed_triplets(test_idx, train_idx, num_triplets): 
    triplets = []
    for _ in range(num_triplets):
        triplet = random.sample(list(test_idx),1) + random.sample(list(train_idx),2)  
        if triplet not in triplets: triplets.append(triplet)
    return triplets

# sys.path.insert(0, '..')
# import evals.embed_evals as evals

def weightedPdist(a, b, w, power=2):
    """ Han's faster version"""
    a = a.reshape(-1,len(w))
    b = b.reshape(-1,len(w))

    diff = a[:,np.newaxis].repeat(len(b),1) - b
    return ((np.abs(diff)**2)*w).sum(-1)**(1/2)

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

def sample_triplets_weighted(indexs, num_triplets, visual_weights, embeds, powers):
    # combs = list(combinations(indexs, 3))
    # triplets = []
    # for trip in combs:
    #     a,p,n = trip
    #     triplets.append([a,p,n])
    #     triplets.append([p,a,n])
    #     triplets.append([n,p,a])
    # selected_triplets = random.sample(triplets, k=num_triplets)
    selected_triplets = sample_triplets(indexs, num_triplets)
    
    return [calc_triplets(t, visual_weights, embeds, powers) for t in selected_triplets]

def sample_mixed_triplets_weighted(test_idx, train_idx, num_triplets, visual_weights, test_embeds, train_embeds, powers):
    # combs = list(combinations(train_idx, 2))
    # triplets = []
    # for a in idx1:
    #     for trip in combs:
    #         p,n = trip
    #         triplets.append([a,p,n])
    # selected_triplets = random.sample(triplets, k=num_triplets)

    selected_triplets = sample_mixed_triplets(test_idx, train_idx, num_triplets)
    return [calc_triplets2(t, visual_weights, test_embeds, train_embeds, powers) for t in selected_triplets]

def calc_triplets(triplet, visual_weights, embeds, powers):
    a,p,n  = triplet
    point1 = embeds[a]
    point2 = embeds[p]
    point3 = embeds[n]
    d_ap = weightedPdist(point1,point2,visual_weights, powers)
    d_an = weightedPdist(point1,point3,visual_weights, powers)
    if d_ap > d_an: return [a,n,p]
    else: return [a,p,n]

def calc_triplets2(triplet, visual_weights, test_embeds, train_embeds, powers):
    a,p,n  = triplet
    point1 = test_embeds[a]
    point2 = train_embeds[p]
    point3 = train_embeds[n]
    d_ap = weightedPdist(point1,point2,visual_weights, powers)
    d_an = weightedPdist(point1,point3,visual_weights, powers)
    if d_ap > d_an: return [a,n,p]
    else: return [a,p,n]

def get_alignment_filtered_triplets(train_features, valid_features, test_features, y_train, y_valid, y_test, weight, total):
    len_train = len(y_train)
    len_valid = len(y_valid)
    len_test = len(y_test)
    train_triplets = sample_triplets_weighted(np.arange(len_train), int(total*0.8), weight, train_features, [2,2])
    valid_triplets = sample_mixed_triplets_weighted(np.arange(len_valid),np.arange(len_train), int(total*0.2), weight, valid_features, train_features, [2,2])
    test_triplets = sample_mixed_triplets_weighted(np.arange(len_test),np.arange(len_train), total, weight, test_features, train_features, [2,2])
    train_triplets_filtered = filter_train_triplets(train_triplets, y_train)
    valid_triplets_filtered = filter_mixed_triplets(valid_triplets, y_train, y_valid)
    test_triplets_filtered = filter_mixed_triplets(test_triplets, y_train, y_test)

    return train_triplets, valid_triplets, test_triplets, train_triplets_filtered, valid_triplets_filtered, test_triplets_filtered

