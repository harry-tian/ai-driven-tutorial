import os, pickle
import numpy as np
import torch
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
# import eval_ds as utils
np.random.seed(42)
def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def resn_evals(z_train, y_train, z_test, y_test, syn_x_train, syn_x_test, weights, powers, k=1, dist=None, RESN_embeds=None):
    evals = {}
    
    NINO = get_NINO(z_train, y_train, z_test, y_test, k)
    ds_dist = decision_support_with_dist if k==1 else knn_decision_support_with_dist
    if dist is not None:
        NINO_ds_acc, NINO_ds_err = ds_dist(dist, NINO, y_train, y_test)
    else:
        NINO_ds_acc, NINO_ds_err = decision_support(syn_x_train, y_train, syn_x_test, y_test, NINO, weights, powers)
    evals['NINO_ds_acc'], evals['NINO_ds_err'] = NINO_ds_acc, NINO_ds_err
    
    rNINO = get_rNINO(z_train, y_train, z_test, y_test, k)
    if dist is not None:
        rNINO_ds_acc, rNINO_ds_err = ds_dist(dist, rNINO, y_train, y_test)
    else:
        rNINO_ds_acc, rNINO_ds_err = decision_support(syn_x_train, y_train, syn_x_test, y_test, rNINO, weights, powers)
    evals['rNINO_ds_acc'], evals['rNINO_ds_err'] = rNINO_ds_acc, rNINO_ds_err

    NIFO = get_NIFO(z_train, y_train, z_test, y_test, k)
    if dist is not None:
        NIFO_ds_acc, NIFO_ds_err = ds_dist(dist, NIFO, y_train, y_test)
    else:
        NIFO_ds_acc, NIFO_ds_err = decision_support(syn_x_train, y_train, syn_x_test, y_test, NIFO, weights, powers)
    evals['NIFO_ds_acc'], evals['NIFO_ds_err'] = NIFO_ds_acc, NIFO_ds_err

    return evals
    
def syn_evals(z_train, y_train, z_test, y_test, syn_x_train, syn_x_test, weights, powers, k=1, dist=None, RESN_embeds=None):
    evals = {}
    
    NINO = get_NINO(z_train, y_train, z_test, y_test, k)
    ds_dist = decision_support_with_dist if True else knn_decision_support_with_dist
    if dist is not None:
        NINO_ds_acc, NINO_ds_err = ds_dist(dist, NINO, y_train, y_test)
    else:
        NINO_ds_acc, NINO_ds_err = decision_support(syn_x_train, y_train, syn_x_test, y_test, NINO, weights, powers)
    evals['NINO_ds_acc'], evals['NINO_ds_err'] = NINO_ds_acc, NINO_ds_err
    
    rNINO = get_rNINO(z_train, y_train, z_test, y_test, k)
    if dist is not None:
        rNINO_ds_acc, rNINO_ds_err = ds_dist(dist, rNINO, y_train, y_test)
    else:
        rNINO_ds_acc, rNINO_ds_err = decision_support(syn_x_train, y_train, syn_x_test, y_test, rNINO, weights, powers)
    evals['rNINO_ds_acc'], evals['rNINO_ds_err'] = rNINO_ds_acc, rNINO_ds_err

    NIFO = get_NIFO(z_train, y_train, z_test, y_test, k)
    if dist is not None:
        NIFO_ds_acc, NIFO_ds_err = ds_dist(dist, NIFO, y_train, y_test)
    else:
        NIFO_ds_acc, NIFO_ds_err = decision_support(syn_x_train, y_train, syn_x_test, y_test, NIFO, weights, powers)
    evals['NIFO_ds_acc'], evals['NIFO_ds_err'] = NIFO_ds_acc, NIFO_ds_err

    NIs = get_NI(z_train, y_train, z_test, y_test, k)
    evals['NIs'] = NIs

    return evals

def nn_comparison(x_train, x_test, NI1, NI2, weights, powers=[2,2]):
    ni1_score, ni2_score, ties = 0, 0, 0
    for test_idx, (ni1, ni2) in enumerate(zip(NI1,NI2)):
        dist_ni1 = [weightedPdist(x_test[test_idx], x_train[ni], weights, powers) for ni in ni1]
        dist_ni2 = [weightedPdist(x_test[test_idx], x_train[ni], weights, powers) for ni in ni2]
        if  min(dist_ni1) < min(dist_ni2): ni1_score += 1
        elif min(dist_ni1) > min(dist_ni2): ni2_score += 1
        else: ties += 1

    return ni1_score, ni2_score, ties

def get_NINO(x_train, y_train, x_test, y_test, k=1):
    ''' Neareast In-class & Nearest Out-of-class '''
    NIs = get_NI(x_train, y_train, x_test, y_test, k)

    classes = np.unique(y_train)
    idx_by_class = {c: np.where(y_train==c)[0] for c in classes}

    NINOs = []
    for x, y, NI in zip(x_test, y_test, NIs):
        NINO = [NI]
        for c in classes:
            if y == c: continue
            class_idx = idx_by_class[c]
            dists = np.array([euc_dist(x, x_) for x_ in x_train[class_idx]])
            # nn = np.argmin(dists)
            # class_nn = idx_by_class[c][nn]
            nns = np.argsort(dists)[:k]
            class_nn = np.array([class_idx[nn] for nn in nns])
            NINO.append(class_nn)
        NINOs.append(NINO)

    return np.array(NINOs)

def get_rNINO(x_train, y_train, x_test, y_test, k=1):
    ''' NINO with NO no nearer than NI '''
    NIs = get_NI(x_train, y_train, x_test, y_test, k)

    classes = np.unique(y_train)
    idx_by_class = {c: np.where(y_train==c)[0] for c in classes}

    NINOs = []
    for x, y, NI in zip(x_test, y_test, NIs):
        NINO = [NI]
        for c in classes:
            if y == c: continue
            class_idx = idx_by_class[c]
            dists = np.array([euc_dist(x, x_) for x_ in x_train[class_idx]])
            dists = [d if d > euc_dist(x, x_train[NI[0]]) else np.inf for d in dists]
            if min(dists) == np.inf: ## signal of error in rNINO
                NINO.append(NI)
            else:
                # nn = np.argmin(dists)
                # class_nn = idx_by_class[c][nn]
                nns = np.argsort(dists)[:k]
                class_nn = np.array([class_idx[nn] for nn in nns])
                NINO.append(class_nn)
        NINOs.append(NINO)

    return np.array(NINOs)

def get_NI(x_train, y_train, x_test, y_test, k=1):
    ''' Neareast In-class '''
    classes = np.unique(y_train)
    idx_by_class = {c: np.where(y_train==c)[0] for c in classes}

    NIs = []
    for x, y in zip(x_test, y_test):
        in_class = x_train[idx_by_class[y]]
        dists = np.array([euc_dist(x, x_) for x_ in in_class])
        nns = np.argsort(dists)[:k]
        NI = [idx_by_class[y][nn] for nn in nns]
        # NN = np.argmin(dists)
        # NI = idx_by_class[y][NN]
        NIs.append(NI)

    return np.array(NIs)
        
def get_NIFO(x_train, y_train, x_test, y_test, k=1):
    ''' Neareast In-class & Furthest Out-of-class '''
    NIs = get_NI(x_train, y_train, x_test, y_test, k)

    classes = np.unique(y_train)
    idx_by_class = {c: np.where(y_train==c)[0] for c in classes}

    NINOs = []
    for x, y, NI in zip(x_test, y_test, NIs):
        NINO = [NI]
        for c in classes:
            if y == c: continue
            class_idx = idx_by_class[c]
            dists = np.array([euc_dist(x, x_) for x_ in x_train[class_idx]])
            # nn = np.argmax(dists)
            # class_nn = idx_by_class[c][nn]
            nns = np.argsort(dists)[-k:]
            class_nn = np.array([class_idx[nn] for nn in nns])
            NINO.append(class_nn)
        NINOs.append(NINO)

    return np.array(NINOs)

def decision_support(x_train, y_train, x_test, y_test, examples, weights, powers):
    correct = 0
    err = []
    for test_idx, examples_idx in enumerate(examples):
        examples_idx = examples_idx.flatten()
        # print(examples_idx)
        if len(np.unique(examples_idx)) < len(examples_idx): ## signal of error in rNINO
            err.append([test_idx, examples_idx[0], examples_idx[1]])
            continue
        ref = x_test[test_idx]
        if weights is None and powers is None:
            dist_fn = lambda x, y: euclidean_distances(x.reshape(1, -1), y.reshape(1, -1))
        else:
            dist_fn = lambda x, y: weightedPdist(x, y, weights, powers)
        dists = [dist_fn(ref, x_train[idx]) for idx in examples_idx]
        # dists = [weightedPdist(ref, x_train[idx], weights, powers) for idx in examples_idx]
        y_pred = y_train[examples_idx[np.argmin(dists)]]
        if y_pred == y_test[test_idx]: correct += 1
        else: err.append([test_idx, examples_idx[0], examples_idx[1]])
    return correct/len(y_test), err

def decision_support_with_dist(dist, examples, y_train, y_test):
    correct = 0
    err = []
    for test_idx, examples_idx in enumerate(examples):       
        examples_idx = examples_idx.flatten()
        if len(np.unique(examples_idx)) < len(examples_idx): ## signal of error in rNINO
            err.append([test_idx, examples_idx[0], examples_idx[1]])
            continue
        y_pred = y_train[examples_idx[np.argmin(dist[test_idx][examples_idx])]]
        if y_pred == y_test[test_idx]: correct += 1
        else: err.append([test_idx, examples_idx[0], examples_idx[1]])
    return correct/len(y_test), err

def knn_decision_support_with_dist(dist, examples, y_train, y_test):
    correct = 0
    err = []
    for test_idx, examples_idx in enumerate(examples):       
        examples_idx = examples_idx.flatten()
        if len(np.unique(examples_idx)) < len(examples_idx): ## signal of error in rNINO
            err.append([test_idx, examples_idx[0], examples_idx[1]])
            continue
        dists = dist[test_idx][examples_idx]
        y_pred = y_train[examples_idx]
        w = 1 / dists
        y_pred = np.bincount(y_pred, w).argmax()
        if y_pred == y_test[test_idx]: correct += 1
        else: err.append([test_idx, examples_idx[0], examples_idx[1]])
    return correct/len(y_test), err

def get_triplet_acc(embeds, triplets, dist_f=euc_dist):
    ''' Return triplet accuracy given ground-truth triplets.''' 
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
    ''' Return K=1NN accuracy. ''' 
    knc = KNeighborsClassifier(n_neighbors=k, weights=weights)
    knc.fit(x_train, y_train)
    if metric == 'auc':
        probs = knc.predict_proba(x_valid)
        probs = probs[:, 1] if probs.shape[1] > 1 else probs
        score = roc_auc_score(y_valid, probs)
    else:
        score = knc.score(x_valid, y_valid)
    return score

def weightedPdist(a, b, w, power=2):
    """ Han's faster version"""
    a = a.reshape(-1,len(w))
    b = b.reshape(-1,len(w))

    diff = a[:,np.newaxis].repeat(len(b),1) - b
    return ((np.abs(diff)**2)*w).sum(-1)**(1/2)

import time
def distorted_1nn(x_train, y_train, x_test, y_test, weights, powers=None):
    """ Han's faster version"""
    # start = time.perf_counter()
    dist = weightedPdist(x_test,x_train,weights)
    # dist_time = time.perf_counter()
    examples = get_ds(dist,y_test,y_train)
    # getNINO_time = time.perf_counter()
    acc = eval_ds(dist,examples,y_test,y_train).mean()
    # eval_time = time.perf_counter()
    # print(dist_time-start, getNINO_time-dist_time,eval_time-getNINO_time)
    return acc



def get_ds(dist, y_test, y_train):
    mask_train = np.tile(y_train, (len(y_test), 1))
    apply_mask = lambda x, m: x + (-(m - 1) * x.max())
    ds = np.arange(len(y_test)).reshape(-1, 1)
    for label in np.sort(np.unique(y_train)):
        mask_in = label == mask_train
        in1nn = np.argmin(apply_mask(dist, mask_in), 1)
        ds = np.hstack([ds, in1nn.reshape(-1, 1)])
    return ds

def eval_ds(dist, ds, y_test, y_train):
    dst = dist.take(ds[:,0], 0)
    dnn = np.vstack([np.take_along_axis(
        dst, ds[:, c].reshape(-1,1), 1).ravel() for c in np.arange(1, ds.shape[1])])
    y_pred = np.unique(y_train).take(dnn.argmin(0))
    y_true = y_test.take(ds[:,0])
    return y_pred == y_true

def get_ds_choice(dist, ds):
    dst = dist.take(ds[:,0], 0)
    dnn = np.vstack([np.take_along_axis(
        dst, ds[:, c].reshape(-1,1), 1).ravel() for c in np.arange(1, ds.shape[1])])
    return dnn.argmin(0)

def get_ds_chosen(choice, ds):
    chosen = np.take_along_axis(ds[:,1:], choice.reshape(-1, 1), 1).ravel()
    return chosen

def eval_ds_choice(choice, ds, y_test, y_train):
    chosen = np.take_along_axis(ds[:,1:], choice.reshape(-1, 1), 1).ravel()
    y_pred = y_train.take(chosen)
    y_true = y_test.take(ds[:,0])
    return y_pred == y_true



# def weightedPdist(a, b, weights,powers=[2,2]):
#     weights = np.array(weights)
#     # return np.sqrt(np.sum((weights*np.abs(a-b))**powers))
#     q = np.abs(a-b)
#     return np.sqrt(np.sum(weights*(q**2)))

# def distorted_1nn(x_train, y_train, x_test, y_test, weights, powers=None):
#     ''' 1nn acc given distored weight and power metrics '''
#     correct = 0
#     for x,y in zip(x_test,y_test):
#         dists = [weightedPdist(x, x_, weights, powers) for x_ in x_train]
#         nn = np.argmin(dists)
#         if y_train[nn] == y: correct += 1
        
#     return correct/len(y_test)

def get_wv_df():
    wee_ves_dir = '/net/scratch/hanliu-shared/data/image-data/output/one-class_syn2_size-color-diff-2D'
    tsv_file = os.path.join(wee_ves_dir,'images-config.tsv')
    df = pd.read_table(tsv_file,delim_whitespace=True,header=None)
    df = df.rename(columns={0: "label", 
                    1: "name",
                    2: "index",
                    4: "bodyheadszratio",
                    5: "bodyheadcolordiff",
                    6: "bodysz",
                    7: "bodycolor",
                    8: "bodycolorlighter"
                    })
    features =  ["bodyheadszratio",
                "bodyheadcolordiff",
                "bodysz",
                "bodycolor",
                "bodycolorlighter"]
    def extract_feature(x):
        x_new= x.split('=')[1]
        return x_new
        
    for feature in features:
        df[feature] = df.apply(lambda row : extract_feature(row[feature]), axis = 1)
    for fea in features:
        df[fea] = df[fea].astype('float')

    return df

# def NINO(x_train, y_train, x_test, y_test):
#     ''' Neareast In-class & Nearest Out-of-class '''
#     classes = np.unique(y_train)
#     idx_by_class = {c: np.where(y_train==c)[0] for c in classes}

#     examples = []
#     for x, y in zip(x_test, y_test):
#         example = []
#         for c in classes:
#             class_idx = x_train[idx_by_class[c]]
#             dists = np.array([euc_dist(x, x_) for x_ in class_idx])
#             nn = np.argmin(dists)
#             class_nn = idx_by_class[c][nn]
#             example.append(class_nn)
#         examples.append(example)

#     return np.array(examples)

# def class_1NN_score(x_train, y_train, x_test, y_test):
#     classes = np.unique(y_train)
#     idx_by_class = {c: np.where(y_train==c) for c in classes}
#     dists = euclidean_distances(x_test, x_train)

#     correct = 0
#     total = len(y_test)
#     for i in range(len(y_test)):
#         cur_dist = dists[i]
#         d2idx = {d:j for j,d in enumerate(cur_dist)}
#         examples = [d2idx[min(cur_dist[idx_by_class[c]])] for c in classes]

#         correct += get_knn_score(bm_train_embs[examples], y_train[examples], [bm_valid_embs[i]], [y_test[i]])

#     return correct/total

# def human_1NN_align(x_train, x_valid):
#     embeds = np.concatenate((np.array(x_train),np.array(x_valid)))
#     human_embs = np.concatenate((bm_train_embs,bm_valid_embs))
#     assert(len(embeds) == len(human_embs))

#     correct = 0
#     total = 0
#     for i in range(len(human_embs)):
#         total += 1
#         if get_1nn(human_embs, i) == get_1nn(embeds, i):
#             correct += 1
    
#     return correct/total

def get_1nn(data, index):
    dist = euclidean_distances(data)
    return np.argsort(dist[index])[1]


def bm_eval_human(x_train, y_train, x_valid, y_valid):
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

    print("=" * 60)
    print("human triplet evals:")
    print(f"train_triplet_acc: {train_triplet_acc}")
    print(f"valid_triplet_acc: {valid_triplet_acc}")
    # print(f"val2train_triplet_acc: {val2train_triplet_acc}")
    print(f"knn_acc: {knn_acc}")
    print(f"knn_auc: {knn_auc}")

    return train_triplet_acc, valid_triplet_acc, knn_acc, knn_auc #,val2train_triplet_acc



# bm_triplets_train = np.array(pickle.load(open("../data/bm_triplets/3c2_unique=182/train_triplets.pkl", "rb")))
# bm_triplets_valid = np.array(pickle.load(open("../data/bm_triplets/3c2_unique=182/test_triplets.pkl", "rb")))
# bm_train_embs = np.array(pickle.load(open("embeds/bm/human/TN_train_emb10.pkl","rb")))
# bm_valid_embs = np.array(pickle.load(open("embeds/bm/human/TN_valid_emb10.pkl","rb")))


def wv_eval_human(x_train, x_valid, x_test, y_train, y_valid, y_test, wv_triplets_train_path, wv_triplets_valid_path, wv_triplets_test_path):
    # y_train = np.array([0]*80+[1]*80)
    # y_valid = np.array([0]*20+[1]*20)
    # assert(x_train.shape[0] == 160)
    # assert(x_valid.shape[0] == 40)

    wv_triplets_train = np.array(pickle.load(open(wv_triplets_train_path, "rb")))
    wv_triplets_valid = np.array(pickle.load(open(wv_triplets_valid_path, "rb")))
    wv_triplets_test = np.array(pickle.load(open(wv_triplets_test_path, "rb")))

    train_triplet_acc = get_triplet_acc(x_train, wv_triplets_train)
    valid_triplet_acc = get_triplet_acc(x_valid, wv_triplets_valid)
    test_triplet_acc = get_triplet_acc(x_test, wv_triplets_test)
    knn_acc = get_knn_score(x_train, y_train, x_valid, y_valid, metric="")
    knn_auc = get_knn_score(x_train, y_train, x_valid, y_valid, metric="auc")
    # knn_auc = get_knn_score(x_train, y_train, x_valid, y_valid, metric="auc")
    # human_align = human_1NN_align(x_train, x_valid)
    # class_1NN = class_1NN_score(x_train, y_train, x_valid, y_valid)

    results = {
        "train_triplet_acc":train_triplet_acc,
        "valid_triplet_acc":valid_triplet_acc,
        "test_triplet_acc":test_triplet_acc,
        "knn_acc":knn_acc,
        "knn_auc":knn_auc,
        # "human_1NN_align":human_align,
        # "class_1NN":class_1NN,
    }
    print(results)

    return results


















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