# import pandas as pd
import numpy as np
from scipy import stats
# from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
# from sklearn.svm import SVC, LinearSVC
# from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
# import math, pickle
np.random.seed(42)
import sys, pickle
sys.path.insert(0,'..')
import seaborn as sn

def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
y_train = np.array([0]*80+[1]*80)
y_valid = np.array([0]*20+[1]*20)
full_idx = np.arange(160)

########### knn helpers ############################

def get_full_random(data,m_range):
    x_train, y_train, x_valid, y_valid = data
    return get_knn_score(x_train, y_train, x_valid, y_valid), get_random_knn_scores(data, m_range)

def get_random_knn_scores(data, m_range, k=1, n_trials=100):
    def get_ci(samples, confidence=0.95):
        return 2 * stats.sem(samples) * stats.t.ppf((1 + confidence) / 2., len(samples)-1)

    x_train, y_train, x_valid, y_valid = data
    random_knn_scores = []
    for m in m_range:
        knn_scores = []
        i = 0
        while i < n_trials:
            random_idx = np.random.choice(full_idx, m, replace=False)
            if len(np.unique(y_train[random_idx])) < 2: continue
            knn_scores.append(get_knn_score(x_train[random_idx], y_train[random_idx],x_valid, y_valid, k=k))
            i += 1

        random_knn_scores.append(np.array(knn_scores))

    random_knn_scores = np.array(random_knn_scores)
    random_knn_ci = np.array([get_ci(random_knn_scores[m]) for m in range(len(m_range))])
    random_knn_scores = random_knn_scores.mean(axis=-1)

    return random_knn_scores, random_knn_ci

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


# def get_lpips_knn_score(prototype_idx=full_idx, k=1):
#     lpips = pickle.load(open("embeds/lpips/lpips.bm.trxvl.pkl","rb"))
        
#     dists = lpips[prototype_idx].T
#     total = len(y_valid)
#     assert(total==len(dists))
#     correct = 0
#     for y, dist in zip(y_valid, dists):
#         nn_idx = np.argmin(dist)
#         if y_train[prototype_idx][nn_idx] == y: correct += 1

#     return correct/total

# def get_euc_knn_score(x_train, y_train, x_valid, y_valid, prototype_idx,
#                     k=1, metric="acc", weights="uniform"):
#     x_train, y_train = x_train[prototype_idx], y_train[prototype_idx]
#     knc = KNeighborsClassifier(n_neighbors=k, weights=weights)
#     knc.fit(x_train, y_train)
#     if metric == 'auc':
#         probs = knc.predict_proba(x_valid)
#         probs = probs[:, 1] if probs.shape[1] > 1 else probs
#         score = roc_auc_score(y_valid, probs)
#     else:
#         score = knc.score(x_valid, y_valid)
#     return score



# def get_all_knn_scores(data, knn_f="euc", m_range=np.arange(2,11), selection_alg="protodash", k=1, knn_metric="acc"):
#     if knn_f == "euc":
#         knn_f = get_euc_knn_score
#     elif knn_f == "lpips":
#         knn_f = get_lpips_knn_score

#     x_train, y_train, x_valid, y_valid = data
    
#     full_knn_score = knn_f(x_train, y_train, x_valid, y_valid, np.arange(len(x_train)), k, metric=knn_metric)
#     random_knn_scores, random_knn_ci = get_random_knn_scores(x_train, y_train, x_valid, y_valid, knn_f,
#                                                                 k=k, metric=knn_metric, m_range=m_range)
#     global_prototype_knn_scores_LK = get_global_prototype_knn_scores(x_train, y_train, x_valid, y_valid, knn_f,
#                                                                 k=k, metric=knn_metric, m_range=m_range, 
#                                                                 selection_alg=selection_alg, kernel="Linear")
#     # global_prototype_knn_scores_RBFK = get_global_prototype_knn_scores(x_train, y_train, x_valid, y_valid, knn_f,
#     #                                                             k=k, metric=knn_metric, m_range=m_range, 
#     #                                                             selection_alg=selection_alg, kernel="Gaussian")
#     # local_prototype_knn_scores = get_local_prototype_knn_scores(x_train, y_train, x_valid, y_valid, k, m_range, selection_alg)

#     print(f"full_knn_score: {full_knn_score}")
#     print(f"random_knn_scores: {random_knn_scores}")
#     print(f"global_prototype_linear kernel: {global_prototype_knn_scores_LK}")
#     # print(f"global_prototype_gaussian kernel: {global_prototype_knn_scores_RBFK}")

#     return full_knn_score, (random_knn_scores, random_knn_ci), global_prototype_knn_scores_LK #, global_prototype_knn_scores_RBFK , local_prototype_knn_scores


# def get_lpips_1nn_score(x_train, y_train, x_valid, y_valid, lpips_distM):
#     if lpips_distM.shape != (len(y_valid), len(y_train)):
#         lpips_distM = lpips_distM.T
#     assert(lpips_distM.shape==(len(y_valid), len(y_train)))

#     total = len(y_valid)
#     correct = 0
#     for x, y, dists in zip(x_valid, y_valid, lpips_distM):
#         nn_idx = np.argmin(lpips_distM)
#         if y_train[nn_idx] == y: correct += 1

#     return correct/total

# def get_knn_scores_krange(x_train, y_train, x_valid, y_valid, k_range):
#     f_scores_knn = [get_knn_score(x_train, y_train, x_valid, y_valid, k=k) for k in k_range]
#     return np.array(f_scores_knn) 

# def get_local_prototype_knn_scores(x_train, y_train, x_valid, y_valid, k, m_range, selection_alg):
#     if selection_alg == "protodash":
#         selection_alg = protodash

#     classes = np.unique(y_train)
#     total = len(x_valid)
#     local_prototype_knn_scores = []
#     for m in m_range:
#         global_prototype_idx = selection_alg(x_train, m)
#         # print(global_prototype_idx)
#         global_prototype_labels = y_train[global_prototype_idx]
#         global_prototypes_byclass = [global_prototype_idx[np.where(global_prototype_labels==c)[0]] for c in classes]
#         # print(global_prototypes_byclass)
#         correct = 0
#         for i, x in enumerate(x_valid):
#             x, y = x_valid[i], y_valid[i]
#             local_prototype_idx = []
#             for prototypes_idx in global_prototypes_byclass:
#                 local_prototype_idx.append(get_nearest_prototype(x_train, prototypes_idx, x))
#             local_prototype_idx = np.array(local_prototype_idx)
#             # print(local_prototype_idx)
#             correct += get_knn_score(x_train[local_prototype_idx], y_train[local_prototype_idx], np.array([x]), [y], k=k)

#         local_prototype_knn_scores.append(correct/total)

#     return local_prototype_knn_scores

# def get_nearest_prototype(x_train, prototypes_idx, x):
#     idx = get_1nn(x_train[prototypes_idx], x) 
#     nearest_prototype_idx = prototypes_idx[idx-1]
#     # print(nearest_prototype_idx)
#     return nearest_prototype_idx

# def get_1nn_score(x_train, y_train, x_valid, y_valid, dist_f=euc_dist):
#     total = len(y_valid)
#     correct = 0
#     for x, y in zip(x_valid, y_valid):
#         nn_idx = get_1nn(x_train, x, dist_f)
#         if y_train[nn_idx] == y: correct += 1

#     return correct/total


# def get_1nn(data, target, dist_f):
#     if len(data) < 2: return 0
#     min_dist = math.inf
#     min_idx = 0
#     for idx, cur_data in enumerate(data):
#         dist = dist_f(cur_data, target)
#         if dist < min_dist: 
#             min_dist = dist
#             min_idx = idx

#     return min_idx
