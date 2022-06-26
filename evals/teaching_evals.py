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
# sys.path.insert(0,'..')
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter

def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))


def get_knn_score_lpips(lpips_dist, teaching_idx, y_train, y_test, k=1):
    ''' Takes lpips_dist, a distance matrix in the shape of (len(y_train), len(y_test)) '''
    assert(lpips_dist.shape == (len(y_test), len(y_train)))
    assert(len(teaching_idx) > 0 and len(teaching_idx) <= len(y_train))
    assert(min(teaching_idx) >= 0 and max(teaching_idx) <= len(y_train))

    return get_knn_score_dist(lpips_dist[:,teaching_idx], y_train[teaching_idx], y_test, k=k)

def get_knn_score_dist(dist_M, y_train, y_test, k=1):
    assert(len(y_test)==len(dist_M))
    correct = 0
    for y, dists in zip(y_test, dist_M):
        nn_idx = np.argsort(dists)[:k]
        nns = y_train[nn_idx] 
        y_hat = Counter(nns).most_common(1)[0][0]
        if y_hat == y: 
            correct += 1

    return correct/len(y_test)

# def get_full_random(data,m_range):
#     x_train, y_train, x_valid, y_test = data
#     return get_knn_score(x_train, y_train, x_valid, y_test), get_random_knn_scores(data, m_range)

def get_random_knn_scores(data, m_range, k=1, n_trials=100):
    def get_ci(samples, confidence=0.95):
        return 2 * stats.sem(samples) * stats.t.ppf((1 + confidence) / 2., len(samples)-1)

    x_train, y_train, x_valid, y_test = data
    random_knn_scores = []
    for m in m_range:
        knn_scores = []
        i = 0
        while i < n_trials:
            random_idx = np.random.choice(np.arange(len(y_train)), m, replace=False)
            if len(np.unique(y_train[random_idx])) < 2: continue
            knn_scores.append(get_knn_score(x_train[random_idx], y_train[random_idx],x_valid, y_test, k=k))
            i += 1

        random_knn_scores.append(np.array(knn_scores))

    random_knn_scores = np.array(random_knn_scores)
    random_knn_ci = np.array([get_ci(random_knn_scores[m]) for m in range(len(m_range))])
    random_knn_scores = random_knn_scores.mean(axis=-1)

    return random_knn_scores, random_knn_ci

def get_knn_score(x_train, y_train, x_valid, y_test, 
                k=1, metric="acc", weights="uniform"):
    knc = KNeighborsClassifier(n_neighbors=k, weights=weights)
    knc.fit(x_train, y_train)
    if metric == 'auc':
        probs = knc.predict_proba(x_valid)
        probs = probs[:, 1] if probs.shape[1] > 1 else probs
        score = roc_auc_score(y_test, probs)
    else:
        score = knc.score(x_valid, y_test)
    return score

def get_1nn(data, index):
    dist = euclidean_distances(data)
    return np.argsort(dist[index])[1]


# def human_1NN_align(embeds, proto_idx):
#     human_embs = pickle.load(open("data/embeds/bm/human/TN_train_emb10.pkl","rb"))
#     assert(len(embeds) == len(human_embs))

#     correct = 0
#     total = 0
#     human_embs = human_embs[proto_idx]
#     embeds = embeds[proto_idx]
#     for idx in range(len(proto_idx)):
#         total += 1
#         if get_1nn(human_embs, idx) == get_1nn(embeds, idx):
#             correct += 1
    
#     return correct/total

# def get_euc_knn_score(x_train, y_train, x_valid, y_test, prototype_idx,
#                     k=1, metric="acc", weights="uniform"):
#     x_train, y_train = x_train[prototype_idx], y_train[prototype_idx]
#     knc = KNeighborsClassifier(n_neighbors=k, weights=weights)
#     knc.fit(x_train, y_train)
#     if metric == 'auc':
#         probs = knc.predict_proba(x_valid)
#         probs = probs[:, 1] if probs.shape[1] > 1 else probs
#         score = roc_auc_score(y_test, probs)
#     else:
#         score = knc.score(x_valid, y_test)
#     return score



# def get_all_knn_scores(data, knn_f="euc", m_range=np.arange(2,11), selection_alg="protodash", k=1, knn_metric="acc"):
#     if knn_f == "euc":
#         knn_f = get_euc_knn_score
#     elif knn_f == "lpips":
#         knn_f = get_lpips_knn_score

#     x_train, y_train, x_valid, y_test = data
    
#     full_knn_score = knn_f(x_train, y_train, x_valid, y_test, np.arange(len(x_train)), k, metric=knn_metric)
#     random_knn_scores, random_knn_ci = get_random_knn_scores(x_train, y_train, x_valid, y_test, knn_f,
#                                                                 k=k, metric=knn_metric, m_range=m_range)
#     global_prototype_knn_scores_LK = get_global_prototype_knn_scores(x_train, y_train, x_valid, y_test, knn_f,
#                                                                 k=k, metric=knn_metric, m_range=m_range, 
#                                                                 selection_alg=selection_alg, kernel="Linear")
#     # global_prototype_knn_scores_RBFK = get_global_prototype_knn_scores(x_train, y_train, x_valid, y_test, knn_f,
#     #                                                             k=k, metric=knn_metric, m_range=m_range, 
#     #                                                             selection_alg=selection_alg, kernel="Gaussian")
#     # local_prototype_knn_scores = get_local_prototype_knn_scores(x_train, y_train, x_valid, y_test, k, m_range, selection_alg)

#     print(f"full_knn_score: {full_knn_score}")
#     print(f"random_knn_scores: {random_knn_scores}")
#     print(f"global_prototype_linear kernel: {global_prototype_knn_scores_LK}")
#     # print(f"global_prototype_gaussian kernel: {global_prototype_knn_scores_RBFK}")

#     return full_knn_score, (random_knn_scores, random_knn_ci), global_prototype_knn_scores_LK #, global_prototype_knn_scores_RBFK , local_prototype_knn_scores


# def get_lpips_1nn_score(x_train, y_train, x_valid, y_test, lpips_distM):
#     if lpips_distM.shape != (len(y_test), len(y_train)):
#         lpips_distM = lpips_distM.T
#     assert(lpips_distM.shape==(len(y_test), len(y_train)))

#     total = len(y_test)
#     correct = 0
#     for x, y, dists in zip(x_valid, y_test, lpips_distM):
#         nn_idx = np.argmin(lpips_distM)
#         if y_train[nn_idx] == y: correct += 1

#     return correct/total

# def get_knn_scores_krange(x_train, y_train, x_valid, y_test, k_range):
#     f_scores_knn = [get_knn_score(x_train, y_train, x_valid, y_test, k=k) for k in k_range]
#     return np.array(f_scores_knn) 

# def get_local_prototype_knn_scores(x_train, y_train, x_valid, y_test, k, m_range, selection_alg):
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
#             x, y = x_valid[i], y_test[i]
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

# def get_1nn_score(x_train, y_train, x_valid, y_test, dist_f=euc_dist):
#     total = len(y_test)
#     correct = 0
#     for x, y in zip(x_valid, y_test):
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
