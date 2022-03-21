# import pandas as pd
import numpy as np
from scipy import stats
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC, LinearSVC
from aix360.algorithms.protodash import ProtodashExplainer
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import math, pickle
np.random.seed(42)
import sys
sys.path.insert(0,'..')
import algorithms.pdash as pdash

def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
# global y_train
# global y_valid
y_train = np.array([0]*80+[1]*80)
y_valid = np.array([0]*20+[1]*20)
full_idx = np.arange(160)

########### Synthetic learners ############################

def get_lpips_knn_score(prototype_idx=full_idx, k=1):
    lpips = pickle.load(open("embeds/lpips/lpips.bm.trxvl.pkl","rb"))
        
    dists = lpips[prototype_idx].T
    total = len(y_valid)
    assert(total==len(dists))
    correct = 0
    for y, dist in zip(y_valid, dists):
        nn_idx = np.argmin(dist)
        if y_train[prototype_idx][nn_idx] == y: correct += 1

    return correct/total

def get_htriplet_knn_score(prototype_idx=full_idx, k=1):
    train_embs = pickle.load(open("embeds/bm/human/TN_train_emb10.pkl","rb"))
    valid_embs = pickle.load(open("embeds/bm/human/TN_valid_emb10.pkl","rb"))
    x_train = train_embs[prototype_idx]
    x_valid = valid_embs

    return get_knn_score(x_train, y_train[prototype_idx], x_valid, y_valid, metric="acc")

########### selection algorithms ############################
def protodash(X, m, kernel="Gaussian"):
    protodash = ProtodashExplainer()
    _, prototype_idx, _ = protodash.explain(X, X, m=m, kernelType=kernel)
    return prototype_idx

def protogreedy(X, m, kernel=None):
    return pdash.proto_g(X, np.arange(len(X)), m)

########### knn helpers ############################

def get_prototype_knn_scores(X, knn_dist, m_range, selection_alg, kernel="Linear", k=1):
    if selection_alg == "protodash":
        selection_alg = protodash
    elif selection_alg == "protogreedy":
        selection_alg = protogreedy
    knn = get_lpips_knn_score if knn_dist == "lpips" else get_htriplet_knn_score

    prototype_knn_scores = []
    for m in m_range:
        prototype_idx = selection_alg(X, m, kernel)
        knn_score = knn(prototype_idx, k=k)
        prototype_knn_scores.append(knn_score)

    print(prototype_knn_scores)
    return prototype_knn_scores

def get_random_knn_scores(knn_dist, m_range, k=1, n_trials=100):
    def get_ci(samples, confidence=0.95):
        return 2 * stats.sem(samples) * stats.t.ppf((1 + confidence) / 2., len(samples)-1)
    knn = get_lpips_knn_score if knn_dist == "lpips" else get_htriplet_knn_score

    random_knn_scores = []
    for m in m_range:
        knn_scores = []
        i = 0
        while i < n_trials:
            random_idx = np.random.choice(full_idx, m, replace=False)
            if len(np.unique(y_train[random_idx])) < 2: continue
            knn_scores.append(knn(random_idx, k=k))
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

########### visualizations ############################

if True:
    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    colors = ["black", "","red", "blue", "green","yellow"]
    linewidth = 4.0

def vis_all_knn_scores(m_range, all_scores, legend=None, title=None, save=False):
    plt.figure(figsize=(16,10))

    for i, scores in enumerate(all_scores):
        if i == 0:
            full_knn_score = scores    
            plt.axhline(full_knn_score, c='black', linewidth=linewidth)
        elif i == 1:
            random_knn_scores, random_knn_ci = scores
            plt.plot(m_range, random_knn_scores, linewidth=linewidth)
            plt.fill_between(m_range, random_knn_scores + random_knn_ci / 2, random_knn_scores - random_knn_ci / 2, alpha=0.5)
        else:
            plt.plot(m_range, scores, linewidth=linewidth)#, c=colors[i])

        # plt.set_xticks(m_range)
        plt.legend(legend)

    if not title:
        title = "knn_scores"
    plt.title(title)
    if save:
        plt.savefig(f"{title}.png", dpi=600)

def vis_all_knn_scores_multiplot(m_range, all_scores, legend=None, subtitles=None, title=None, save=False):
    num_figs = len(all_scores)
    fig, ax = plt.subplots(1, num_figs, figsize=(8*num_figs, 6), sharey=True)
    for i, scores in enumerate(all_scores):
        if i == 0:
            full_knn_score = scores    
            ax[i].axhline(full_knn_score, c='black')
        elif i == 1:
            random_knn_scores, random_knn_ci = scores
            ax[i].plot(m_range, random_knn_scores)
            ax[i].fill_between(m_range, random_knn_scores + random_knn_ci / 2, random_knn_scores - random_knn_ci / 2, alpha=0.5)
        else:
            ax[i].plot(m_range, scores, c=colors[i])
        # for j, score in enumerate(scores):

        ax[i].set_ylim(0.3, 1)
        ax[i].set_xticks(m_range)
        if subtitles: ax[i].set_title(subtitles[i])
        if i == 0 or i == num_figs - 1:
            ax[i].legend(legend)

    if not title:
        title = "knn_scores"
    fig.suptitle(title)
    if save:
        plt.savefig(f"{title}.png", dpi=600)


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
