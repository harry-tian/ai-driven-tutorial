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
import pickle
np.random.seed(42)
def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def get_all_knn_scores(x_train, y_train, x_valid, y_valid, k=1, m_range=np.arange(2,11), selection_alg="protodash"):
    full_knn_score = get_knn_score(x_train, y_train, x_valid, y_valid, k)
    random_knn_scores, random_knn_ci = get_random_knn_scores(x_train, y_train, x_valid, y_valid, k, m_range)
    global_prototype_knn_scores = get_global_prototype_knn_scores(x_train, y_train, x_valid, y_valid, k, m_range, selection_alg)
    # local_prototype_knn_scores = get_local_prototype_knn_scores(x_train, y_train, x_valid, y_valid, k, m_range, selection_alg)
    local_prototype_knn_scores = 0
    print(f"full_knn_score: {full_knn_score}")
    print(f"random_knn_scores: {random_knn_scores}")
    print(f"global_prototype_knn_scores: {global_prototype_knn_scores}")
    # print(f"local_prototype_knn_scores: {local_prototype_knn_scores}")

    return full_knn_score, (random_knn_scores, random_knn_ci), global_prototype_knn_scores, local_prototype_knn_scores

def get_local_prototype_knn_scores(x_train, y_train, x_valid, y_valid, k, m_range, selection_alg):
    if selection_alg == "protodash":
        selection_alg = protodash

    classes = np.unique(y_train)
    total = len(x_valid)
    local_prototype_knn_scores = []
    for m in m_range:
        global_prototype_idx = selection_alg(x_train, m)
        # print(global_prototype_idx)
        global_prototype_labels = y_train[global_prototype_idx]
        global_prototypes_byclass = [global_prototype_idx[np.where(global_prototype_labels==c)[0]] for c in classes]
        # print(global_prototypes_byclass)
        correct = 0
        for i, x in enumerate(x_valid):
            x, y = x_valid[i], y_valid[i]
            local_prototype_idx = []
            for prototypes_idx in global_prototypes_byclass:
                local_prototype_idx.append(get_nearest_prototype(x_train, prototypes_idx, x))
            local_prototype_idx = np.array(local_prototype_idx)
            # print(local_prototype_idx)
            correct += get_knn_score(x_train[local_prototype_idx], y_train[local_prototype_idx], np.array([x]), [y], k=k)

        local_prototype_knn_scores.append(correct/total)

    return local_prototype_knn_scores

def get_nearest_prototype(x_train, prototypes_idx, x):
    idx = get_1nn(x_train[prototypes_idx], x) 
    nearest_prototype_idx = prototypes_idx[idx-1]
    # print(nearest_prototype_idx)
    return nearest_prototype_idx

def get_1nn(data, target):
    if len(data) < 2: return 0
    data = np.concatenate((np.array([target]), data))
    dist = euclidean_distances(data)
    nn = np.argsort(dist[0])[1]
    return nn

def get_global_prototype_knn_scores(x_train, y_train, x_valid, y_valid, k, m_range, selection_alg):
    if selection_alg == "protodash":
        selection_alg = protodash

    global_prototype_knn_scores = []
    for m in m_range:
        prototype_idx = selection_alg(x_train, m)
        knn_score = get_knn_score(x_train[prototype_idx], y_train[prototype_idx], x_valid, y_valid, k=k)
        global_prototype_knn_scores.append(knn_score)

    return global_prototype_knn_scores

def protodash(x_train, m, kernelType="Linear"):
    protodash = ProtodashExplainer()
    _, prototype_idx, _ = protodash.explain(x_train, x_train, m=m, kernelType=kernelType)
    return prototype_idx

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

def get_ci(samples, confidence=0.95):
    return 2 * stats.sem(samples) * stats.t.ppf((1 + confidence) / 2., len(samples)-1)

def get_random_knn_scores(x_train, y_train, x_valid, y_valid, k, m_range, n_trials=100):
    random_knn_scores = []
    for m in m_range:
        knn_scores = []
        i = 0
        while i < n_trials:
            random_idx = np.random.choice(range(len(x_train)), m, replace=False)
            if len(np.unique(y_train[random_idx])) < 2: continue
            knn_scores.append(get_knn_score(x_train[random_idx], y_train[random_idx], x_valid, y_valid, k=k))
            i += 1

        random_knn_scores.append(np.array(knn_scores))

    random_knn_scores = np.array(random_knn_scores)
    random_knn_ci = np.array([get_ci(random_knn_scores[m]) for m in range(len(m_range))])
    random_knn_scores = random_knn_scores.mean(axis=-1)

    return random_knn_scores, random_knn_ci

# def get_knn_scores_krange(x_train, y_train, x_valid, y_valid, k_range):
#     f_scores_knn = [get_knn_score(x_train, y_train, x_valid, y_valid, k=k) for k in k_range]
#     return np.array(f_scores_knn) 