
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC, LinearSVC
from aix360.algorithms.protodash import ProtodashExplainer
from torchvision import transforms
import tste
import pickle

def get_tste(distance_matrix, triplets_fname, tste_fname, no_dims=2, max_iter=1000):
    triplets = []
    for point0, point0_dist_list in enumerate(distance_matrix):
        for point1, point1_dist in enumerate(point0_dist_list[point0+1:]):
            for point2, point2_dist in enumerate(point0_dist_list[point0+2:]):
                anchor = point0
                if point1_dist < point2_dist:
                    pos = point1+1
                    neg = point2+2
                else: 
                    pos = point2+2
                    neg = point1+1
                triplets.append([anchor, pos, neg])

    triplets = np.array(triplets)
    pickle.dump(triplets, open(triplets_fname,"wb"))   
    print(f"triplets saved to {triplets_fname}")

    embedding = tste.tste(triplets, no_dims=no_dims, verbose=True, max_iter=max_iter)
    pickle.dump(embedding, open(tste_fname,"wb"))  
    print(f"tste embedding saved to {tste_fname}")

    return embedding

def bm_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

METRIC = "auc"
WEIGHTS = "uniform"
LINEAR = True

def get_knn_score(k, data, index, metric="auc", weights="uniform"):
    x_train, y_train, x_valid, y_valid = data
    knc = KNeighborsClassifier(n_neighbors=k, weights=weights)
    knc.fit(x_train[index], y_train[index])
    if metric == 'auc':
        probs = knc.predict_proba(x_valid)
        probs = probs[:, 1] if probs.shape[1] > 1 else probs
        score = roc_auc_score(y_valid, probs)
    else:
        score = knc.score(x_valid, y_valid)
    return score


def get_svm_score(k, data, index, metric=METRIC, linear=LINEAR):
    X_train, y_train, X_valid, y_valid = data
    svc = LinearSVC(random_state=42) if linear else SVC(probability=True, random_state=42)
    svc.fit(X_train[index], y_train[index])
    if metric == 'auc':
        probs = svc.decision_function(X_valid) # if linear else svc.predict_proba(X_valid)
        probs = probs[:, 1] if len(probs.shape) > 1 and probs.shape[1] > 1 else probs
        score = roc_auc_score(y_valid, probs)
    else:
        score = svc.score(X_valid, y_valid)
    return score

def get_ci(samples, confidence=0.95):
    return 2 * stats.sem(samples) * stats.t.ppf((1 + confidence) / 2., len(samples)-1)

def get_full_score(data, k_range):
    f_scores_knn = []
    for k in k_range:
        score = get_knn_score(k, data, list(range(len(data[0]))))
        f_scores_knn.append(score)
    f_scores_knn = np.array(f_scores_knn)
    f_score_svm = get_svm_score(k, data, list(range(len(data[0]))))
    return f_scores_knn, f_score_svm


def get_random_score(data, k_range, m_range, n_trials=100):
    X_train, y_train, X_valid, y_valid = data
    
    r_scores_knn, r_scores_svm = [], []
    np.random.seed(42)
    for k in k_range:
        for m in m_range:
            scores_knn, scores_svm = [], []
            i = 0
            while i < n_trials:
                index = np.random.choice(range(len(X_train)), m, replace=False)
                if len(np.unique(y_train[index])) < 2: continue
                scores_knn.append(get_knn_score(k, data, index))
                scores_svm.append(get_svm_score(k, data, index))
                i += 1
            r_scores_knn.append((scores_knn))
            r_scores_svm.append((scores_svm))
    r_scores_knn = np.array(r_scores_knn).reshape(len(k_range), len(m_range), n_trials)
    r_scores_svm = np.array(r_scores_svm).reshape(len(k_range), len(m_range), n_trials)
    r_means_knn = r_scores_knn.mean(axis=-1)
    r_confs_knn = np.array([get_ci(r_scores_knn[k][m]) for m in range(len(m_range)) for k in range(len(k_range))]).reshape(len(k_range), len(m_range))
    r_means_svm = r_scores_svm.mean(axis=-1)
    r_confs_svm = np.array([get_ci(r_scores_svm[k][m]) for m in range(len(m_range)) for k in range(len(k_range))]).reshape(len(k_range), len(m_range))

    return r_means_knn, r_confs_knn, r_means_svm, r_confs_svm

def get_protodash_score(data, k_range, m_range):
    p_idss = {}
    X_train, y_train, X_valid, y_valid = data
    for m in m_range:
        if m not in p_idss:
            try:
                protodash = ProtodashExplainer()
                _, index, _ = protodash.explain(X_train, X_train, m=m, kernelType="Gaussian")
            except AttributeError:
                index = [0] * m
                print("error for m={}".format(m))
            p_idss[m] = index
    # pickle.dump(p_idss, open("p_index.dwac.emb10.merged.pkl", "wb"))
    # p_idss = pickle.load(open("p_index.dwac.emb10.merged.pkl", "rb"))
    p_scores_knn, p_scores_svm = [], []
    for k in k_range:
        for m in m_range:
            p_scores_knn.append(get_knn_score(k, data, p_idss[m]))
            try:
                s = get_svm_score(k, data, p_idss[m])
            except:
                s = 0
            p_scores_svm.append(s)
    p_scores_knn = np.array(p_scores_knn).reshape(len(k_range), len(m_range))
    p_scores_svm = np.array(p_scores_svm).reshape(len(k_range), len(m_range))
    return p_scores_knn, p_scores_svm

def get_nn(index, samples, m=1):
    dist = euclidean_distances(samples)
    neighbors = []
    for i in index:
        neighbors.append(np.argsort(dist[i])[1:m+1])
    return np.array(neighbors)

def prototype_knn(data, proto_idx, k_range, m_range):
    scores = []
    for k in k_range:
        for m in m_range:
            scores.append(get_knn_score(k, data, proto_idx[m]))

    return np.array(scores).reshape(len(k_range), len(m_range))

def normalize_xylim(ax):
    min_x0 = np.inf
    max_x1 = np.NINF
    min_y0 = np.inf
    max_y1 = np.NINF
    for i in range(2):
        xlim = ax[i].get_xlim()
        ylim = ax[i].get_ylim()
        min_x0 = xlim[0] if xlim[0] < min_x0 else min_x0
        max_x1 = xlim[1] if xlim[1] > max_x1 else max_x1
        min_y0 = ylim[0] if ylim[0] < min_y0 else min_y0
        max_y1 = ylim[1] if ylim[1] > max_y1 else max_y1
    ax[0].set_xlim(min_x0, max_x1)
    ax[0].set_ylim(min_y0, max_y1)
    ax[1].set_xlim(min_x0, max_x1)
    ax[1].set_ylim(min_y0, max_y1)
