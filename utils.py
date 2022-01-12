
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
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
    pickle.dump(np.array(triplets), open(triplets_fname,"wb"))   
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

def get_ci(samples, confidence=0.95):
    return 2 * stats.sem(samples) * stats.t.ppf((1 + confidence) / 2., len(samples)-1)

def get_full_score(x_train, y_train, x_valid, y_valid, k_range):
    data = x_train, y_train, x_valid, y_valid
    lr = LogisticRegression(random_state=42)
    lr.fit(x_train, y_train)
    print(lr.score(x_valid, y_valid))
    coef = np.vstack([lr.coef_, -lr.coef_])
    # W = extract_W_multiclass(coef, x_train, y_train, coo=False)

    f_scores = []
    for k in k_range:
        score = get_knn_score(k, data, list(range(len(data[0]))))
        f_scores.append(score)
    f_scores = np.array(f_scores)
    return f_scores

def get_random_score(x_train, y_train, x_valid, y_valid, k_range, m_range, n_trials=100):
    data = x_train, y_train, x_valid, y_valid
    r_scores = []
    np.random.seed(42)
    for k in k_range:
        for m in m_range:
            scores = []
            for i in range(n_trials):
                index = np.random.choice(range(len(x_train)), m, replace=False)
                scores.append(get_knn_score(k, data, index))
            r_scores.append((scores))
    r_scores = np.array(r_scores).reshape(len(k_range), len(m_range), n_trials)
    # pickle.dump(r_scores, open("rscores.{}.emb10.pkl".format(WEIGHTS), "wb"))
    # r_scores = pickle.load(open("rscores.{}.emb10.pkl".format(WEIGHTS), "rb"))
    r_means = r_scores.mean(axis=-1)
    r_confs = np.array([get_ci(r_scores[k][m]) for m in range(len(m_range)) for k in range(len(k_range))])
    return r_means, r_confs

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