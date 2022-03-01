
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC, LinearSVC
from aix360.algorithms.protodash import ProtodashExplainer
from sklearn.metrics.pairwise import euclidean_distances
from torchvision import transforms
import tste
import pickle
import matplotlib.pyplot as plt

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

def get_1nn_lpips(index, samples):
    target = samples[index].copy()
    sample_index = {sample:i for i,sample in enumerate(target)}
    target.sort()
    return sample_index[target[1]]

def get_1nn(index, samples):
    if len(samples) < 2:
        return 0
    # print(samples.shape)
    # print(samples)
    dist = euclidean_distances(samples)
    nn = np.argsort(dist[index])[1]

    return nn

def get_nns(index, samples, m=1):
    dist = euclidean_distances(samples)
    neighbors = []
    for i in index:
        neighbors.append(np.argsort(dist[i])[1:m+1])
    return np.array(neighbors)

def get_nn_triplets(index, triplets, m=1):
    neighbors = {}
    for triplet in triplets:
        if triplet[0] == index:
            pos = triplet[1]
            if pos not in neighbors:
                neighbors[pos] = 1 
            else:
                neighbors[pos] += 1

    max_count = 0
    for value in neighbors.values():
        if value > max_count:
            max_count = value

    nns = []
    for key, value in neighbors.items():
        if value == max_count:
            nns.append(key)

    return np.array(nns)

def prototype_knn(data, proto_idx, k_range, m_range):
    scores = []
    for k in k_range:
        for m in m_range:
            scores.append(get_knn_score(k, data, proto_idx[m]))

    return np.array(scores).reshape(len(k_range), len(m_range))



### visualization stuff ######################

if True:
    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def vis_data(x_train, y_train, x_valid, y_valid, title, save=False):
    x_all = np.concatenate((x_train, x_valid))
    y_all = np.concatenate((y_train, y_valid))
    classes = np.unique(y_train)
    subtitles = ["all", "train", "valid"]
    fig, ax = plt.subplots(1, len(subtitles), figsize=(24, 6))

    for i, data in enumerate([(x_all, y_all), (x_train, y_train),(x_valid, y_valid)]):
        x, y = data
        for c in classes:
            c_idx = np.where(y==c)[0]
            ax[i].scatter(x[c_idx][:,0], x[c_idx][:,1])

        ax[i].legend(['non-cancer','cancer'])
        ax[i].set_title("{}".format(subtitles[i]))
    
    utils.normalize_xylim(ax)
    fig.suptitle(title)
    if save:
        plt.savefig(f"{title}.png", dpi=300)

def vis_data_all(x, y, title, save=False):
    classes = np.unique(y)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    for i, data in enumerate([(x,y),(x,y)]):
        x, y = data
        for c in classes:
            c_idx = np.where(y==c)[0]
            ax[i].scatter(x[c_idx][:,0], x[c_idx][:,1])

        ax[i].legend(['non-cancer','cancer'])
    
    utils.normalize_xylim(ax)
    fig.suptitle(title)
    if save:
        plt.savefig(f"{title}.png", dpi=300)

def vis_proto(x_train, y_train,train_proto_idx, title, save=False, order=False, eps=0, x_valid=None, y_valid=None,valid_proto_idx=None):
    classes = np.unique(y_train)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # for i, data in enumerate([(x_train, y_train),(x_valid, y_valid)]):
    for i, data in enumerate([(x_train, y_train)]):
        x, y = data
        for c in classes:
            c_idx = np.where(y==c)[0]
            ax[i].scatter(x[c_idx][:,0], x[c_idx][:,1])
        ax[i].legend(['non-cancer','cancer'])
        ax[i].set_title(title)
        # ax[i].set_title("{}.{}.{}".format(model, name, "train" if i == 0 else "valid"))

    for j, c in enumerate(classes):
        train_proto = x_train[[int(i) for i in train_proto_idx if y_train[i] == c]]
        ax[0].scatter(train_proto[:,0], train_proto[:,1], s=300, c=f"C{str(j)}", marker='^', linewidths=1, edgecolors='k') 
        # valid_proto = x_valid[[int(i) for i in valid_proto_idx if y_valid[i] == c]]
        # ax[1].scatter(valid_proto[:,0], valid_proto[:,1], s=300, c=f"C{str(j)}",marker='^', linewidths=1, edgecolors='k') 

    for o, i in enumerate(train_proto_idx):
        proto = x_train[i]
        c = plt.Circle(proto, radius=eps, fill=False,lw=0.75)
        
        if order:
            ax[0].text(proto[0],proto[1],str(o), size="xx-large", weight="bold", c="0")
        ax[0].add_patch(c)

    if save:
        plt.savefig(f"{title}.png", dpi=300)

def vis_triplets(x_train, y_train, triplet_dict, title, save=False):
    classes = np.unique(y_train)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for i, data in enumerate([(x_train, y_train)]):
        x, y = data
        for c in classes:
            c_idx = np.where(y==c)[0]
            ax[i].scatter(x[c_idx][:,0], x[c_idx][:,1])
        
        ax[i].legend(['non-cancer','cancer'])
        ax[i].set_title("{}".format("train" if i == 0 else "valid"))
    
    for i, triplet in enumerate(triplet_dict):
        anchor, pos, neg = triplet
        anchor = x_train[anchor]
        pos = x_train[pos]
        neg = x_train[neg]
        ax[0].scatter(anchor[0], anchor[1], s=300, c='0.5', marker='^', linewidths=1, edgecolors='k') 
        ax[0].scatter(pos[0], pos[1], s=300, c='0', marker='^', linewidths=1, edgecolors='k') 
        ax[0].scatter(neg[0], neg[1], s=300, c='1', marker='^', linewidths=1, edgecolors='k') 
        ax[0].text(anchor[0], anchor[1], str(i), c='0', size="xx-large", weight="bold") 
        ax[0].text(pos[0], pos[1], str(i), c='0', size="xx-large", weight="bold") 
        ax[0].text(neg[0], neg[1], str(i), c='0', size="xx-large", weight="bold") 

    # utils.normalize_xylim(ax)
    fig.suptitle(title)
    if save:
        plt.savefig(f"{title}.png", dpi=300)

def vis_proto_2(x_train, y_train, proto_idx1, proto_idx2, title, save=False, order=False, eps=0):
    classes = np.unique(y_train)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for i, data in enumerate([(x_train, y_train)]):
        x, y = data
        for c in classes:
            c_idx = np.where(y==c)[0]
            ax[i].scatter(x[c_idx][:,0], x[c_idx][:,1])
        ax[i].legend(['non-cancer','cancer'])
        ax[i].set_title(title)

        protos1 = x_train[proto_idx1]
        protos2 = x_train[proto_idx2]
        ax[i].scatter(protos1[:,0], protos1[:,1], s=300, c = 'g', marker='^', linewidths=1, edgecolors='k') 
        ax[i].scatter(protos2[:,0], protos2[:,1], s=300, c = 'r', marker='^', linewidths=1, edgecolors='k') 
    if save:
        plt.savefig(f"{title}.png", dpi=300)
    
def vis_knn(k_range, m_range, scores, legend, title, save=False):
    lw = 3
    fig, ax = plt.subplots(1, len(k_range), figsize=(24,6), sharey=True)
    for k in range(len(k_range)):
        for score in scores:
            ax[k].plot(m_range, score[k], lw=lw)

        ax[k].legend(legend, loc='lower right')
        ax[k].set_title('K={}'.format(k_range[k]))
    fig.suptitle(title)

    if save:
        plt.savefig(f"{title}.png", dpi=300)

def normalize_xylim(ax):
    min_x0 = np.inf
    max_x1 = np.NINF
    min_y0 = np.inf
    max_y1 = np.NINF
    xlims = []
    ylims = []
    for i in range(len(ax)):
        xlims.append(ax[i].get_xlim()[0])
        xlims.append(ax[i].get_xlim()[1])
        ylims.append(ax[i].get_ylim()[0])
        ylims.append(ax[i].get_ylim()[1])

    min_x0 = min(xlims)
    max_x1 = max(xlims)
    min_y0 = min(ylims)
    max_y1 = max(ylims)
    
    for i in range(len(ax)):
        ax[i].set_xlim(min_x0, max_x1)
        ax[i].set_ylim(min_y0, max_y1)

### not very useful stuff: ###

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
