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
import sys, pickle, random
# sys.path.insert(0,'..')
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter

def normalize(data): return (data-np.min(data)) / (np.max(data)-np.min(data))
def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
def most_common(S, k=2): 
    """ ASSUMES 2 CLASSES
        takes the most common element in a list, breaks ties randomly
    """
    if len(S) == 1 or len(np.unique(S)) == 1: return S[0]
    counts = Counter(S).most_common(k)
    majority = 0 if counts[0][1] > counts[1][1] else random.choice([0,1])
    return counts[majority][0]
def dist2sim(M): return np.e**(-20*M)
def max_dict(d):
    max_v = -np.inf
    for k,v in d.items(): 
        if v >= max_v:
            max_k = k
            max_v = v
    return max_k
# def min_dict(d):

### learners: 1NN, exemplar, contrastive voter

def eval_KNN(dist_M, teaching_idx, y_train, y_test, sim=False, k=1):
    """ Takes dist_M, a distance matrix in the shape of (len(y_test), len(y_train)) 
    """
    assert(dist_M.shape == (len(y_test), len(y_train)))
    assert(len(teaching_idx) > 0 and len(teaching_idx) <= len(y_train))
    assert(min(teaching_idx) >= 0 and max(teaching_idx) <= len(y_train))
    fn = get_knn_score_sim if sim else get_knn_score_dist
    return fn(dist_M[:,teaching_idx], y_train[teaching_idx], y_test, k=k)

def get_knn_score_dist(dist_M, y_train, y_test, k=1):
    assert(len(y_test)==len(dist_M))
    correct = 0
    for y, dists in zip(y_test, dist_M):
        nn_idx = np.argsort(dists)[:k]
        nns = y_train[nn_idx] 
        y_hat = most_common(nns)
        if y_hat == y: 
            correct += 1

    return correct/len(y_test)

def get_knn_score_sim(sim_M, y_train, y_test, k=1):
    assert(len(y_test)==len(sim_M))
    correct = 0
    for y, dists in zip(y_test, sim_M):
        nn_idx = np.argsort(dists)[-k:]
        nns = y_train[nn_idx] 
        y_hat = most_common(nns)
        if y_hat == y: 
            correct += 1

    return correct/len(y_test)

def eval_exemplar(dist_M, exemplar_idx, y_train, y_test, sim=False):
    ''' Takes dist_M, a distance matrix in the shape of (len(y_test), len(y_train)) '''
    dist_M = dist_M[:, exemplar_idx]
    y_train = y_train[exemplar_idx]
    classes = np.unique(y_train)
    idx_by_class = {c: np.where(y_train==c)[0] for c in classes}
    correct = 0

    sim_M = dist2sim(dist_M) if not sim else dist_M
    for y, sims in zip(y_test, sim_M):
        max_sim = -np.inf
        for c in classes:
            sim = sims[idx_by_class[c]].sum()
            if sim > max_sim:
                max_sim = sim
                y_hat = c
        if y_hat == y: 
            correct += 1

    return correct/len(y_test)

def eval_CV(M, pairs, y_train, y_test, weight, sim=False):
    ''' contrastive voter: takes in pairs as teaching examples, 
        votes within each pair and takes majorit vote
        M is either a similarity matrix or a distance matrix
     '''
    num_classes = len(np.unique(y_train))
    y_pred = [contrastive_vote(pairs, row, y_train, num_classes, weight, sim) for row in M]
          
    assert(len(y_pred)==len(y_test))

    return (np.array(y_pred)==np.array(y_test)).sum()/len(y_test)#, y_pred

def contrastive_vote(pairs, row, y_train, num_class, weight, sim=False):
    """ assuming labels are 0,1,2,... """

    votes = [0] * num_class
    for pair in pairs:
        cand = pair[np.argmax(row[pair])] if sim else pair[np.argmin(row[pair])]
        vote = y_train[cand]
        if weight == 'sim':
            w = row[cand] 
            w = w if sim else dist2sim(w)
        elif weight == 'abs':
            w = abs(row[pair[0]] -  row[pair[1]])
        elif weight == None:
            w = 1
        votes[vote] += w

    if votes.count(max(votes)) > 1 and not weight: # break ties randomly
        y_hat = random.choice([0,1])
    else:
        y_hat = np.argmax(votes)
    
    return y_hat

### experiment functions
def rand_idx(X, m):
    return np.random.choice(np.arange(len(X)), m, replace=False)

def random_1NN(m_range, dist_M, y_train, y_test, sim=True, n_trials=1000):
    return [np.array([eval_KNN(dist_M, rand_idx(y_train, m), y_train, y_test, sim=sim) for _ in range(n_trials)]).mean() for m in m_range]

def random_exemplar(m_range, dist_M, y_train, y_test, sim=True, n_trials=1000):
    return [np.array([eval_exemplar(dist_M, rand_idx(y_train, m), y_train, y_test, sim=sim) for _ in range(n_trials)]).mean() for m in m_range]

def random_CV(m_range, dist_M, y_train, y_test, weight='abs', sim=True, n_trials=1000):
    idx_by_class = {c: np.where(y_train==c)[0] for c in np.unique(y_train)}
    paired_idx = np.array([[i,j] for i in idx_by_class[0] for j in idx_by_class[1]])
    return [np.array([eval_CV(dist_M, paired_idx[rand_idx(paired_idx, m)], y_train, y_test, weight, sim=sim) for _ in range(n_trials)]).mean() for m in m_range]

def full_1NN(m_range, dist_M, y_train, y_test, sim=True):
    return [eval_KNN(dist_M, np.arange(len(y_train)), y_train, y_test, sim=sim)] * len(m_range)

def full_exemplar(m_range, dist_M, y_train, y_test, sim=True):
    return [eval_exemplar(dist_M, np.arange(len(y_train)), y_train, y_test, sim=sim)] * len(m_range)

def full_CV(m_range, dist_M, y_train, y_test, weight='abs', sim=True):
    idx_by_class = {c: np.where(y_train==c)[0] for c in np.unique(y_train)}
    paired_idx = [[i,j] for i in idx_by_class[0] for j in idx_by_class[1]]
    return [eval_CV(dist_M, np.array(paired_idx), y_train, y_test, weight, sim=sim)] * len(m_range)



def concat_embeds(embeds, labels):
    """ concats an embedding by class into len (n/2)^2
        returns the concated embed and the paired indices
    """
    classes = np.unique(labels)
    idx_by_class = {c: np.where(labels==c)[0] for c in classes}
    concat_embeds = []
    idx = []
    for i in idx_by_class[0]:
        for j in idx_by_class[1]:
            concat_embeds.append(np.hstack([embeds[i],embeds[j]]))
            idx.append([i,j])
    idx = np.array(idx)
    concat_embeds = np.array(concat_embeds)
    return concat_embeds, idx

def diff_embeds(embeds, labels, dir=0):
    """ diff is the opposite of concat
        diffs an embedding by class into len (n/2)^2
        returns the diffed embed and the paired indices
    """
    embeds = np.array(embeds)
    classes = np.unique(labels)
    idx_by_class = {c: np.where(labels==c)[0] for c in classes}
    diff_embeds = []
    idx = []

    if dir:
        for i in idx_by_class[0]:
            for j in idx_by_class[1]:
                diff_embeds.append([embeds[i] - embeds[j]])
                idx.append([i,j])
    else:
        for i in idx_by_class[1]:
            for j in idx_by_class[0]:
                diff_embeds.append([embeds[i] - embeds[j]])
                idx.append([i,j])

    idx = np.array(idx)
    diff_embeds = np.array(diff_embeds).squeeze()
    return diff_embeds, idx

def embed2dist_M(z_train, z_test): return euclidean_distances(z_test, z_train)






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


# def get_lpips_1nn_score(x_train, y_train, x_valid, y_test, dist_MM):
#     if dist_MM.shape != (len(y_test), len(y_train)):
#         dist_MM = dist_MM.T
#     assert(dist_MM.shape==(len(y_test), len(y_train)))

#     total = len(y_test)
#     correct = 0
#     for x, y, dists in zip(x_valid, y_test, dist_MM):
#         nn_idx = np.argmin(dist_MM)
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
