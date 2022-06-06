import numpy as np


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