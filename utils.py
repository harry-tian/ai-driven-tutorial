from itertools import combinations
from copy import deepcopy
import pickle

def distM2triplets(dist_m, out_dir):
    triplets = []
    arange = list(range(len(dist_m)))
    combs = []
    for a in arange:
        temp = deepcopy(arange)
        temp.remove(a)
        for comb in list(combinations(temp, r=2)):
            comb = [a] + list(comb)
            combs.append(comb)
    for comb in combs:
        a, p, n = comb[0], comb[1], comb[2]
        if dist_m[a, p] > dist_m[a, n]:
            triplet = [a, n, p]
        else:
            triplet = [a, p, n]
        triplets.append(triplet)

    pickle.dump(triplets, open(out_dir,"wb"))
        