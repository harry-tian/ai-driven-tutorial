import os, pickle
import numpy as np
import torch
import heapq
from collections import defaultdict
from copy import deepcopy

skey = lambda k: tuple(sorted(k))

def get_topk(scores, visits, topk, type="count", confidence=0.01, verbose=False):
    topk = np.min([len(scores), topk])
    if type == "count":
        keys = heapq.nlargest(topk, scores, key=scores.__getitem__)
        weights = [scores[k] for k in keys]
    elif type == "acc":
        acc = {b: scores[b] / (visits[b] + 1e-10) for b in scores}
        keys = heapq.nlargest(topk, acc, key=acc.__getitem__)
        weights = [acc[k] for k in keys]
    elif type == "lcb":
        t = sum(visits.values())
        acc = {b: scores[b] / (visits[b] + 1e-10) for b in scores}
        lcb = {b: acc[b] - confidence * np.sqrt(t / visits[b] if visits[b] > 0 else 0) for b in scores}
        if verbose: 
            print({b: lcb[b] for b in heapq.nlargest(3, lcb, key=lcb.__getitem__)})
        keys = heapq.nlargest(topk, lcb, key=lcb.__getitem__)
        weights = [lcb[k] for k in keys] 
    elif type == "ucb":
        t = sum(visits.values())
        acc = {b: scores[b] / (visits[b] + 1e-10) for b in scores}
        ucb = {b: acc[b] + confidence * np.sqrt(t / visits[b] if visits[b] > 0 else 0) for b in scores}
        if verbose: 
            print({b: ucb[b] for b in heapq.nlargest(3, ucb, key=ucb.__getitem__)})
        keys = heapq.nlargest(topk, ucb, key=ucb.__getitem__)
        weights = [ucb[k] for k in keys] 
    if verbose: 
        print(keys[0], weights[0], topk)
    return keys, weights


def get_candidates_with_label(scores, visits, labels):
    pos = np.where(labels > 0)[0]
    neg = np.where(labels == 0)[0]
    candidates = [skey([p, n]) for p in pos for n in neg]
    candidate_scores = {c: scores[c] for c in candidates}
    candidate_visits = {c: visits[c] for c in candidates}
    return candidate_scores, candidate_visits


def select(embeds, m, triplets, labels=None, topk=10, verbose=False):
    if m < 3: 
        raise Exception("Cannot select less than 2 examples!")
    
    z = torch.tensor(embeds)
    dist = torch.cdist(z, z).numpy()
    uni = np.unique(triplets)
    n = len(uni)

    scores = defaultdict(lambda: 0)
    visits = defaultdict(lambda: 0)

    ### using distance matrix with shape (n,n)
    if len(triplets[0]) > 3:
        uni = np.arange(len(triplets))
        n = len(uni)
        for i in uni:
            for j in range(i):
                correct = (dist[i] <= dist[j]) & (triplets[i] <= triplets[j])
                scores[(j, i)] += correct.sum() - 2
                visits[(j, i)] += n - 2
    ### using triplets with shape (T, 3)
    else:
        for t in triplets:
            a, p, n = t
            key = skey([p, n])
            if dist[a, p] <= dist[a, n]:      
                scores[key] += 1
            visits[key] += 1

    curr_scores = deepcopy(scores)
    curr_visits = deepcopy(visits)
    if labels is not None:
        cand_scores, cand_visits = get_candidates_with_label(scores, visits, labels)
        beam, w = get_topk(cand_scores, cand_visits, topk, type="count", verbose=verbose)
    else:
        beam, w = get_topk(curr_scores, curr_visits, topk, type="count", verbose=verbose)
    for _ in range(3, m+1):
        new_scores = defaultdict(lambda: 0)
        new_visits = deepcopy(new_scores)
        for b in beam:
            for k in uni:
                if k not in b:
                    key = skey(b + (k,))
                    base_score = new_scores[key]
                    base_visit = new_visits[key]
                    if base_score == 0 and curr_scores[b] > 0:
                        new_scores[key] += curr_scores[b]
                    if base_visit == 0 and curr_visits[b] > 0:
                        new_visits[key] += curr_visits[b]
                    for j in b:
                        new_scores[key] += scores[skey([j, k])]
                        new_visits[key] += visits[skey([j, k])]
        curr_scores = new_scores
        curr_visits = new_visits
        beam, w = get_topk(curr_scores, curr_visits, topk, type="count", verbose=verbose)
    return beam[0], w[0]


def select2(embeds, m, triplets, labels=None, topk=10, verbose=False):
    if m < 3: 
        raise Exception("Cannot select less than 2 examples!")
    
    z = torch.tensor(embeds)
    dist = torch.cdist(z, z).numpy()
    uni = np.unique(triplets)
    n = len(uni)

    scores = defaultdict(lambda: 0)
    visits = defaultdict(lambda: 0)

    ### using distance matrix with shape (n,n)
    if len(triplets[0]) > 3:
        uni = np.arange(len(triplets))
        n = len(uni)
        for i in uni:
            for j in range(i):
                correct = (dist[i] <= dist[j]) & (triplets[i] <= triplets[j])
                scores[(j, i)] += correct.sum() - 2
                visits[(j, i)] += n - 2
    ### using triplets with shape (T, 3)
    else:
        for t in triplets:
            a, p, n = t
            key = skey([p, n])
            if dist[a, p] <= dist[a, n]:      
                scores[key] += 1
            visits[key] += 1

    curr_scores = deepcopy(scores)
    curr_visits = deepcopy(visits)
    beam = list(scores.keys())
    
    for _ in range(3, m+1):
        new_scores = defaultdict(lambda: 0)
        new_visits = deepcopy(new_scores)
        for b in beam:
            for k in uni:
                if k not in b:
                    key = skey(b + (k,))
                    base_score = new_scores[key]
                    base_visit = new_visits[key]
                    if base_score == 0 and curr_scores[b] > 0:
                        new_scores[key] += curr_scores[b]
                    if base_visit == 0 and curr_visits[b] > 0:
                        new_visits[key] += curr_visits[b]
                    for j in b:
                        new_scores[key] += scores[skey([j, k])]
                        new_visits[key] += visits[skey([j, k])]
        curr_scores = new_scores
        curr_visits = new_visits
        beam, w = get_topk(curr_scores, curr_visits, topk, type="count", verbose=verbose)
    return beam[0], w[0]