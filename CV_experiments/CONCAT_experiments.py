from utils import *
import sys
import os
import pathlib
import pandas as pd

sys.path.insert(0,'..')
from evals.teaching_evals import *
import algorithms.teaching_algs as algs

def full_experiments(m_range, dist_Ms, y_train, y_test, sim=True):
    def experiment(dist_M):
        return {
            '1NN': full_1NN(m_range, dist_M, y_train, y_test, sim=sim),
            'exemplar': full_exemplar(m_range, dist_M, y_train, y_test, sim=sim),
            'CV': full_CV(m_range, dist_M, y_train, y_test, weight=None, sim=sim),
            'CV_w=dist': full_CV(m_range, dist_M, y_train, y_test, weight='sim', sim=sim),
            'CV_w=ddiff': full_CV(m_range, dist_M, y_train, y_test, weight='abs', sim=sim),
        }
    results = [experiment(dist_M) for dist_M in dist_Ms]
    acc = {key: np.mean(np.array([result[key] for result in results]), axis=0) for key in results[0].keys()}
    return acc

def random_experiments(m_range, dist_Ms, y_train, y_test, sim=True):
    def experiment(dist_M):
        return {
            '1NN': random_1NN(m_range, dist_M, y_train, y_test, sim=sim, class_balance=True),
            'exemplar': random_exemplar(m_range, dist_M, y_train, y_test, sim=sim, class_balance=True),
            'CV': random_CV(m_range, dist_M, y_train, y_test, weight=None, sim=sim),
            'CV_w=dist': random_CV(m_range, dist_M, y_train, y_test, weight='sim', sim=sim),
            'CV_w=ddiff': random_CV(m_range, dist_M, y_train, y_test, weight='abs', sim=sim),
        }
    results = [experiment(dist_M) for dist_M in dist_Ms]
    acc = {key: np.mean(np.array([result[key] for result in results]), axis=0) for key in results[0].keys()}
    return acc

def teaching_experiments(m_range, dist_Ms, alg, paired_zs, y_train, y_test, sim=True):
    def experiment(dist_M, paired_z):
        nn, exemplar, CV, CV_w, CV_abs = np.zeros(len(m_range)), np.zeros(len(m_range)), np.zeros(len(m_range)), np.zeros(len(m_range)), np.zeros(len(m_range))
        paired_z, idx = paired_z
        for j, m in enumerate(m_range):
            S_concat = alg(paired_z, m)
            S_pairs = idx[S_concat]
            S_flatten = S_pairs.flatten()
            nn[j] = eval_KNN(dist_M, S_flatten, y_train, y_test, sim=sim)
            exemplar[j] = eval_exemplar(dist_M, S_flatten, y_train, y_test, sim=sim)
            CV[j] = eval_CV(dist_M, S_pairs, y_train, y_test, weight=None, sim=sim)
            CV_w[j] = eval_CV(dist_M, S_pairs, y_train, y_test, weight='sim', sim=sim)
            CV_abs[j] = eval_CV(dist_M, S_pairs, y_train, y_test, weight='abs', sim=sim)

        return {
            '1NN': nn,
            'exemplar': exemplar,
            'CV': CV,
            'CV_w=dist': CV_w,
            'CV_w=ddiff': CV_abs,
        }
    results = [experiment(dist_M, paired_z, z) for paired_z, z, dist_M in zip(paired_zs, dist_Ms)]
    acc = {key: np.mean(np.array([result[key] for result in results]), axis=0) for key in results[0].keys()}
    ste = {key: get_ci(np.array([result[key] for result in results])) for key in results[0].keys()}
    return acc, ste


m_range = np.arange(1, 41)
alg = algs.mmd_greedy
alg_name = "MMD"


dataset = "bm"
dist = 'prolific'
seeds = np.arange(10)
SIM = True
dist_Ms, zs, y_train, y_test = get_prolific_data(dataset, seeds, sim=SIM)


full_acc = full_experiments(m_range, dist_Ms, y_train, y_test, sim=SIM)
random_acc = random_experiments(m_range, dist_Ms, y_train, y_test, sim=SIM)
paired_zs = [concat_embeds(z, y_train) for z in zs]
teaching_acc, teaching_ste = teaching_experiments(m_range, dist_Ms, alg, paired_zs,y_train, y_test, sim=SIM)

data = []
learners = full_acc.keys()
for learner in learners:
    full_row = np.hstack([[learner, 'full'], full_acc[learner]])
    random_row = np.hstack([[learner, 'random'], random_acc[learner]])
    teaching_row = np.hstack([[learner, 'MMD'], teaching_acc[learner]])
    teaching_ste_row = np.hstack([[learner, 'MMD_ste'], teaching_ste[learner]])

    data.extend([full_row, random_row, teaching_row, teaching_ste_row])

columns = np.hstack([['learner', 'algorithm'], m_range])
df = pd.DataFrame(np.array(data), columns=columns)
df.to_csv(f"{dataset}.{dist}_experiments.csv", index=False)
