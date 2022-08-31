from utils import *
import sys
import os
import pathlib
sys.path.insert(0,'..')
from evals.teaching_evals import *
import algorithms.teaching_algs as algs

def random_experiments(m_range, dist_Ms, y_train, y_test, sim=True):
    def experiment(dist_M):
        return {
            '1NN_CONCAT': random_1NN(m_range, dist_M, y_train, y_test, sim=sim, class_balance=True),
            '1NN': random_1NN(m_range*2, dist_M, y_train, y_test, sim=sim, class_balance=False),
            'exemplar_CONCAT': random_exemplar(m_range, dist_M, y_train, y_test, sim=sim, class_balance=True),
            'exemplar': random_exemplar(m_range*2, dist_M, y_train, y_test, sim=sim, class_balance=False),
        }
    results = [experiment(dist_M) for dist_M in dist_Ms]
    acc = {key: np.mean(np.array([result[key] for result in results]), axis=0) for key in results[0].keys()}
    return acc

def teaching_experiments(m_range, dist_Ms, alg, paired_zs, zs, y_train, y_test, sim=True):
    class_ratios = []
    def experiment(dist_M, paired_z, z):
        nn_paired, exemplar_paired, nn_single, exemplar_single = np.zeros(len(m_range)), np.zeros(len(m_range)), np.zeros(len(m_range)), np.zeros(len(m_range))
        paired_z, idx = paired_z
        for j, m in enumerate(m_range):
            S_concat = alg(paired_z, m)
            S_pairs = idx[S_concat]
            S_single = alg(z, m*2)
            S_flatten = S_pairs.flatten()
            nn_paired[j] = eval_KNN(dist_M, S_flatten, y_train, y_test, sim=sim)
            exemplar_paired[j] = eval_exemplar(dist_M, S_flatten, y_train, y_test, sim=sim)
            nn_single[j] = eval_KNN(dist_M, S_single, y_train, y_test, sim=sim)
            exemplar_single[j] = eval_exemplar(dist_M, S_single, y_train, y_test, sim=sim)

            classes = y_train[S_single]
            class_ratios.append(classes.sum()/len(classes))

        return {
            '1NN_CONCAT': nn_paired,
            '1NN': nn_single,
            'exemplar_CONCAT': exemplar_paired,
            'exemplar': exemplar_single,
        }
    results = [experiment(dist_M, paired_z, z) for paired_z, z, dist_M in zip(paired_zs, zs, dist_Ms)]
    acc = {key: np.mean(np.array([result[key] for result in results]), axis=0) for key in results[0].keys()}
    ste = {key: get_ci(np.array([result[key] for result in results])) for key in results[0].keys()}
    return acc, ste, class_ratios


SAVE_FIG = True





m_range = np.arange(1, 40)
alg = algs.mmd_greedy
alg_name = "MMD"


dataset = "bm"
dist = 'prolific'
seeds = np.arange(10)
SIM = True
dist_Ms, zs, y_train, y_test = get_prolific_data(dataset, seeds, sim=SIM)

save_dir = os.path.join('figures', os.path.basename(__file__)[:-3])
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)



plot_configs = {     
    "1NN_CONCAT":         {'c':"b", 'ls':"solid", 'marker':"",}, 
    "1NN":                {'c':"c", 'ls':"solid", 'marker':"",},
    "exemplar_CONCAT":    {'c':"orange", 'ls':"dashdot", 'marker':"o",},
    "exemplar":           {'c':"y", 'ls':"solid", 'marker':"s",}
        }

paired_zs = [concat_embeds(z, y_train) for z in zs]
teaching_acc, teaching_ste, _ = teaching_experiments(m_range, dist_Ms, alg, paired_zs, zs, y_train, y_test, sim=SIM)
plot_teaching(teaching_acc, m_range, ste=teaching_ste, title=f"{dataset}.{dist}_{alg_name}", plot_configs=plot_configs, save_fig=SAVE_FIG, save_dir=save_dir)


random_acc = random_experiments(m_range, dist_Ms, y_train, y_test, sim=SIM)
plot_random(random_acc, m_range, title=f"{dataset}.{dist}_random", plot_configs=plot_configs, save_fig=SAVE_FIG, save_dir=save_dir)



