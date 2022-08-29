import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import torchvision

import models.transforms as transforms
import algorithms.teaching_algs as algs
from evals.teaching_evals import *

def get_ci(samples, confidence=0.95):  return stats.sem(samples) * stats.t.ppf((1 + confidence) / 2., len(samples)-1)

## getting data
def get_lpips_data(dataset, seeds, sim=True):    
    transform = transforms.bird_transform()
    train = torchvision.datasets.ImageFolder(f"data/datasets/{dataset}/train", transform=transform)
    test = torchvision.datasets.ImageFolder(f"data/datasets/{dataset}/test", transform=transform)
    y_train = np.array([x[1] for x in train])
    y_test = np.array([x[1] for x in test])

    dist_M = pickle.load(open(f"data/dist/lpips/{dataset}/train_test.pkl","rb"))
    dist_M = dist2sim(dist_M) if sim else dist_M
    zs = [pickle.load(open(f"data/embeds/bm_lpips/TN_train_d50_seed{seed}.pkl","rb")) for seed in seeds]

    return dist_M, zs, y_train, y_test

def get_prolific_data(dataset, seeds, sim=True):    
    transform = transforms.bird_transform()
    train = torchvision.datasets.ImageFolder(f"data/datasets/{dataset}/train", transform=transform)
    test = torchvision.datasets.ImageFolder(f"data/datasets/{dataset}/test", transform=transform)
    y_train = np.array([x[1] for x in train])
    y_test = np.array([x[1] for x in test])

    dist_Ms = [pickle.load(open(f"data/dist/prolific/{dataset}/train_test_seed{seed}.pkl","rb")) for seed in seeds]
    dist_Ms = [dist2sim(m) if sim else m for m in dist_Ms]
    zs = [pickle.load(open(f"data/embeds/{dataset}_prolific/TN_train_d50_seed{seed}.pkl","rb")) for seed in seeds]

    return dist_Ms, zs, y_train, y_test

def random_experiments(m_range, dist_M, y_train, y_test, sim=True):
    results = {
        '1NN': random_1NN(m_range, dist_M, y_train, y_test, sim=sim),
        '1NN_double': random_1NN(m_range*2, dist_M, y_train, y_test, sim=sim),
        'exemplar': random_exemplar(m_range, dist_M, y_train, y_test, sim=sim),
        'CV': random_CV(m_range, dist_M, y_train, y_test, weight=None, sim=sim),
        'CV_w=dist': random_CV(m_range, dist_M, y_train, y_test, weight='sim', sim=sim),
        'CV_w=ddiff': random_CV(m_range, dist_M, y_train, y_test, weight='abs', sim=sim),
    }
    return results

def full_experiments(m_range, dist_M, y_train, y_test, sim=True):
    results = {
        '1NN': full_1NN(m_range, dist_M, y_train, y_test, sim=sim),
        '1NN_double': full_1NN(m_range*2, dist_M, y_train, y_test, sim=sim),
        'exemplar': full_exemplar(m_range, dist_M, y_train, y_test, sim=sim),
        'CV': full_CV(m_range, dist_M, y_train, y_test, weight=None, sim=sim),
        'CV_w=dist': full_CV(m_range, dist_M, y_train, y_test, weight='sim', sim=sim),
        'CV_w=ddiff': full_CV(m_range, dist_M, y_train, y_test, weight='abs', sim=sim),
    }
    return results

def teaching_experiments(m_range, dist_M, alg, paired_z, z, y_train, y_test, sim=True):
    nn, nn_double, exemplar, CV, CV_w, CV_abs = np.zeros(len(m_range)), np.zeros(len(m_range)), np.zeros(len(m_range)), np.zeros(len(m_range)), np.zeros(len(m_range)), np.zeros(len(m_range))

    paired_z, idx = paired_z
    for j, m in enumerate(m_range):
        S_concat = alg(paired_z, m)
        S_pairs = idx[S_concat]
        S_single = alg(z, m)
        S_double = alg(z, m*2)
        nn[j] = eval_KNN(dist_M, S_single, y_train, y_test, sim=sim)
        nn_double[j] = eval_KNN(dist_M, S_double, y_train, y_test, sim=sim)
        exemplar[j] = eval_exemplar(dist_M, S_single, y_train, y_test, sim=sim)
        CV[j] = eval_CV(dist_M, S_pairs, y_train, y_test, weight=None, sim=sim)
        CV_w[j] = eval_CV(dist_M, S_pairs, y_train, y_test, weight='sim', sim=sim)
        CV_abs[j] = eval_CV(dist_M, S_pairs, y_train, y_test, weight='abs', sim=sim)
    
    results = {
        '1NN': nn,
        '1NN_double': nn_double,
        'exemplar': exemplar,
        'CV': CV,
        'CV_w=dist': CV_w,
        'CV_w=ddiff': CV_abs,
    }
    return results

if True:
    sns.set_theme()
    sns.set_color_codes("bright")
    LINE_WIDTH = 2.5
    MARKER_SIZE = 8
    plot_configs = {     "1NN":                     {'c':"c", 'lw':LINE_WIDTH, 'ls':"solid", 'marker':"", 'ms':MARKER_SIZE, }, 
                        "1NN_double":           {'c':"b", 'lw':LINE_WIDTH, 'ls':"solid", 'marker':"", 'ms':MARKER_SIZE, },
                        "exemplar":             {'c':"y", 'lw':LINE_WIDTH, 'ls':"dashdot", 'marker':"o", 'ms':MARKER_SIZE, },
                        "CV":                   {'c':"r", 'lw':LINE_WIDTH, 'ls':"solid", 'marker':"s", 'ms':MARKER_SIZE, },
                        "CV_w=dist":            {'c':"g", 'lw':LINE_WIDTH, 'ls':"dotted", 'marker':"^", 'ms':MARKER_SIZE, },
                        "CV_w=ddiff":           {'c':"m", 'lw':LINE_WIDTH, 'ls':"solid", 'marker':"x", 'ms':MARKER_SIZE, },
                        }

def plot_random(results, m_range, title, ste=None):
    plt.figure(figsize=(14,8))
    for key in results.keys():
        config = plot_configs[key]
        plt.errorbar(m_range, results[key], yerr=ste, alpha=0.7, **config)

    plt.xticks(m_range)
    plt.xlabel("m")
    bot, top = plt.ylim()
    plt.yticks(np.arange(bot,top, (top-bot)/10))
    plt.legend(results.keys(), loc="lower center", ncol=len(results))
    plt.title(title)
    if SAVE_FIG:
        plt.savefig(f"{title}.pdf")
    else:
        plt.show()

def plot_full(results, m_range, title):
    plt.figure(figsize=(14,8))
    plt.bar(list(results.keys()), [v[0] for v in results.values()], width=0.5)
    bot, top = plt.ylim()
    plt.yticks(np.arange(bot,top,0.05))
    plt.title(title)
    if SAVE_FIG:
        plt.savefig(f"{title}.pdf")
    else:
        plt.show()

def plot_teaching(results, m_range, title, ste=None, idx=None):
    plt.figure(figsize=(14,8))
    for key in results.keys():
        config = plot_configs[key]
        result = results[key] if not idx else results[key][idx]
        plt.plot(m_range, result,alpha=0.8, **config)
        if ste is not None:
            yerr = ste[key][idx]/2 if idx else ste[key]/2
            plt.errorbar(m_range, result, yerr=yerr, alpha=0.5,elinewidth=LINE_WIDTH/2, **config)

    plt.xticks(m_range)
    plt.xlabel("m")
    bot, top = plt.ylim()
    plt.yticks(np.arange(bot,top, (top-bot)/10))
    plt.legend(results.keys(), loc="lower center", ncol=len(results))
    plt.title(title)
    if SAVE_FIG:
        plt.savefig(f"{title}.pdf")
    else:
        plt.show()

def fixed_learner_experiments(m_range, dist_M, y_train, y_test, paired_zs, zs, prefix, sim):
    full_acc = full_1NN(m_range, dist_M, y_train, y_test, sim=sim)
    random_acc = random_1NN(m_range*2, dist_M, y_train, y_test, sim=sim)
    teaching_results = np.array([[eval_KNN(dist_M, alg(z, m*2), y_train, y_test, sim=sim) for m in m_range] for z in zs])
    teaching_acc = teaching_results.mean(axis=0)
    teaching_ste = [get_ci(col) for col in teaching_results.transpose()]
    plot_full_rand_teach(full_acc, random_acc, teaching_acc, m_range, alg_name, f"{prefix}_1NN_double", teaching_ste)

    full_acc = full_exemplar(m_range, dist_M, y_train, y_test, sim=sim)
    random_acc = random_exemplar(m_range*2, dist_M, y_train, y_test, sim=sim)
    teaching_results = np.array([[eval_exemplar(dist_M, alg(z, m*2), y_train, y_test, sim=sim) for m in m_range] for z in zs])
    teaching_acc = teaching_results.mean(axis=0)
    teaching_ste = [get_ci(col) for col in teaching_results.transpose()]
    plot_full_rand_teach(full_acc, random_acc, teaching_acc, m_range, alg_name, f"{prefix}_exemplar", teaching_ste)

    full_acc = full_CV(m_range, dist_M, y_train, y_test, weight=None, sim=sim)
    random_acc = random_CV(m_range, dist_M, y_train, y_test, weight=None, sim=sim)
    teaching_results = np.array([[eval_CV(dist_M, paired_z[1][alg(paired_z[0], m)], y_train, y_test, weight=None, sim=sim) for m in m_range] for paired_z in paired_zs])
    teaching_acc = teaching_results.mean(axis=0)
    teaching_ste = [get_ci(col) for col in teaching_results.transpose()]
    plot_full_rand_teach(full_acc, random_acc, teaching_acc, m_range, alg_name, f"{prefix}_CV", teaching_ste)

    full_acc = full_CV(m_range, dist_M, y_train, y_test, weight='sim', sim=sim)
    random_acc = random_CV(m_range, dist_M, y_train, y_test, weight='sim', sim=sim)
    teaching_results = np.array([[eval_CV(dist_M, paired_z[1][alg(paired_z[0], m)], y_train, y_test, weight='sim', sim=sim) for m in m_range] for paired_z in paired_zs])
    teaching_acc = teaching_results.mean(axis=0)
    teaching_ste = [get_ci(col) for col in teaching_results.transpose()]
    plot_full_rand_teach(full_acc, random_acc, teaching_acc, m_range, alg_name, f"{prefix}_CV_w=dist", teaching_ste)

    full_acc = full_CV(m_range, dist_M, y_train, y_test, weight='abs', sim=sim)
    random_acc = random_CV(m_range, dist_M, y_train, y_test, weight='abs', sim=sim)
    teaching_results = np.array([[eval_CV(dist_M, paired_z[1][alg(paired_z[0], m)], y_train, y_test, weight='abs', sim=sim) for m in m_range] for paired_z in paired_zs])
    teaching_acc = teaching_results.mean(axis=0)
    teaching_ste = [get_ci(col) for col in teaching_results.transpose()]
    plot_full_rand_teach(full_acc, random_acc, teaching_acc, m_range, alg_name, f"{prefix}_CV_w=ddiff", teaching_ste)

def plot_full_rand_teach(full, random, teach, m_range, alg, title, ste=None):
    configs = {     "full":          {'color': "k", 'lw': 2, 'ls':"dashed", 'marker': ""}, 
                    "random":        {'color': "y", 'lw': 2, 'ls':"dashed", 'marker': ""},
                    "teach":        {'color': "r", 'lw': 4, 'ls':"solid", 'marker': "^", 'ms':10},
    }

    sns.set_color_codes("bright")
    plt.figure(figsize=(14,8))
    plt.plot(m_range, full, alpha=0.8, **configs['full'])
    plt.plot(m_range, random, alpha=0.8, **configs['random'])
    plt.plot(m_range, teach, alpha=0.8, **configs['teach'])
    if ste is not None:
        yerr = ste
        plt.errorbar(m_range, teach, yerr=yerr, alpha=0.5, elinewidth=configs['teach']['lw']/2, **configs['teach'])

    p_val = np.round(stats.ttest_ind(teach, random), 3)

    plt.xticks(m_range)
    plt.xlabel("m")
    plt.legend(['full', 'random', alg], loc="lower center", ncol=len(configs))
    plt.title(f"{title}: p_val={p_val}")
    if SAVE_FIG:
        plt.savefig(f"{title}.pdf")
    else:
        plt.show()


SAVE_FIG = True

FIX_LEARNER = True
FIX_TEACHER = True
BM_LPIPS = True
BM_PROLIFIC = True



m_range = np.arange(1, 40)
alg = algs.mmd_greedy
alg_name = "mmd_greedy"

## fixed teacher
if FIX_TEACHER:
    ## BM_lpips
    if BM_LPIPS:
        dataset = "bm"
        seeds = np.arange(10)
        SIM = True
        dist_M, zs, y_train, y_test = get_lpips_data(dataset, seeds, sim=SIM)

        full_results = full_experiments(m_range, dist_M, y_train, y_test, sim=SIM)
        full_acc = full_results
        plot_full(full_acc, m_range, f"{dataset}.lpips_full")

        random_results = random_experiments(m_range, dist_M, y_train, y_test, sim=SIM)
        random_acc = random_results
        plot_random(random_acc, m_range, f"{dataset}.lpips_random")

        paired_zs = [concat_embeds(z, y_train) for z in zs]
        teaching_results = [teaching_experiments(m_range, dist_M, alg, paired_z, z, y_train, y_test, sim=SIM) for paired_z, z in zip(paired_zs, zs)]
        teaching_acc = {key: np.mean(np.array([result[key] for result in teaching_results]), axis=0) for key in teaching_results[0].keys()}
        teaching_ste = {key: get_ci(np.array([result[key] for result in teaching_results])) for key in teaching_results[0].keys()}
        plot_teaching(teaching_acc, m_range, f"{dataset}.lpips_mmd", ste=teaching_ste)


    ## BM_prolific
    if BM_PROLIFIC:
        dataset = "bm"
        seeds = np.arange(10)
        SIM = True
        dist_Ms, zs, y_train, y_test = get_prolific_data(dataset, seeds, sim=SIM)

        full_results = [full_experiments(m_range, dist_M, y_train, y_test, sim=SIM) for dist_M in dist_Ms]
        full_acc = {key: np.mean(np.array([result[key] for result in full_results]), axis=0) for key in full_results[0].keys()}
        plot_full(full_acc, m_range, f"{dataset}.prolific_full")

        random_results = [random_experiments(m_range, dist_M, y_train, y_test, sim=SIM) for dist_M in dist_Ms]
        random_acc = {key: np.mean(np.array([result[key] for result in random_results]), axis=0) for key in random_results[0].keys()}
        plot_random(random_acc, m_range, f"{dataset}.prolific_random")

        paired_zs = [concat_embeds(z, y_train) for z in zs]
        teaching_results = [teaching_experiments(m_range, dist_M, alg, paired_z, z, y_train, y_test, sim=SIM) for paired_z, z, dist_M in zip(paired_zs, zs, dist_Ms)]
        teaching_acc = {key: np.mean(np.array([result[key] for result in teaching_results]), axis=0) for key in teaching_results[0].keys()}
        teaching_ste = {key: get_ci(np.array([result[key] for result in teaching_results])) for key in teaching_results[0].keys()}
        plot_teaching(teaching_acc, m_range, f"{dataset}.prolific_mmd", ste=teaching_ste)

## fixed learner:
if FIX_LEARNER:
    ## BM_LPIPS
    if BM_LPIPS:
        dataset = "bm"
        seeds = np.arange(10)
        SIM = True
        dist_M, zs, y_train, y_test = get_lpips_data(dataset, seeds, sim=SIM)
        paired_zs = [concat_embeds(z, y_train) for z in zs]

        fixed_learner_experiments(m_range, dist_M, y_train, y_test, paired_zs, zs, "bm.lpips", sim=SIM)

    # if BM_PROLIFIC:
    #     dataset = "bm"
    #     seeds = np.arange(10)
    #     SIM = True
    #     dist_Ms, zs, y_train, y_test = get_prolific_data(dataset, seeds, sim=SIM)
    #     paired_zs = [concat_embeds(z, y_train) for z in zs]
