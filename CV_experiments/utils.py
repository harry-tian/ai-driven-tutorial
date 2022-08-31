import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import torchvision

import sys
import os
sys.path.insert(0,'..')
import models.transforms as transforms
from evals.teaching_evals import *

if True:
    sns.set_theme()
    sns.set_color_codes("bright")
    SMALL_SIZE = 10
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 30
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=15)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    LINE_WIDTH = 2.5
    MARKER_SIZE = 8
    STEP = 0.05
    TITLE_WEIGHT = 'bold'

def plot_full(results, m_range, title, plot_configs, ste=None, save_fig=False, save_dir=None):
    plt.figure(figsize=(14,8))
    plt.bar(list(results.keys()), [v[0] for v in results.values()], width=0.5)
    bot, top = plt.ylim()
    plt.yticks(np.arange(bot,top,0.05))
    plt.title(title, weight=TITLE_WEIGHT)

    if save_fig and save_dir:
        plt.savefig(os.path.join(save_dir,f"{title}.pdf"))
    else:
        plt.show()

def plot_random(results, m_range, title, plot_configs, ste=None, save_fig=False, save_dir=None):
    plt.figure(figsize=(14,8))
    for key in results.keys():
        config = plot_configs[key]
        plt.errorbar(m_range, results[key], yerr=ste, alpha=0.7, lw=LINE_WIDTH,  ms=MARKER_SIZE,  **config)

    plt.xticks(m_range)
    plt.xlabel("m")
    bot, top = plt.ylim()
    plt.yticks(np.arange(bot//STEP*STEP, (top//STEP+1)*STEP, STEP))
    plt.legend(results.keys(), loc="lower center", ncol=len(results))
    plt.title(title, weight=TITLE_WEIGHT)

    if save_fig and save_dir:
        plt.savefig(os.path.join(save_dir,f"{title}.pdf"))
    else:
        plt.show()

def plot_teaching(results, m_range, title,  plot_configs, ste=None, idx=None, save_fig=False, save_dir=None):
    plt.figure(figsize=(14,8))
    m_range = m_range[idx] if idx is not None else m_range
    for key in results.keys():
        config = plot_configs[key]
        result = results[key][idx] if idx is not None else results[key]
        plt.plot(m_range, result,alpha=0.8, lw=LINE_WIDTH, ms=MARKER_SIZE, **config)
        if ste is not None:
            yerr = np.array(ste[key])
            yerr = yerr[idx]/2 if idx is not None else yerr/2
            plt.errorbar(m_range, result, yerr=yerr, alpha=0.5,elinewidth=LINE_WIDTH/2, **config)

    plt.xticks(m_range)
    plt.xlabel("m")
    bot, top = plt.ylim()
    plt.yticks(np.arange(bot//STEP*STEP, (top//STEP+1)*STEP, STEP))
    plt.legend(results.keys(), loc="lower center", ncol=len(results))
    plt.title(title, weight=TITLE_WEIGHT)

    if save_fig and save_dir:
        plt.savefig(os.path.join(save_dir,f"{title}.pdf"))
    else:
        plt.show()

def plot_full_rand_teach(full, random, teach, m_range, alg_name, title, plot_configs, ste=None, save_fig=False, save_dir=None):

    plt.figure(figsize=(14,8))
    plt.plot(m_range, full, alpha=0.8, ms=MARKER_SIZE, **plot_configs['full'])
    plt.plot(m_range, random, alpha=0.8, ms=MARKER_SIZE, **plot_configs['random'])
    plt.plot(m_range, teach, alpha=0.8, ms=MARKER_SIZE, **plot_configs['teach'])
    if ste is not None:
        yerr = np.array(ste)/2
        plt.errorbar(m_range, teach, yerr=yerr, alpha=0.5,elinewidth=LINE_WIDTH/2, **plot_configs['teach'])

    p_val = np.round(stats.ttest_ind(teach, random)[1], 4)

    plt.xticks(m_range)
    plt.xlabel("m")
    plt.legend(['full', 'random', alg_name], loc="lower center", ncol=len(plot_configs))
    plt.title(f"{title}: p_val={p_val}", weight=TITLE_WEIGHT)

    if save_fig and save_dir:
        plt.savefig(os.path.join(save_dir,f"{title}.pdf"))
    else:
        plt.show()

def get_ci(samples, confidence=0.95):  return stats.sem(samples) * stats.t.ppf((1 + confidence) / 2., len(samples)-1)

## getting data
def get_lpips_data(dataset, seeds, sim=True):    
    transform = transforms.bird_transform()
    train = torchvision.datasets.ImageFolder(f"../data/datasets/{dataset}/train", transform=transform)
    test = torchvision.datasets.ImageFolder(f"../data/datasets/{dataset}/test", transform=transform)
    y_train = np.array([x[1] for x in train])
    y_test = np.array([x[1] for x in test])

    dist_M = pickle.load(open(f"../data/dist/lpips/{dataset}/train_test.pkl","rb"))
    dist_M = dist2sim(dist_M) if sim else dist_M
    zs = [pickle.load(open(f"../data/embeds/bm_lpips/TN_train_d50_seed{seed}.pkl","rb")) for seed in seeds]

    return [dist_M]*len(seeds), zs, y_train, y_test

def get_prolific_data(dataset, seeds, sim=True):    
    transform = transforms.bird_transform()
    train = torchvision.datasets.ImageFolder(f"../data/datasets/{dataset}/train", transform=transform)
    test = torchvision.datasets.ImageFolder(f"../data/datasets/{dataset}/test", transform=transform)
    y_train = np.array([x[1] for x in train])
    y_test = np.array([x[1] for x in test])

    dist_Ms = [pickle.load(open(f"../data/dist/prolific/{dataset}/train_test_seed{seed}.pkl","rb")) for seed in seeds]
    dist_Ms = [dist2sim(m) if sim else m for m in dist_Ms]
    zs = [pickle.load(open(f"../data/embeds/{dataset}_prolific/TN_train_d50_seed{seed}.pkl","rb")) for seed in seeds]

    return dist_Ms, zs, y_train, y_test