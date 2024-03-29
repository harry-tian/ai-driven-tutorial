import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

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
color_list = ['k','y','m','g','c','r','b','lime']
marker_list = ['o','s','^','D','x','p','*','8']
linestyle_list = ['solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted']
lw=4

def plot_data(X, Y, title="", legend=None, save=False, save_dir=None):
    ''' plots an embedding '''
    if X.shape[1] != 2: X = tsne2(X)

    classes = np.unique(Y)
    plt.figure(figsize=(8, 6))

    for c in classes:
        c_idx = np.where(Y==c)[0]
        plt.scatter(X[c_idx][:,0], X[c_idx][:,1])

    if legend: plt.legend(legend)
    plt.title(title,fontsize=30)
    if save:
        if not save_dir: save_dir = f"figs/out.pdf"
        plt.savefig(save_dir,format="pdf", bbox_inches="tight")

    return X

def plot_all_embeds(x_train,x_valid,x_test,y_train,y_valid,y_test,
sharexy=True,title=None,subtitles=["all","train", "valid", "test"],legend=None,save=False,save_dir=None):
    ''' plots embedding given a dataset with splits'''
    fig, ax = plt.subplots(2,2, figsize=(8*2, 6*2), sharey=sharexy, sharex=sharexy)
    x_all = tsne2(np.concatenate([x_train,x_valid,x_test]))
    y_all = y_train + y_valid + y_test
    x_train = x_all[np.arange(len(x_train))]
    x_valid = x_all[np.arange(len(x_train),len(x_train)+len(x_valid))]
    x_test = x_all[np.arange(len(x_train)+len(x_valid),len(x_train)+len(x_valid)+len(x_test))]
    
    for i, (x,y) in enumerate(zip([x_all,x_train,x_valid,x_test],[y_all, y_train,y_valid,y_test])):
        classes = np.unique(y)

        for c in classes:
            c_idx = np.where(y==c)[0]
            ax[i//2][i%2].scatter(x[c_idx][:,0], x[c_idx][:,1])

        ax[i//2][i%2].set_title(subtitles[i])
    
    if legend: ax[i//2][i%2].legend(legend)
    if title: fig.suptitle(title, fontsize=30)
    if save:
        if not save_dir: save_dir = f"{title}.pdf"
        plt.savefig(save_dir, format="pdf", bbox_inches="tight")

    return x_train, x_valid, x_test


def plot_data_multiplot(all_data, legend, sharey=True, title=None, subtitles=None, save=False, save_dir=None):
    n = len(all_data)
    fig, ax = plt.subplots(1, n, figsize=(8*n, 6), sharey=sharey)
    x_all = [x[0] for x in all_data]
    y_all = [y[0] for y in all_data]

    for k, (x,y) in enumerate(zip(x_all, y_all)):
        # x, y = data

        classes = np.unique(y)

        for c in classes:
            c_idx = np.where(y==c)[0]
            ax[k].scatter(x[c_idx][:,0], x[c_idx][:,1])

        if subtitles: ax[k].set_title(subtitles[k])

    if legend: ax[n-1].legend(legend)
    if title: fig.suptitle(title, fontsize=30)
    if save:
        if not save_dir: save_dir = f"figs/out.pdf"
        plt.savefig(save_dir, format="pdf", bbox_inches="tight")

def plot_knn_scores(m_range, scores, title=None, save=False, save_dir=None):
    plt.figure(figsize=(10,6))

    plt.axhline(scores["full"] , c='black', linewidth=2, linestyle="solid", label="full")  

    random_knn_scores, random_knn_ci = scores["random_scores"]
    plt.plot(m_range, random_knn_scores, linewidth=2, linestyle="dashed", label="random score")
    plt.fill_between(m_range, random_knn_scores + random_knn_ci / 2, random_knn_scores - random_knn_ci / 2, alpha=0.5, label="random ci")
    
    for model, score in scores.items():
        if model == "full" or model == "random_scores": continue
        plt.plot(m_range, score,linewidth=4,linestyle="solid",label=model)

    plt.xlim((m_range[0]-1,m_range[-1]+1))
    plt.ylim((0.5, 1.05))
    plt.xlabel("number of examples")
    plt.ylabel("acc")
    plt.legend(loc='upper right', bbox_to_anchor=(1.02, -0.1),fancybox=True, shadow=True, ncol=20, fontsize=15)

    if not title:
        title = "knn_scores"
    plt.title(title,fontsize=30)
    if save:
        if not save_dir: save_dir = f"figs/out.pdf"
        plt.savefig(save_dir, format="pdf", bbox_inches="tight")
    plt.show()

def plot_knn_scores_multiplot(m_range, allall_scores, subtitles=None, title=None, save=False, save_dir=None):
    n = len(allall_scores)
    fig, ax = plt.subplots(1, n, figsize=(8*n, 6), sharey=True)
    for j, all_scores in enumerate(allall_scores):    
        ax[j].axhline(all_scores["full_score"] , c='black', linewidth=2, linestyle="solid", label="full score")  
        random_knn_scores, random_knn_ci = all_scores["random_scores"]
        ax[j].plot(m_range, random_knn_scores, linewidth=2, linestyle="dashed", label="random score")
        ax[j].fill_between(m_range, random_knn_scores + random_knn_ci / 2, random_knn_scores - random_knn_ci / 2, alpha=0.5, label="random ci")
        for model, score in all_scores.items():
            if model == "full_score" or model == "random_scores": continue
            ax[j].plot(m_range, score, linewidth=4, label=model)

        if subtitles: ax[j].set_title(subtitles[j], fontsize=25)

    fig.supxlabel("number of examples", fontsize=25)
    fig.supylabel("acc",x=0.09, fontsize=25)
    plt.legend(loc='upper right', bbox_to_anchor=(1.02, -0.12),fancybox=True, shadow=True, ncol=7, fontsize=20)

    if title: fig.suptitle(title, fontsize=30)
    if save:
        if not save_dir: save_dir = f"figs/{title}.pdf"
        plt.savefig(save_dir, format="pdf", bbox_inches="tight")

def plot_clf_human(results, clf, human, xlabel, ylabel, legend, title=None, save=False, save_dir=None):
    plt.figure(figsize=(10, 6))

    for i, result in enumerate(results):
        plt.scatter(result[human],result[clf], s=300, marker=marker_list[i], label=legend[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1, -0.11),fancybox=True, shadow=True, ncol=4)

    if title: plt.title(title,fontsize=30)
    if save:
        if not save_dir: save_dir = f"figs/out.pdf"
        plt.savefig(save_dir, format="pdf", bbox_inches="tight")

def plot_decision_support(x_train, y_train, x_test, y_test, prototype_idx, title, legend, save=False, save_dir=None):
    if x_train.shape[1] != 2 and x_test.shape[1] != 2: 
        temp = np.concatenate([x_test,x_train])
        temp = tsne2(temp)
        x_test = temp[0]
        x_train = temp[1:]

    classes = np.unique(y_train)
    plt.figure(figsize=(8, 6))

    for c in classes:
        c_idx = np.where(y_train==c)[0]
        plt.scatter(x_train[c_idx][:,0], x_train[c_idx][:,1])

    for j, c in enumerate(classes):
        train_proto = x_train[[int(i) for i in prototype_idx if y_train[i] == c]]
        plt.scatter(train_proto[:,0], train_proto[:,1], s=300, c=f"C{str(j)}", marker='^', linewidths=1, edgecolors='k') 

    
    plt.scatter(x_test[0], x_test[1], s=300,  c=f"C{str(y_test)}", marker='o', linewidths=1, edgecolors='k') 

    plt.legend(legend)
    plt.title(title,fontsize=30)
    if save:
        if not save_dir: save_dir = f"figs/out.pdf"
        plt.savefig(save_dir,format="pdf", bbox_inches="tight")

    return x_train

def show_decision_support(test, examples, test_dir=None, example_dir=None):
    import os
    test = os.path.join(test_dir, test)
    print("\n Test image:")
    plt.imshow(plt.imread(test))
    print("\n examples:")
    for example in examples:
        example = os.path.join(example_dir, example)
        plt.imshow(plt.imread(example))



def tsne2(embeds):
    print("TSNEing")
    return TSNE(n_components=2, learning_rate='auto', init='random', random_state=42).fit_transform(embeds)

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


# def vis_all_data(x_train, y_train, x_valid, y_valid, title, legend, prototype_idx=None, save=False, save_dir=None):
#     x_all = np.concatenate((x_train, x_valid))
#     if x_train.shape[1] != 2 and x_valid.shape[1] != 2: 
#         split = len(x_train)
#         x_all = tsne2(x_all)
#         x_train = x_all[:split]
#         x_valid = x_all[split:]

#     x_all = np.concatenate((x_train, x_valid))
#     y_all = np.concatenate((y_train, y_valid))
#     classes = np.unique(y_train)
#     subtitles = ["all", "train", "valid"]
#     fig, ax = plt.subplots(1, len(subtitles), figsize=(24, 6))

#     for i, data in enumerate([(x_all, y_all), (x_train, y_train),(x_valid, y_valid)]):
#         x, y = data
#         for c in classes:
#             c_idx = np.where(y==c)[0]
#             ax[i].scatter(x[c_idx][:,0], x[c_idx][:,1])

#         if prototype_idx:
#             for j, c in enumerate(classes):
#                 train_proto = x_train[[int(i) for i in prototype_idx if y_train[i] == c]]
#                 ax[i].scatter(train_proto[:,0], train_proto[:,1], s=300, c=f"C{str(j)}", marker='^', linewidths=1, edgecolors='k') 

#         ax[i].set_title(subtitles[i], fontsize=MEDIUM_SIZE)
#     ax[2].legend(legend)
    
#     normalize_xylim(ax)
#     fig.suptitle(title,fontsize=30)
#     if save:
#         if not save_dir: save_dir = f"figs/out.pdf"
#         plt.savefig(save_dir, format="pdf", bbox_inches="tight")

#     return x_train, x_valid