import numpy as np
import evals.embed_evals as evals
import torchvision
import pickle

train_embeds = pickle.load(open("/net/scratch/tianh-shared/wv-3d/pseudo_label/auto_split/train.pkl", "rb"))
valid_embeds = pickle.load(open("/net/scratch/tianh-shared/wv-3d/pseudo_label/auto_split/valid.pkl", "rb"))
test_embeds = pickle.load(open("/net/scratch/tianh-shared/wv-3d/pseudo_label/auto_split/test.pkl", "rb"))
dataset = torchvision.datasets.ImageFolder("/net/scratch/tianh-shared/wv-3d/pseudo_label/auto_split/train")
syn_y_train = [x[1] for x in dataset]
dataset = torchvision.datasets.ImageFolder("/net/scratch/tianh-shared/wv-3d/pseudo_label/auto_split/test")
syn_y_test = [x[1] for x in dataset]


powers=2
scores = []
weights = []
w3=1
for w1 in np.arange(-10000, 10, 10):
    for w2 in np.arange(-10000, 10, 10):
        for w4 in np.arange(-10000, 10, 10):
            w = [w1,w2,w3,w4]
            weights.append(w)
            scores.append(evals.distorted_1nn(train_embeds, syn_y_train, test_embeds, syn_y_test, w, powers))

pickle.dump((weights, scores), open("scores.pkl","wb"))