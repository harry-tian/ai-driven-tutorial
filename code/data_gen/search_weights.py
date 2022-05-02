from http.client import ImproperConnectionState
import evals.embed_evals as evals
import torchvision
import pickle, random
import pandas as pd
import numpy as np
from tqdm import tqdm

train_embeds = pickle.load(open("/net/scratch/tianh-shared/wv/pseudo_label/auto_split/train.pkl", "rb"))
valid_embeds = pickle.load(open("/net/scratch/tianh-shared/wv/pseudo_label/auto_split/valid.pkl", "rb"))
test_embeds = pickle.load(open("/net/scratch/tianh-shared/wv/pseudo_label/auto_split/test.pkl", "rb"))
dataset = torchvision.datasets.ImageFolder("/net/scratch/tianh-shared/wv/pseudo_label/auto_split/train")
syn_y_train = [x[1] for x in dataset]
dataset = torchvision.datasets.ImageFolder("/net/scratch/tianh-shared/wv/pseudo_label/auto_split/test")
syn_y_test = [x[1] for x in dataset]


p_cands = [-3, -2, -1, 1, 2, 3]
powers = []
scores = []
weights = []
for _ in tqdm(range(10000)):
    p = [random.choice(p_cands) for _ in range(4)]
    w = [random.randint(-1000,1000) for _ in range(4)]
    score =  evals.distorted_1nn(train_embeds, syn_y_train, test_embeds, syn_y_test, w, p)
    # if np.around(score, 4) != 0.4839: 
    scores.append(score)
    weights.append(w)
    powers.append(p)

d = {'powers': powers, 'weights': weights, "scores":scores}
df = pd.DataFrame(data=d)
df.to_csv("temp1.csv")