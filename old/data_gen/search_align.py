
import torchvision,sys
import pickle, random
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '..')
import evals.embed_evals as evals

train_embeds = pickle.load(open("../data/datasets/wv_3d/train_features.pkl","rb"))
test_embeds = pickle.load(open("../data/datasets/wv_3d/test_features.pkl","rb"))
y_train = np.array([x[1] for x in torchvision.datasets.ImageFolder("../data/datasets/wv_3d/train")])
y_test = np.array([x[1] for x in torchvision.datasets.ImageFolder("../data/datasets/wv_3d/test")])



max = 0.000003
min = 0
num = 100



w_list = []
aligns = []
for w in tqdm(np.linspace(min,max,num)):
    weights = [w,0,1,1]
    w_list.append(w)
    aligns.append(evals.distorted_1nn(train_embeds, y_train, test_embeds, y_test, weights))







d = {'weights': w_list, "aligns":aligns}
df = pd.DataFrame(data=d)
df.to_csv("search.csv")