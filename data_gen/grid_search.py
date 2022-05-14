
import torchvision,sys, itertools
import pickle, random
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '..')
import evals.embed_evals as evals

train_embeds = pickle.load(open("../datasets/wv_3d/train_features.pkl","rb"))
test_embeds = pickle.load(open("../datasets/wv_3d/test_features.pkl","rb"))
y_train = np.array([x[1] for x in torchvision.datasets.ImageFolder("../datasets/wv_3d/train")])
y_test = np.array([x[1] for x in torchvision.datasets.ImageFolder("../datasets/wv_3d/test")])

total_weights = list(itertools.product([0]+[10**i for i in range(7)],repeat=4))

# total_weights = list(itertools.product(np.arange(8),repeat=4))




split = int(sys.argv[1])
start = split*512
end = (split+1)*512
w_list = total_weights[start:end]





aligns = []
for weights in tqdm(w_list): aligns.append(evals.distorted_1nn(train_embeds, y_train, test_embeds, y_test, weights))





df = pd.read_csv("grid.csv")

d = {'weights': w_list, "aligns":aligns}
df = pd.concat([df,pd.DataFrame(data=d)])
df.to_csv("grid.csv",index=False)