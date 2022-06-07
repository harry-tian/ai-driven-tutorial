""" 
    Builds a synthetic dataset for experiments

    Assumes a data folder "datasets/<DATASET>" that contains:
        a "raw_imgs" subdir 
        a .npz file used to generate the rwa images

    produces:
        - "df.csv"
        - a "data" subdir that splits the images into classes
        - slits the data intro train val test subsets/subdirs
        - "<SPLIT>_features.pkl"
"""
import os, pickle,  shutil, pathlib, sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/net/scratch/tianh/explain_teach')
import utils.utils as utils

DATASET_SIZE = 2000
data = "wv_4blobs"
npz = "4blobs.npz"



data_dir = f"../datasets/{data}"
pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
features = ["head size","body size","tail size","texture"]

## df.csv
X = np.load(f"{data_dir}/{npz}")["X"]
y = np.load(f"{data_dir}/{npz}")["y"]

df = pd.DataFrame(X, columns=features)
df.insert(0, "label", y)

img_id = []
for i in range(DATASET_SIZE):
    if i < 10:
        img = f"00{i}.png"
    elif i >= 10 and i < 100:
        img = f"0{i}.png"
    else:
        img = f"{i}.png"
    img_id.append(img)
df.insert(0, "img_id", img_id)

df.to_csv(os.path.join(data_dir,"df.csv"),index=False)

## data splitting
raw_data_dir = os.path.join(data_dir, "raw_imgs")
split_dir =  os.path.join(data_dir, "data")
for i in range(DATASET_SIZE):
    img_id = df.iloc[i]["img_id"]
    label = df.iloc[i]["label"]
    src = os.path.join(raw_data_dir,img_id)
    dst = os.path.join(split_dir,str(label))
    pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)

total = np.arange(DATASET_SIZE)
valid = np.random.choice(total, int(DATASET_SIZE*0.2), replace=False)
total = np.setdiff1d(total, valid)
test = np.random.choice(total, int(DATASET_SIZE*0.2), replace=False)
total = np.setdiff1d(total, test)
train = total
train_df = df.iloc[train]
valid_df = df.iloc[valid]
test_df = df.iloc[test]

for split, df in zip(["train","valid","test"],[train_df,valid_df,test_df]):
    for i in range(len(df)):
        img_id = df.iloc[i]["img_id"]
        label = df.iloc[i]["label"]
        src = os.path.join(raw_data_dir,img_id)
        dst = os.path.join(data_dir,split,str(label))
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
        shutil.copy(src,dst)


## <SPLIT>_features.pkl
df = pd.read_csv(os.path.join(data_dir,"df.csv"))

for split in ["train","valid","test"]:
    files = utils.dataset_filenames(os.path.join(data_dir,split))
    files = [x.split("/")[-1] for x in files[:,0]]
    features = np.array([list(df[df["img_id"]==f][["head size","body size","tail size","texture"]].iloc[0]) for f in files])
    pickle.dump(features,open(f"{data_dir}/{split}_features.pkl","wb"))