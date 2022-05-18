
import shutil
import os
import numpy as np
from sklearn.model_selection import KFold
import shutil, pathlib

def files_in_dir(mypath): return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

def dataset_filenames(directory):
    ''' returns a list of filenames with their class labels, in the order of torch.dataset'''
    class_to_idx = find_classes(directory)
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if os.path.isfile(path):
                    item = path, target_class
                    instances.append(item)

    return np.array(instances)

def auto_split(src_dir, dst_dir):
    ''' 
        generates train, valid and test splits given a dataset path;
        the dataset should be divided by class
    '''
    instances = dataset_filenames(src_dir)
    classes = find_classes(src_dir).keys()
    train_dir = os.path.join(dst_dir, "train")
    valid_dir = os.path.join(dst_dir, "valid")
    test_dir = os.path.join(dst_dir, "test")
    pathlib.Path(train_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(valid_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True)

    for c in classes:
        if c == "auto_split": continue
        c_idx = np.where(instances[:,1] == c)[0]
        split = len(c_idx)//10
        c_test = np.random.choice(c_idx, split*2, replace=False)
        c_idx = np.setdiff1d(c_idx,c_test)
        c_valid = np.random.choice(c_idx, split*2, replace=False)
        c_idx = np.setdiff1d(c_idx,c_valid)
        c_train = c_idx
        c_train_dir = os.path.join(train_dir, c)
        pathlib.Path(c_train_dir).mkdir(parents=True, exist_ok=True)
        for f in instances[c_train,0]: shutil.copy(f,c_train_dir)
        c_valid_dir = os.path.join(valid_dir, c)
        pathlib.Path(c_valid_dir).mkdir(parents=True, exist_ok=True)
        for f in instances[c_valid,0]: shutil.copy(f,c_valid_dir)
        c_test_dir = os.path.join(test_dir, c)
        pathlib.Path(c_test_dir).mkdir(parents=True, exist_ok=True)
        for f in instances[c_test,0]: shutil.copy(f,c_test_dir)

 
     
######## not really used anymore ####################   

def cross_val_multiclass(idxs, k=10):
    splits_by_class = [gen_cross_val(idx, k=k) for idx in idxs]

    splits = []
    for i in range(k-1):
        splits.append([])
        for j in range(3):
            split_i = np.concatenate([split[i][j] for split in splits_by_class])
            splits[i].append(split_i)
        splits[i] = np.array(splits[i])

    for split in splits:
        temp = np.concatenate([split[0],split[1],split[2]])
        assert(np.equal(np.sort(np.unique(temp)),np.concatenate(idxs)).all())

    return np.array(splits)

def gen_cross_val(indexes, k=10):
    splits = []
    test = np.random.choice(indexes, len(indexes)//10,replace=False)
    indexes = np.setdiff1d(indexes, test)
    kf = KFold(n_splits=k-1, shuffle=True)
    for train, valid in kf.split(indexes):
        splits.append(np.array([indexes[train], indexes[valid], test]))
    return np.array(splits)

def gen_split(src_dir, dst_dir, split):
    instances = dataset_filenames(src_dir)
    cp_split(dst_dir, split, instances)

def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return class_to_idx
    
def cp_split(dst_dir, split, instances):
    train, valid, test = split
    for instance in instances[train]:
        split_dir = os.path.join(dst_dir, "train")
        f_name = instance[0]
        label = instance[1]
        name = f_name.split("/")[-1]
        class_dir = os.path.join(split_dir, label)
        if not os.path.isdir(class_dir): os.mkdir(class_dir)
        dst = os.path.join(class_dir, name)
        shutil.copyfile(f_name, dst)
    for instance in instances[valid]:
        split_dir = os.path.join(dst_dir, "valid")
        f_name = instance[0]
        label = instance[1]
        name = f_name.split("/")[-1]
        class_dir = os.path.join(split_dir, label)
        if not os.path.isdir(class_dir): os.mkdir(class_dir)
        dst = os.path.join(class_dir, name)
        shutil.copyfile(f_name, dst)
    for instance in instances[test]:
        split_dir = os.path.join(dst_dir, "test")
        f_name = instance[0]
        label = instance[1]
        name = f_name.split("/")[-1]
        class_dir = os.path.join(split_dir, label)
        if not os.path.isdir(class_dir): os.mkdir(class_dir)
        dst = os.path.join(class_dir, name)
        shutil.copyfile(f_name, dst)
