from zmq import device
from torchmetrics.functional.classification import auroc, stat_scores, average_precision, precision_recall_curve, auc

import time
import shutil
from pytorch_lightning.loggers import WandbLogger
import wandb
import os, pickle
from pathlib import Path
import pytorch_lightning as pl
import torch, argparse
import numpy as np
from pydoc import locate
import torchvision
from sklearn.model_selection import KFold
import shutil
from os import listdir
from os.path import isfile, join
from omegaconf import OmegaConf as oc

######## cross validation ####################

def auto_split(src_dir, dst_dir):
    instances = dataset_filenames(src_dir)
    classes = find_classes(src_dir).keys()
    train_dir = os.path.join(dst_dir, "train")
    valid_dir = os.path.join(dst_dir, "valid")
    test_dir = os.path.join(dst_dir, "test")
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(valid_dir): os.mkdir(valid_dir)
    if not os.path.isdir(test_dir): os.mkdir(test_dir)

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
        if not os.path.isdir(c_train_dir): os.mkdir(c_train_dir)
        for f in instances[c_train,0]: shutil.copy(f,c_train_dir)
        c_valid_dir = os.path.join(valid_dir, c)
        if not os.path.isdir(c_valid_dir): os.mkdir(c_valid_dir)
        for f in instances[c_valid,0]: shutil.copy(f,c_valid_dir)
        c_test_dir = os.path.join(test_dir, c)
        if not os.path.isdir(c_test_dir): os.mkdir(c_test_dir)
        for f in instances[c_test,0]: shutil.copy(f,c_test_dir)
        

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
    
def dataset_filenames(directory):
    class_to_idx = find_classes(directory)
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if isfile(path):
                    item = path, target_class
                    instances.append(item)

    return np.array(instances)

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

######## training helpers ####################

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default='/net/scratch/tianh/explain_teach/models/configs/base.yaml', type=str, required=False)
    parser.add_argument("--model_config", default=None, type=str, required=True)
    parser.add_argument("--dataset_config", default=None, type=str, required=False)
    parser.add_argument("--triplet_config", default=None, type=str, required=False)
    return parser

def test_configs(configs):
    required_args = ["gpus", "seed", "dataloader_num_workers",
    "max_epochs", "learning_rate", "train_batch_size", "embed_dim",
    "num_class" ,"train_dir", "valid_dir", "test_dir", "transform", 
    "wandb_group", "wandb_mode", "wandb_project", "wandb_entity",  "wandb_name",
    "do_train", "do_test",
    "train_triplets", "valid_triplets", "test_triplets",
    "pretrained", "lamda", "syn"]
    syn_args = ["syn", "train_synthetic", "test_synthetic", "weights", "powers"]
    
    if "syn" in configs:
        if configs["syn"]: 
            for arg in syn_args: assert(arg in configs)
            required_args += syn_args
            
    if set(configs) != set(required_args):
        print("\n Missing args:")
        print(np.setdiff1d(required_args, configs))
        print("Unrecognized args:")
        print(np.setdiff1d(configs, required_args))
        assert(False)

def load_configs(config_files):
    base_config = oc.load(config_files.base_config)
    model_config = oc.load(config_files.model_config)
    dataset_config = oc.load(config_files.dataset_config) if config_files.dataset_config else {}
    triplet_config = oc.load(config_files.triplet_config) if config_files.triplet_config else {}
    configs = oc.merge(base_config, dataset_config,  model_config, triplet_config)
    test_configs(configs)
    return configs

def generic_train(model, args, monitor, 
                    early_stopping_callback=False, extra_callbacks=[], checkpoint_callback=None, logging_callback=None,  **extra_train_kwargs):
    output_dir = os.path.join("results", model.hparams.wandb_project)
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    wandb_name = f"{time.strftime('%m/%d_%H:%M')}" if not args["wandb_name"] else args["wandb_name"]
    experiment = wandb.init(
        entity=args["wandb_entity"],
        project=args["wandb_project"],
        mode=args["wandb_mode"], 
        group=args["wandb_group"],
        name=wandb_name)

    logger = WandbLogger(project="imagenet_bm", experiment=experiment)

    ckpt_path = os.path.join(output_dir, logger.version, "checkpoints")
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{valid_loss:.2f}", monitor=monitor, mode="min", save_last=True, save_top_k=3, verbose=True)

    train_params = {}
    train_params["max_epochs"] = args["max_epochs"]
    if args["gpus"] == -1 or args["gpus"] > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(
        auto_select_gpus=True,
        gpus=args["gpus"],
        weights_summary=None,
        callbacks=extra_callbacks + [checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=1,
        **train_params)

    if args["do_train"]:
        trainer.fit(model)
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)

        ckpts = [f for f in listdir(ckpt_path)]
        for ckpt in ckpts:
            if ckpt != "best_model.ckpt":
                os.remove(os.path.join(ckpt_path, ckpt))

    if args["do_test"]:
        trainer.test(model, ckpt_path='best')

    return trainer

def do_test(model, args, ckpt_path):
    wandb_name = f"{time.strftime('%m/%d_%H:%M')}" if not args["wandb_name"] else args["wandb_name"]
    experiment = wandb.init(
        entity=args["wandb_entity"],
        project=args["wandb_project"],
        mode=args["wandb_mode"], 
        group=args["wandb_group"],
        name=wandb_name)

    logger = WandbLogger(experiment=experiment)

    train_params = {}
    train_params["max_epochs"] = args["max_epochs"]
    if args["gpus"] == -1 or args["gpus"] > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(
        gpus=args["gpus"],
        auto_select_gpus=True, ## true
        weights_summary=None,
        logger=logger,
        check_val_every_n_epoch=1,
        **train_params)
        
    trainer.test(model, ckpt_path=ckpt_path)

    return trainer

def get_dataloader(dataset, batch_size, split, num_workers):
    drop_last = True if split == "train" else False
    shuffle = True if split == "train" else False
    batch_size = min(len(dataset), batch_size)
    # batch_size = min(128,batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        num_workers=num_workers, drop_last=drop_last, shuffle=shuffle)
    return dataloader

def sample_dataset(dataset, index):
    input = torch.tensor(np.array([data[0].numpy() for data in dataset]))[index]
    label = torch.tensor(np.array([data[1] for data in dataset]))[index]
    dataset = torch.utils.data.TensorDataset(input, label)
    return dataset

def metrics(probs, target, threshold=0.5, num_class=2):
    if num_class > 2:
        pred = torch.argmax(probs, dim=1)
    else:
        pred = (probs >= threshold).long()
    target = target.int()

    tp, fp, tn, fn, sup = stat_scores(pred, target, ignore_index=0)
    if 0 < sup < len(target):
        precision, recall, _ = precision_recall_curve(pred, target)
        auprc = auc(recall, precision)
    m = {}
    m['pred'] = pred
    m['acc'] = (tp + tn) / (tp + tn + fp + fn)
    if num_class < 3:
        m['auc'] = auroc(probs, target) if 0 < sup < len(target) else None
        # m['tpr'] = tp / (tp + fn)
        # m['tnr'] = tn / (tn + fp)
        # m['ppv'] = tp / (tp + fp)
        # m['f1'] = 2 * tp / (2 * tp + fp + fn)
        # m['ap'] = average_precision(probs, target)
        # m['auprc'] = auprc if 0 < sup < len(target) else None
    return m   

# def get_acc(probs, target, threshold=0.5, multiclass=False, verbose=False): 
#     if multiclass:
#         pred = torch.argmax(probs, dim=1)
#     else:
#         pred = (probs >= threshold).long()

#     correct = (pred == target).sum().item()
#     total = len(target)

#     return correct/total

######## misc ##################################

def dataset_with_indices(cls):

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


# def add_generic_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gpus", default=1, type=int)
#     parser.add_argument("--seed", default=42, type=int)

#     parser.add_argument("--max_epochs", default=200, type=int)
#     parser.add_argument("--learning_rate", default=1e-4, type=float)
#     parser.add_argument("--train_batch_size", default=16, type=int)
#     parser.add_argument("--valid_batch_size", default=64, type=int)
#     parser.add_argument("--test_batch_size", default=64, type=int)
#     parser.add_argument("--dataloader_num_workers", default=4, type=int)
#     parser.add_argument("--num_class", default=2, type=int)

#     parser.add_argument("--train_dir", default=None, type=str, required=True)
#     parser.add_argument("--valid_dir", default=None, type=str, required=False)
#     parser.add_argument("--test_dir", default=None, type=str, required=False)

#     parser.add_argument("--wandb_group", default=None, type=str)
#     parser.add_argument("--wandb_mode", default="offline", type=str)
#     parser.add_argument("--wandb_project", default="?", type=str)
#     parser.add_argument("--wandb_entity", default="ai-driven-tutorial", type=str)
#     parser.add_argument("--wandb_name", default=None, type=str)

#     parser.add_argument("--do_train", action="store_true")
#     parser.add_argument("--do_test", action="store_true")
#     parser.add_argument("--ckpt_path", default=None, type=str, required=False)
    
#     parser.add_argument("--transform", default="bm", type=str)

#     parser.add_argument("--train_triplets", default=None, type=str, required=False)
#     parser.add_argument("--valid_triplets", default=None, type=str, required=False) 
#     parser.add_argument("--test_triplets", default=None, type=str, required=False) 

#     parser.add_argument("--train_synthetic", default=None, type=str, required=False) 
#     parser.add_argument("--test_synthetic", default=None, type=str, required=False) 

#     parser.add_argument("--syn", action="store_true") 
#     parser.add_argument("--w1", default=1, type=float)
#     parser.add_argument("--w2", default=1, type=float)

#     return parser



# def old_train(model, args, monitor, 
#                     early_stopping_callback=False, extra_callbacks=[], checkpoint_callback=None, logging_callback=None,  **extra_train_kwargs):
#     output_dir = os.path.join("results", model.hparams.wandb_project)
#     odir = Path(output_dir)
#     odir.mkdir(parents=True, exist_ok=True)
#     log_dir = Path(os.path.join(output_dir, 'logs'))
#     log_dir.mkdir(parents=True, exist_ok=True)

#     wandb_name = f"{time.strftime('%m/%d_%H:%M')}" if not args.wandb_name else args.wandb_name
#     experiment = wandb.init(
#         entity=args.wandb_entity,
#         project=args.wandb_project,
#         mode=args.wandb_mode, 
#         group=args.wandb_group,
#         name=wandb_name)

#     logger = WandbLogger(project="imagenet_bm", experiment=experiment)

#     ckpt_path = os.path.join(output_dir, logger.version, "checkpoints")
#     if checkpoint_callback is None:
#         checkpoint_callback = pl.callbacks.ModelCheckpoint(
#             dirpath=ckpt_path, filename="{epoch}-{valid_loss:.2f}", monitor=monitor, mode="min", save_last=True, save_top_k=3, verbose=True)

#     train_params = {}
#     train_params["max_epochs"] = args.max_epochs
#     if args.gpus == -1 or args.gpus > 1:
#         train_params["distributed_backend"] = "ddp"

#     trainer = pl.Trainer.from_argparse_args(
#         args,
#         auto_select_gpus=True, ## true
#         weights_summary=None,
#         callbacks=extra_callbacks + [checkpoint_callback],
#         logger=logger,
#         check_val_every_n_epoch=1,
#         **train_params)

#     if args.do_train:
#         trainer.fit(model)
#         target_path = os.path.join(ckpt_path, 'best_model.ckpt')
#         print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
#         shutil.copy(checkpoint_callback.best_model_path, target_path)

#     if args.do_test:
#         trainer.test(model, ckpt_path='best')

#     return trainer
