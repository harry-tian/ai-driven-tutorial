
from torchvision import transforms
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

######## cross validation ####################

def cross_val_multiclass(idxs, k=10):
    splits_by_class = []
    for idx in idxs:
        splits_by_class.append(gen_cross_val(idx, k=k))

    splits = np.copy(splits_by_class[0])  
    for data, split in zip(idxs[1:],splits_by_class[1:]):
        for i in range(k):
            for j in range(3):
                splits[i,j] = np.concatenate((splits[i,j], data[split[i,j]]))

    for split in splits:
        temp = np.concatenate((split[0],split[1],split[2]))
        assert(len(np.unique(temp))==np.concatenate(idxs).shape[0])

    return splits

def gen_cross_val(indexes, k=10):
    splits = []
    kf = KFold(n_splits=k, shuffle=True)
    for i, (trainval, test) in enumerate(kf.split(indexes)):
        valid = np.random.choice(trainval, len(test),replace=False)
        train = np.setdiff1d(trainval, valid)
        splits.append((train, valid, test))
    return np.array(splits)

def gen_split(src_dir, dst_dir, split):
    instances = dataset_filenames(src_dir)
    cp_split(dst_dir, split, instances)

def dataset_filenames(directory):
    def find_classes(directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

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

def add_generic_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--dataloader_num_workers", default=4, type=int)
    parser.add_argument("--num_class", default=2, type=int)

    parser.add_argument("--train_dir", default=None, type=str, required=True)
    parser.add_argument("--valid_dir", default=None, type=str, required=False)
    parser.add_argument("--test_dir", default=None, type=str, required=False)

    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_mode", default="offline", type=str)
    parser.add_argument("--wandb_project", default="?", type=str)
    parser.add_argument("--wandb_entity", default="ai-driven-tutorial", type=str)
    parser.add_argument("--wandb_name", default=None, type=str)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    return parser

def generic_train(model, args, monitor, 
                    early_stopping_callback=False, extra_callbacks=[], checkpoint_callback=None, logging_callback=None,  **extra_train_kwargs):
    output_dir = os.path.join("results", model.hparams.wandb_project)
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    wandb_name = f"{time.strftime('%m/%d_%H:%M')}" if not args.wandb_name else args.wandb_name
    experiment = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        mode=args.wandb_mode, 
        group=args.wandb_group,
        name=wandb_name)

    logger = WandbLogger(project="imagenet_bm", experiment=experiment)

    ckpt_path = os.path.join(output_dir, logger.version, "checkpoints")
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{valid_loss:.2f}", monitor=monitor, mode="min", save_last=True, save_top_k=3, verbose=True)

    train_params = {}
    train_params["max_epochs"] = args.max_epochs
    if args.gpus == -1 or args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        auto_select_gpus=True,
        weights_summary=None,
        callbacks=extra_callbacks + [checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=1,
        **train_params)

    if args.do_train:
        trainer.fit(model)
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)

    if args.do_test:
        trainer.test(model)

    return trainer

def get_dataloader(dataset, batch_size, split, num_workers):
    drop_last = True if split == "train" else False
    shuffle = True if split == "train" else False
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

def get_acc(probs, target, threshold=0.5, multiclass=False, verbose=False): 
    if multiclass:
        pred = torch.argmax(probs, dim=1)
    else:
        pred = (probs >= threshold).long()

    correct = (pred == target).sum().item()
    total = len(target)

    return correct/total

######### transforms ############################

def get_transform(dataset, aug=True):
    if dataset == "bm":
        return bm_transform_aug() if aug else bm_transform()
    elif dataset == "xray":
        return xray_transform_aug() if aug else xray_transform()
    elif dataset == "bird":
        return bird_transform_aug() if aug else bird_transform()
    elif dataset == "inception":
        return inception_transform_aug() if aug else inception_transform()

def food_transform():
    transform = transforms.Compose([
        transforms.Resize([230,230]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def bm_transform_aug(hparams=None):
    affine = {}
    affine["degrees"] = 30 #hparams.rotate
    # if hparams.translate > 0: 
    #     translate = hparams.translate
    #     affine["translate"] = (translate, translate)
    if 0.2 > 0: 
        scale = 0.2
        affine["scale"] = (1 - scale, 1 + scale)
    # if hparams.shear > 0:
    #     shear = hparams.shear
    #     affine["shear"] = (-shear, shear, -shear, shear)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(0),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomAffine(**affine)
    ])
    return transform

def bm_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def xray_transform_aug():
    return transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def xray_transform():
    return transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def bird_transform_aug():
    return transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(224),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def bird_transform():
    return transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def inception_transform_aug():
    return transforms.Compose([
        transforms.Resize((299,299)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def inception_transform():
    return transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


######## misc ##################################

def dataset_with_indices(cls):

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def get_bm_datasets(train_dir="/net/scratch/hanliu-shared/data/bm/train", 
                    valid_dir="/net/scratch/hanliu-shared/data/bm/valid",
                    train_idx=None, valid_idx=None):
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=bm_transform())
    valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=bm_transform())
    train_inputs = torch.tensor(np.array([data[0].numpy() for data in train_dataset]))
    valid_inputs = torch.tensor(np.array([data[0].numpy() for data in valid_dataset]))
    train_labels = torch.tensor(np.array([data[1] for data in train_dataset]))
    valid_labels = torch.tensor(np.array([data[1] for data in valid_dataset]))
    
    if train_idx: train_inputs = train_inputs[train_idx]
    if valid_idx: valid_inputs = valid_inputs[valid_idx]
    return train_inputs, valid_inputs

