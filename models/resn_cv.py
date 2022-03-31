# -*- coding: utf-8 -*-
import os
import time
import argparse
from sklearn import multiclass
import torch
import torchvision
from torch import nn
from torchvision import  models
import pytorch_lightning as pl
import wandb
import utils

import warnings
warnings.filterwarnings("ignore")

from resn_args import RESN


class RESN_cv(RESN):
    def __init__(self, split, **config_kwargs):
        super().__init__(**config_kwargs)
        self.train_idx, self.valid_idx, self.test_idx = split

    def train_dataloader(self):
        transform = utils.get_transform(self.hparams.transform, aug=True) 
        dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=transform)
        dataset = utils.sample_dataset(dataset, self.train_idx)
        print(f"\n train:{len(dataset)}")
        return utils.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        transform = utils.get_transform(self.hparams.transform, aug=False)
        dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=transform)
        dataset = utils.sample_dataset(dataset, self.valid_idx)
        print(f"\n valid:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)

    def test_dataloader(self):
        transform = utils.get_transform(self.hparams.transform, aug=False)
        dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=transform)
        dataset = utils.sample_dataset(dataset, self.test_idx)
        print(f"\n test:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "test", self.hparams.dataloader_num_workers)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
        parser.add_argument("--transform", default="bm", type=str)
        parser.add_argument("--splits", default=None, type=str, required=True)
        parser.add_argument("--split_idx", default=0, type=int, required=False)
        return parser
    
def main():
    parser = utils.add_generic_args()
    RESN_cv.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    
    import pickle
    splits = pickle.load(open(dict_args["splits"],"rb"))

    model = RESN_cv(splits[dict_args["split_idx"]], **dict_args)
    trainer = utils.generic_train(model, args, "valid_loss")

if __name__ == "__main__":
    main()
