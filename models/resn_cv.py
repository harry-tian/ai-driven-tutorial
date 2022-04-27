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
import trainer

import warnings
warnings.filterwarnings("ignore")

from models.RESN import RESN


class RESN_cv(RESN):
    def __init__(self, split, **config_kwargs):
        super().__init__(**config_kwargs)
        self.train_idx, self.valid_idx = split

    def train_dataloader(self):
        dataset = trainer.sample_dataset(self.train_dataset, self.train_idx)
        print(f"\n train:{len(dataset)}")
        return trainer.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        dataset = trainer.sample_dataset(self.valid_dataset, self.valid_idx)
        print(f"\n valid:{len(dataset)}")
        return trainer.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)

    @staticmethod
    def add_model_specific_args(parser):
        parser = RESN.add_model_specific_args()
        parser.add_argument("--splits", default=None, type=str, required=True)
        parser.add_argument("--split_idx", default=0, type=int, required=False)
        return parser
    
def main():
    parser = trainer.add_generic_args()
    RESN_cv.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    
    import pickle
    splits = pickle.load(open(dict_args["splits"],"rb"))

    model = RESN_cv(splits[dict_args["split_idx"]], **dict_args)
    trainer = trainer.generic_train(model, args, "valid_loss")

if __name__ == "__main__":
    main()
