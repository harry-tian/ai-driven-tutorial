# -*- coding: utf-8 -*-
from email.policy import default
import os, pickle
import argparse
from torch import nn

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import warnings
from torchvision import  models
warnings.filterwarnings("ignore")
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from TN import TN
import trainer
from pytorch_lightning.trainer.supporters import CombinedLoader
import sys
sys.path.insert(0, '..')
import evals.embed_evals as evals

class MTL(TN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

    def train_triplets_step(self, triplet_idx, labels):
        uniques = np.unique(triplet_idx.cpu().detach().numpy().flatten())
        if len(uniques) < len(self.train_input):
            val2idx = {val:i for i,val in enumerate(uniques)}
            for i, triplet in enumerate(triplet_idx):
                for j, val in enumerate(triplet):
                    triplet_idx[i][j] = val2idx[int(val)]
            triplet_idx = triplet_idx.long()
            input = self.train_input[uniques]
        else: input = self.train_input
        # print(input.shape)
        embeds = self(input)
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)

        triplet_loss, triplet_acc = self.triplet_loss_acc(triplets)
        clf_loss, m = self.clf_loss_acc(self.train_embeds, labels)
        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss
        return clf_loss, m, triplet_loss, triplet_acc, total_loss

    def mixed_triplets_step(self, triplet_idx, input, labels):
        self.train_embeds = self(self.train_input)
        
        embeds = self(input)
        x1, x2, x3 = embeds[triplet_idx[:,0]], self.train_embeds[triplet_idx[:,1]], self.train_embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)

        triplet_loss, triplet_acc = self.triplet_loss_acc(triplets)
        clf_loss, m = self.clf_loss_acc(embeds, labels)
        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss
        return clf_loss, m, triplet_loss, triplet_acc, total_loss

    def training_step(self, batch, batch_idx):
        labels = self.train_label
        triplet_idx = batch[0]
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.train_triplets_step(triplet_idx, labels)

        self.log('train_clf_loss', clf_loss)
        self.log('train_clf_acc', m['acc'], prog_bar=False)
        self.log('train_triplet_loss', triplet_loss)
        self.log('train_triplet_acc', triplet_acc, prog_bar=False)
        self.log('train_total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        input = self.valid_input
        labels = self.valid_label
        triplet_idx = batch[0]
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.mixed_triplets_step(triplet_idx, input, labels)

        self.log('valid_clf_loss', clf_loss)
        self.log('valid_clf_acc', m['acc'], prog_bar=False)
        self.log('valid_auc', m['auc'], prog_bar=False)
        self.log('valid_triplet_loss', triplet_loss)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=False)
        self.log('valid_total_loss', total_loss)

    def test_step(self, batch, batch_idx):
        input = self.test_input
        labels = self.test_label
        triplet_idx = batch[0]
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.mixed_triplets_step(triplet_idx, input, labels)

        self.log('test_clf_loss', clf_loss)
        self.log('test_clf_acc', m['acc'], prog_bar=False)
        self.log('test_auc', m['auc'], prog_bar=False)
        self.log('test_triplet_loss', triplet_loss)
        self.log('test_triplet_acc', triplet_acc, prog_bar=False)
        self.log('test_total_loss', total_loss)
        
        knn_acc, ds_acc = self.test_evals()

        df = pd.read_csv("results.csv")
        df = pd.concat([df, pd.DataFrame({"wandb_group": [self.hparams.wandb_group], "wandb_name": [self.hparams.wandb_name],
            "test_clf_acc": [m['acc'].item()], "test_clf_loss": [clf_loss.item()], "test_1nn_acc": [knn_acc], "test_triplet_acc":[triplet_acc.item()], "decision_support": [ds_acc]})], sort=False)
        df.to_csv("results.csv", index=False)

    @staticmethod
    def add_model_specific_args(parser):
        parser = TN.add_model_specific_args(parser)
        parser.add_argument("--check_val_every_n_epoch", default = 1, type=int)
        parser.add_argument("--early_stop_patience", default = 10, type=int)
        return parser

def main():
    parser = trainer.config_parser()
    config_files = parser.parse_args()
    configs = trainer.load_configs(config_files)
    print(configs)

    pl.seed_everything(configs["seed"])
    model = MTL(**configs)
    monitor = "valid_total_loss"
    trainer.generic_train(model, configs, monitor)

if __name__ == "__main__":
    main()
