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
import utils
from omegaconf import OmegaConf as oc

import pandas as pd


class MTL(TN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

    def train_triplets_step(self, triplet_idx, labels):
        self.train_embeds = self(self.train_input)
        x1, x2, x3 = self.train_embeds[triplet_idx[:,0]], self.train_embeds[triplet_idx[:,1]], self.train_embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)

        triplet_loss, triplet_acc = self.triplet_loss_acc(triplets)
        clf_loss, m = self.clf_loss_acc(self.train_embeds, labels)
        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss
        return clf_loss, m, triplet_loss, triplet_acc, total_loss

    def mixed_triplets_step(self, triplet_idx, input, labels):
        if self.train_embeds is None: self.train_embeds = self(self.train_input)
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

        self.log('train_clf_loss', clf_loss, sync_dist=True)
        self.log('train_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_total_loss', total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        input = self.valid_input
        labels = self.valid_label
        triplet_idx = batch[0]
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.mixed_triplets_step(triplet_idx, input, labels)

        self.log('valid_clf_loss', clf_loss, sync_dist=True)
        self.log('valid_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_total_loss', total_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        input = self.test_input
        labels = self.test_label
        triplet_idx = batch[0]
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.mixed_triplets_step(triplet_idx, input, labels)

        self.log('test_clf_loss', clf_loss, sync_dist=True)
        self.log('test_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('test_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('test_triplet_loss', triplet_loss, sync_dist=True)
        self.log('test_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('test_total_loss', total_loss, sync_dist=True)
        
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
    parser = utils.config_parser()
    config_files = parser.parse_args()
    configs = utils.load_configs(config_files)

    # wandb_name = "MTL_pretrained" if configs["pretrained"] else "MTL"
    # wandb_name = oc.create({"wandb_name": wandb_name}) 
    # configs = oc.merge(configs, wandb_name)
    print(configs)

    pl.seed_everything(configs["seed"])
    model = MTL(**configs)
    monitor = "valid_total_loss"
    trainer = utils.generic_train(model, configs, monitor)

    # print(configs.early_stop_patience, configs.check_val_every_n_epoch)
    # early_stop_callback = EarlyStopping(monitor="valid_total_loss", min_delta=0.00, patience=args.early_stop_patience, verbose=True, mode="min")
    # trainer = Trainer(callbacks=[early_stop_callback])
    # trainer = utils.generic_train(model, configs, monitor, callbacks = [early_stop_callback], check_val_every_n_epoch = args.check_val_every_n_epoch)

if __name__ == "__main__":
    main()



    # def get_loss_acc(self, batch, input, labels):
    #     triplet_idx = batch[0]

    #     embeds = self(input)        
    #     logits = self.classifier(embeds)
        
    #     triplet_idx = triplet_idx.long()
    #     x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
    #     triplets = (x1, x2, x3)

    #     probs = torch.sigmoid(logits)

    #     clf_loss = self.criterion(logits, labels.type_as(logits).unsqueeze(1))
    #     triplet_loss = self.triplet_loss(*triplets)
    #     with torch.no_grad():
    #         m = utils.metrics(probs, labels.unsqueeze(1))
    #         d_ap = self.pdist(triplets[0], triplets[1])
    #         d_an = self.pdist(triplets[0], triplets[2])
    #         triplet_acc = (d_ap < d_an).float().mean()

    #     total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss

    #     return clf_loss, m, triplet_loss, triplet_acc, total_loss