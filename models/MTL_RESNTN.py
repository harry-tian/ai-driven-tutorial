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

import sys
sys.path.insert(0, '..')
import evals.embed_evals as evals


class MTL_RESNTN(TN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.feature_extractor = models.resnet18(pretrained=True)

    def get_loss_acc(self, batch, input, labels):
        triplet_idx = batch[0]

        embeds = self(input)        
        logits = self.classifier(embeds)
        
        triplet_idx = triplet_idx.long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)

        probs = torch.sigmoid(logits)

        clf_loss = self.criterion(logits, labels.type_as(logits).unsqueeze(1))
        triplet_loss = self.triplet_loss(*triplets)
        with torch.no_grad():
            m = utils.metrics(probs, labels.unsqueeze(1))
            d_ap = self.pdist(triplets[0], triplets[1])
            d_an = self.pdist(triplets[0], triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()

        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss

        return clf_loss, m, triplet_loss, triplet_acc, total_loss

    def training_step(self, batch, batch_idx):
        input = self.train_input
        labels = self.train_label
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(batch, input, labels)

        self.log('train_clf_loss', clf_loss, sync_dist=True)
        self.log('train_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_total_loss', total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        input = self.valid_input
        labels = self.valid_label
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(batch, input, labels)

        self.log('valid_clf_loss', clf_loss, sync_dist=True)
        self.log('valid_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_total_loss', total_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        input = self.test_input
        labels = self.test_label
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(batch, input, labels)

        self.log('test_clf_loss', clf_loss, sync_dist=True)
        self.log('test_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('test_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('test_triplet_loss', triplet_loss, sync_dist=True)
        self.log('test_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('test_total_loss', total_loss, sync_dist=True)
        
        self.test_evals()

    @staticmethod
    def add_model_specific_args(parser):
        parser = TN.add_model_specific_args(parser)
        parser.add_argument("--lamda", default=0.5, type=float)
        parser.add_argument("--check_val_every_n_epoch", default = 1, type=int)
        parser.add_argument("--early_stop_patience", default = 10, type=int)
        return parser

def main():
    parser = utils.add_generic_args()
    MTL_RESNTN.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    model = MTL_RESNTN(**dict_args)

    monitor = "valid_total_loss"
    print(args.early_stop_patience, args.check_val_every_n_epoch)
    early_stop_callback = EarlyStopping(monitor="valid_total_loss", min_delta=0.00, patience=args.early_stop_patience, verbose=True, mode="min")
    # trainer = Trainer(callbacks=[early_stop_callback])
    trainer = utils.generic_train(model, args, monitor, callbacks = [early_stop_callback], check_val_every_n_epoch = args.check_val_every_n_epoch)

if __name__ == "__main__":
    main()