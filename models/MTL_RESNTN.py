# -*- coding: utf-8 -*-
import os, pickle
import argparse
from torch import nn

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import warnings
warnings.filterwarnings("ignore")

from TN import TN
import utils

class MTL_RESNTN(TN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

    def forward(self, triplet_idx, clf_idx, batch_idx):
        if self.trainer.training:
            inputs = self.train_input
        else:
            inputs = self.valid_input

        embeds = self.embed(inputs)

        if self.hparams.MTL_hparam:
            clf_data = embeds[clf_idx]
        else:
            clf_data = embeds
        
        logits = self.classifier(clf_data)
        
        triplet_idx = triplet_idx.long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        
        return logits, triplets

    def get_loss_acc(self, batch, batch_idx):
        triplet_idx = batch[0]
        clf_idx = torch.unique(torch.flatten(triplet_idx))
        logits, triplets = self(triplet_idx, clf_idx, batch_idx)

        probs = torch.sigmoid(logits)

        if self.trainer.training:
            labels = self.train_label
        else:
            labels = self.valid_label
        if self.hparams.MTL_hparam:
            labels = labels[clf_idx]

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
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(batch, batch_idx)

        self.log('train_clf_loss', clf_loss, sync_dist=True)
        self.log('train_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_total_loss', total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(batch, batch_idx)

        self.log('valid_clf_loss', clf_loss, sync_dist=True)
        self.log('valid_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_total_loss', total_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(batch, batch_idx)

        self.log('test_clf_loss', clf_loss, sync_dist=True)
        self.log('test_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('test_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('test_triplet_loss', triplet_loss, sync_dist=True)
        self.log('test_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('test_total_loss', total_loss, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parser):
        # parser.add_argument("--pretrained", action="store_true")
        # parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
        # parser.add_argument("--transform", default="bm", type=str)    
        
        # parser.add_argument("--train_triplets", default=None, type=str, required=True)
        # parser.add_argument("--valid_triplets", default=None, type=str, required=True) 
        # parser.add_argument("--subset", action="store_true")
        parser = TN.add_model_specific_args(parser)
        parser.add_argument("--MTL_hparam", action="store_true")
        parser.add_argument("--lamda", default=0.5, type=float)
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
    trainer = utils.generic_train(model, args, monitor)

if __name__ == "__main__":
    main()
