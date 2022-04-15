# -*- coding: utf-8 -*-
from dataclasses import replace
import os, pickle
import argparse

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import warnings
from torchvision import  models
warnings.filterwarnings("ignore")

from resn_args import RESN
import utils
from torch import nn

import sys
sys.path.insert(0, '..')
import evals.embed_evals as evals

class TN(RESN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = models.resnet18(pretrained=False)
        self.triplet_loss = nn.TripletMarginLoss()

        self.setup_data()
        self.summarize()
    
    def get_loss_acc(self, triplet_idx, input, knn_acc=False):
        embeds = self(input)

        triplet_idx = triplet_idx.long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        
        triplet_loss = self.triplet_loss(*triplets)
        with torch.no_grad():
            d_ap = self.pdist(triplets[0], triplets[1])
            d_an = self.pdist(triplets[0], triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()

        return triplet_loss, triplet_acc

    def training_step(self, batch, batch_idx):
        input = self.train_input
        triplet_loss, triplet_acc = self.get_loss_acc(batch[0], input, batch_idx)

        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        return triplet_loss

    def validation_step(self, batch, batch_idx):
        input = self.valid_input
        triplet_loss, triplet_acc = self.get_loss_acc(batch[0], input, batch_idx)

        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        input = self.test_input
        triplet_loss, triplet_acc = self.get_loss_acc(batch[0], input, batch_idx)

        self.log('test_triplet_acc', triplet_acc, sync_dist=True)
        self.log('test_triplet_loss', triplet_loss, sync_dist=True)

        train_x = self(self.train_input).cpu().detach().numpy()
        train_y = self.train_label.cpu().detach().numpy()
        test_x = self(self.test_input).cpu().detach().numpy()
        test_y = self.test_label.cpu().detach().numpy()
        knn_acc = evals.get_knn_score(train_x, train_y, test_x, test_y)
        self.log('test_1nn_acc', knn_acc, sync_dist=True)
        
        if self.hparams.syn:
            syn_x_train, syn_y_train = pickle.load(open(self.hparams.train_synthetic,"rb"))
            syn_x_test, syn_y_test = pickle.load(open(self.hparams.test_synthetic,"rb"))
            examples = evals.class_1NN_idx(train_x, train_y, test_x, test_y)
            ds_acc = evals.decision_support(syn_x_train, syn_y_train, syn_x_test, syn_y_test, examples)
            self.log('decision support', ds_acc, sync_dist=True)    

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.train_triplets))
        print(f"\nlen_train:{len(dataset)}")
        return utils.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.valid_triplets))
        print(f"\nlen_valid:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)

    def test_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.test_triplets))
        print(f"\nlen_test:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "test", self.hparams.dataloader_num_workers)

def main():
    parser = utils.add_generic_args()
    TN.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    model = TN(**dict_args)

    monitor = "valid_triplet_loss"
    trainer = utils.generic_train(model, args, monitor)

if __name__ == "__main__":
    main()
