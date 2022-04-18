# -*- coding: utf-8 -*-
from dataclasses import replace
from email.headerregistry import UniqueSingleAddressHeader
import os, pickle
import argparse

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import warnings
from torchvision import  models
warnings.filterwarnings("ignore")

from RESN import RESN
import utils
from torch import nn

class TN(RESN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = models.resnet18(pretrained=False)
        self.triplet_loss = nn.TripletMarginLoss()
        self.train_embeds = None
        self.summarize()

    def train_triplets_step(self, triplet_idx):
        self.train_embeds = self(self.train_input)
        x1, x2, x3 = self.train_embeds[triplet_idx[:,0]], self.train_embeds[triplet_idx[:,1]], self.train_embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        return self.triplet_loss_acc(triplets)

    def mixed_triplets_step(self, triplet_idx, input):
        if self.train_embeds is None: self.train_embeds = self(self.train_input)
        
        embeds = self(input)
        x1, x2, x3 = embeds[triplet_idx[:,0]], self.train_embeds[triplet_idx[:,1]], self.train_embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        return self.triplet_loss_acc(triplets)

    def triplet_loss_acc(self, triplets):
        triplet_loss = self.triplet_loss(*triplets)
        d_ap = self.pdist(triplets[0], triplets[1])
        d_an = self.pdist(triplets[0], triplets[2])
        triplet_acc = (d_ap < d_an).float().mean()
        return triplet_loss, triplet_acc

    def training_step(self, batch, batch_idx):
        triplet_idx = batch[0]
        triplet_loss, triplet_acc = self.train_triplets_step(triplet_idx)

        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        return triplet_loss

    def validation_step(self, batch, batch_idx):
        input = self.valid_input
        triplet_idx = batch[0]

        triplet_loss, triplet_acc = self.mixed_triplets_step(triplet_idx, input)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        input = self.test_input
        triplet_idx = batch[0]
        triplet_loss, triplet_acc = self.mixed_triplets_step(triplet_idx, input)

        self.log('test_triplet_acc', triplet_acc, sync_dist=True)
        self.log('test_triplet_loss', triplet_loss, sync_dist=True)

        self.test_evals()

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

    @staticmethod
    def add_model_specific_args(parser):
        parser = RESN.add_model_specific_args(parser)
        return parser

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


    
    # def get_loss_acc(self, triplet_idx, input):
    #     uniques = np.unique(triplet_idx.cpu().detach().numpy().flatten())
    #     val2idx = {val:i for i,val in enumerate(uniques)}
    #     for i, triplet in enumerate(triplet_idx):
    #         for j, val in enumerate(triplet):
    #             triplet_idx[i][j] = val2idx[int(val)]
    #     triplet_idx = triplet_idx.long()

    #     embeds = self(input[uniques])
    #     x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
    #     triplets = (x1, x2, x3)

    #     triplet_loss = self.triplet_loss(*triplets)
    #     with torch.no_grad():
    #         d_ap = self.pdist(triplets[0], triplets[1])
    #         d_an = self.pdist(triplets[0], triplets[2])
    #         triplet_acc = (d_ap < d_an).float().mean()

    #     return triplet_loss, triplet_acc