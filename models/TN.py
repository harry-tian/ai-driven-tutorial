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

class TN(RESN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = models.resnet18(pretrained=False)
        self.triplet_loss = nn.TripletMarginLoss()
        self.pdist = nn.PairwiseDistance()

        self.setup_data()
        self.summarize()

    def forward(self, triplet_idx, batch_idx):
        if self.trainer.training:
            input = self.train_input
        else:
            input = self.valid_input

        embeds = self.embed(input)
            
        triplet_idx = triplet_idx.long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        return triplets
    
    def get_loss_acc(self, triplet_idx, batch_idx):
        triplets = self(triplet_idx, batch_idx)
        
        triplet_loss = self.triplet_loss(*triplets)
        with torch.no_grad():
            d_ap = self.pdist(triplets[0], triplets[1])
            d_an = self.pdist(triplets[0], triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()

        return triplet_loss, triplet_acc

    def training_step(self, batch, batch_idx):
        triplet_loss, triplet_acc = self.get_loss_acc(batch[0], batch_idx)

        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        return triplet_loss

    def validation_step(self, batch, batch_idx):
        triplet_loss, triplet_acc = self.get_loss_acc(batch[0], batch_idx)

        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        triplet_loss, triplet_acc = self.get_loss_acc(batch[0], batch_idx)

        self.log('test_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('test_triplet_loss', triplet_loss, sync_dist=True)

    def setup_data(self):
        train_transform = utils.get_transform(self.hparams.transform, aug=True)
        valid_transform = utils.get_transform(self.hparams.transform, aug=False)
        train_dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=train_transform)
        valid_dataset = torchvision.datasets.ImageFolder(self.hparams.valid_dir, transform=valid_transform)
        test_dataset = torchvision.datasets.ImageFolder(self.hparams.test_dir, transform=valid_transform)
        self.train_input = torch.tensor(np.array([data[0].numpy() for data in train_dataset])).cuda()
        self.valid_input = torch.tensor(np.array([data[0].numpy() for data in valid_dataset])).cuda()
        self.test_input = torch.tensor(np.array([data[0].numpy() for data in test_dataset])).cuda()
        self.train_label = torch.tensor(np.array([data[1] for data in train_dataset])).cuda()
        self.valid_label = torch.tensor(np.array([data[1] for data in valid_dataset])).cuda()
        self.test_label = torch.tensor(np.array([data[1] for data in test_dataset])).cuda()

        self.train_triplets = pickle.load(open(self.hparams.train_triplets, "rb"))
        self.valid_triplets = pickle.load(open(self.hparams.valid_triplets, "rb"))
        # self.test_triplets = pickle.load(open(self.hparams.test_triplets, "rb"))
        self.test_triplets = self.valid_triplets

        if self.hparams.subset:
            subset_idx = np.random.choice(len(self.train_triplets), len(self.train_triplets)//20, replace=False)
            self.train_triplets = self.train_triplets[subset_idx]
    
        self.train_dataset = torch.utils.data.TensorDataset(torch.tensor(self.train_triplets))
        self.valid_dataset = torch.utils.data.TensorDataset(torch.tensor(self.valid_triplets))
        self.test_dataset = torch.utils.data.TensorDataset(torch.tensor(self.test_triplets))

    def train_dataloader(self):
        dataset = self.train_dataset
        print(f"\nlen_train:{len(dataset)}")
        return utils.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        dataset = self.valid_dataset
        print(f"\nlen_valid:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)

    def test_dataloader(self):
        dataset = self.test_dataset
        print(f"\nlen_test:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "test", self.hparams.dataloader_num_workers)

    @staticmethod
    def add_model_specific_args(parser):   
        parser = RESN.add_model_specific_args(parser)   

        parser.add_argument("--train_triplets", default=None, type=str, required=True)
        parser.add_argument("--valid_triplets", default=None, type=str, required=True) 
        parser.add_argument("--subset", action="store_true")
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
