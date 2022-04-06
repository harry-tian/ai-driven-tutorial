# -*- coding: utf-8 -*-
from dataclasses import replace
import os, pickle
import argparse

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import warnings
warnings.filterwarnings("ignore")

from TN_base import TripletNet, generic_train
import utils

class TN_bm(TripletNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_data()

    def forward(self, triplet_idx, batch_idx):
        if self.trainer.training:
            input = self.train_input
        else:
            input = self.valid_input

        embeds = self.embed(input)
        if self.trainer.testing and self.hparams.do_embed and batch_idx==0:
            if not self.hparams.embed_path:
                embed_path = f"embeds/{self.hparams.wandb_project}.pkl"
            else:
                embed_path = self.hparams.embed_path
            pickle.dump(embeds.cpu(), open(embed_path,"wb"))
            print(f"\n dumped embeds to {embed_path}")
            
        triplet_idx = triplet_idx.long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        return triplets
    
    def setup_data(self):
        train_dir = "/net/scratch/hanliu-shared/data/bm/train"
        valid_dir = "/net/scratch/hanliu-shared/data/bm/valid"
        # all_dir = "/net/scratch/tianh/bm/all"
        # train_triplets = "/net/scratch/tianh/bm/triplets/train_triplets.pkl"
        # valid_triplets = "/net/scratch/tianh/bm/triplets/valid_triplets.pkl"

        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=utils.bm_transform())
        valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=utils.bm_transform())
        self.train_input = torch.tensor(np.array([data[0].numpy() for data in train_dataset])).cuda()
        self.valid_input = torch.tensor(np.array([data[0].numpy() for data in valid_dataset])).cuda()

        self.train_triplets = pickle.load(open(self.hparams.train_triplets, "rb"))
        self.valid_triplets = pickle.load(open(self.hparams.valid_triplets, "rb"))
        self.test_triplets = self.valid_triplets

        if self.hparams.subset:
            subset_idx = np.random.choice(len(self.train_triplets), len(self.train_triplets)//20, replace=False)
            self.train_triplets = self.train_triplets[subset_idx]
        
    
        self.train_dataset = torch.utils.data.TensorDataset(torch.tensor(self.train_triplets))
        self.valid_dataset = torch.utils.data.TensorDataset(torch.tensor(self.valid_triplets))
        self.test_dataset = torch.utils.data.TensorDataset(torch.tensor(self.test_triplets))

    @staticmethod
    def add_model_specific_args(parser):   
        parser.add_argument("--train_triplets", default=None, type=str, required=True)
        parser.add_argument("--valid_triplets", default=None, type=str, required=True)     
        return parser

def main():
    parser = argparse.ArgumentParser()
    TripletNet.add_generic_args(parser)
    parser = TN_bm.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    model = TN_bm(**dict_args)
    trainer = generic_train(model, args)

if __name__ == "__main__":
    main()
