# -*- coding: utf-8 -*-
from dataclasses import replace
import os, pickle
import argparse

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
# from torchmetrics.functional.classification import auroc, stat_scores, average_precision, precision_recall_curve, auc
import warnings
warnings.filterwarnings("ignore")

from TN_base import TripletNet, generic_train
import utils

class TN_bm(TripletNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_data()

    def forward(self, triplet_idx, batch_idx):
        if self.hparams.split_by == "triplet":
            dataset = self.dataset
        elif self.hparams.split_by == "img":
            if self.trainer.training:
                dataset = self.train_dataset
            else:
                dataset = self.valid_dataset

        embeds = self.embed(dataset)
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
        train_triplets = "/net/scratch/tianh/bm/triplets/train_triplets.pkl"
        valid_triplets = "/net/scratch/tianh/bm/triplets/valid_triplets.pkl"

        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=utils.bm_transform())
        valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=utils.bm_transform())
        self.train_input = torch.tensor(np.array([data[0].numpy() for data in train_dataset])).cuda()
        self.valid_input = torch.tensor(np.array([data[0].numpy() for data in valid_dataset])).cuda()

        if self.hparams.split_by == "triplet":
            self.dataset = self.train_input
            self.triplets = pickle.load(open(train_triplets, "rb"))

            subset_idx = np.random.choice(len(self.triplets), len(self.triplets)//10, replace=False)
            self.triplets = self.triplets[subset_idx]

            len_triplets = np.arange(0,self.triplets.shape[0])
            self.train_idx = np.random.choice(len_triplets, int(len(len_triplets)*0.8), replace=False)
            self.valid_idx = np.setdiff1d(len_triplets, self.train_idx)
            self.train_triplets = self.triplets[self.train_idx]
            self.valid_triplets = self.triplets[self.valid_idx]

        elif self.hparams.split_by == "img":
            self.train_triplets = pickle.load(open(train_triplets, "rb"))
            self.valid_triplets = pickle.load(open(valid_triplets, "rb"))

            if self.hparams.subset:
                subset_idx = np.random.choice(len(self.train_triplets), len(self.train_triplets)//20, replace=False)
                self.train_triplets = self.train_triplets[subset_idx]

        if self.hparams.split_by == "triplet":
            self.test_triplets = self.train_triplets
            # self.test_triplets = self.triplets
        elif self.hparams.split_by == "img":
            self.test_triplets = self.valid_triplets
    
        # self.train_triplets = np.array(self.train_triplets)
        # self.valid_triplets = np.array(self.valid_triplets)
        # self.test_triplets = np.array(self.test_triplets)
        self.train_dataset = torch.utils.data.TensorDataset(torch.tensor(self.train_triplets))
        self.valid_dataset = torch.utils.data.TensorDataset(torch.tensor(self.valid_triplets))
        self.test_dataset = torch.utils.data.TensorDataset(torch.tensor(self.test_triplets))

    def get_datasets(self):
        return self.train_input, self.valid_input

    @staticmethod
    def add_model_specific_args(parser):        
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
