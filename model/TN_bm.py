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

from RESN_TN import RESN_TN, generic_train
import utils

class TN_bm(RESN_TN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, combs, batch_idx=0):
        if self.hparams.split_by == "triplet":
            dataset = self.dataset
            pairwise = self.pairwise_distance
        elif self.hparams.split_by == "img":
            if self.trainer.training:
                dataset = self.train_dataset
                pairwise = self.train_pairwise_distance
            else:
                dataset = self.valid_dataset
                pairwise = self.valid_pairwise_distance

        embeds = self.embed(dataset)
        if self.trainer.testing and batch_idx==1:
            embeds_path = f"embeds/{self.hparams.wandb_project}.pkl"
            pickle.dump(embeds.cpu(), open(embeds_path,"wb"))
            print(f"dumped embeds to {embeds_path}")
            
        triplet_idx = []
        for c in combs:
            anchor, pos, neg = c[0], c[1], c[2]
            if pairwise[anchor, pos] > pairwise[anchor, neg]:
                triplet_idx.append((anchor, neg, pos))
            else:
                triplet_idx.append((anchor, pos, neg))

        triplet_idx = torch.Tensor(triplet_idx).long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        return triplets
    
    def setup(self, stage):
        train_dir = "/net/scratch/hanliu-shared/data/bm/train"
        valid_dir = "/net/scratch/hanliu-shared/data/bm/valid"
        train_pairwise_distance= "../embeds/lpips.bm.train.pkl" 
        valid_pairwise_distance= "../embeds/lpips.bm.valid.pkl" 

        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=utils.bm_train_transform())
        valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=utils.bm_valid_transform())
        self.train_dataset = torch.tensor(np.array([data[0].numpy() for data in train_dataset])).cuda()
        self.valid_dataset = torch.tensor(np.array([data[0].numpy() for data in valid_dataset])).cuda()

        self.train_pairwise_distance = torch.Tensor(pickle.load(open(train_pairwise_distance, "rb")), device=self.device)
        self.valid_pairwise_distance = torch.Tensor(pickle.load(open(valid_pairwise_distance, "rb")), device=self.device)
        
        if self.hparams.split_by == "triplet":
            self.dataset = self.train_dataset
            self.pairwise_distance = self.train_pairwise_distance

            self.triplets = torch.combinations(torch.arange(0, len(self.dataset)).int(), r=3)

            subset_idx = np.random.choice(len(self.triplets), len(self.triplets)//10, replace=False)
            self.triplets = self.triplets[subset_idx]

            len_triplets = np.arange(0,self.triplets.shape[0])
            self.train_idx = np.random.choice(len_triplets, int(len(len_triplets)*0.8), replace=False)
            self.valid_idx = np.setdiff1d(len_triplets, self.train_idx)
            self.train_triplets = self.triplets[self.train_idx]
            self.valid_triplets = self.triplets[self.valid_idx]

        elif self.hparams.split_by == "img":
            self.train_triplets = torch.combinations(torch.range(0, len(self.train_dataset)-1).long(), r=3)
            self.valid_triplets = torch.combinations(torch.range(0, len(self.valid_dataset)-1).long(), r=3)

        if self.hparams.split_by == "triplet":
            self.test_triplets = self.valid_triplets
            # self.test_triplets = self.triplets
        elif self.hparams.split_by == "img":
            self.test_triplets = self.train_triplets

    @staticmethod
    def add_model_specific_args(parser):        
        return parser

def main():
    parser = argparse.ArgumentParser()
    RESN_TN.add_generic_args(parser)
    parser = TN_bm.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    model = TN_bm(**dict_args)
    trainer = generic_train(model, args)

if __name__ == "__main__":
    main()
