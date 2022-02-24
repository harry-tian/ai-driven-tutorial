# -*- coding: utf-8 -*-
import os, pickle
import argparse

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import warnings
warnings.filterwarnings("ignore")

from MTL_base import MTL, generic_train
import utils

class MTL_bm(MTL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, idx, batch_idx):
        embeds = self.embed(inputs)
        logits = self.classifier(embeds)
        if self.trainer.testing and self.hparams.do_embed and batch_idx==1:
            if not self.hparams.embed_path:
                embed_path = f"embeds/{self.hparams.wandb_project}.pkl"
            else:
                embed_path = self.hparams.embed_path
            pickle.dump(embeds.cpu(), open(embed_path,"wb"))
            print(f"dumped embeds to {embed_path}")

        if self.trainer.training:
            pairwise = self.train_pairwise_distance[idx][:, idx]
        else:
            pairwise = self.valid_pairwise_distance[idx][:, idx]
            
        comb = torch.combinations(torch.range(0, len(embeds)-1).long(), r=3)
        triplet_idx = []
        for c in comb:
            anchor, pos, neg = c
            if pairwise[anchor, pos] > pairwise[anchor, neg]:
                triplet_idx.append((anchor, neg, pos))
            else:
                triplet_idx.append((anchor, pos, neg))
        
        triplet_idx = torch.Tensor(triplet_idx).long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        
        return logits, triplets
    
    def get_loss_acc(self, batch, batch_idx):
        inputs, labels, idx = batch
        logits, triplets = self(inputs, idx, batch_idx)
        probs = self.sigmoid(logits)
        clf_loss = self.criterion(logits, labels.type_as(logits).unsqueeze(1))
        triplet_loss = self.triplet_loss(*triplets)
        with torch.no_grad():
            m = utils.metrics(probs, labels.unsqueeze(1))
            d_ap = self.pdist(triplets[0], triplets[1])
            d_an = self.pdist(triplets[0], triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()

        total_loss = clf_loss + triplet_loss

        return clf_loss, m, triplet_loss, triplet_acc, total_loss
        
    def setup(self, stage):
        train_dir = "/net/scratch/hanliu-shared/data/bm/train"
        valid_dir = "/net/scratch/hanliu-shared/data/bm/valid"
        train_pairwise_distance = "../embeds/lpips.bm.train.pkl" 
        valid_pairwise_distance = "../embeds/lpips.bm.valid.pkl" 

        ImageWithIndices = utils.dataset_with_indices(torchvision.datasets.ImageFolder)
        self.train_dataset = ImageWithIndices(train_dir, transform=utils.bm_train_transform())
        ImageWithIndices = utils.dataset_with_indices(torchvision.datasets.ImageFolder)
        self.valid_dataset = ImageWithIndices(valid_dir, transform=utils.bm_valid_transform())
        ImageWithIndices = utils.dataset_with_indices(torchvision.datasets.ImageFolder)
        self.test_dataset = ImageWithIndices(valid_dir, transform=utils.bm_valid_transform())

        self.train_pairwise_distance = torch.Tensor(pickle.load(open(train_pairwise_distance, "rb")), device=self.device)
        self.valid_pairwise_distance = torch.Tensor(pickle.load(open(valid_pairwise_distance, "rb")), device=self.device)

    @staticmethod
    def add_model_specific_args(parser):
        return parser

def main():
    parser = argparse.ArgumentParser()
    MTL.add_generic_args(parser)
    parser = MTL_bm.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    model = MTL_bm(**dict_args)
    trainer = generic_train(model, args)

if __name__ == "__main__":
    main()
