# -*- coding: utf-8 -*-
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

class TN_food(TripletNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, triplet_idx, batch_idx=0):
        dataset = self.dataset
        
        embeds = self.embed(dataset)
        if self.trainer.testing and self.hparams.do_embed and batch_idx==1:
            if not self.hparams.embed_path:
                embed_path = f"embeds/{self.hparams.wandb_project}.pkl"
            else:
                embed_path = self.hparams.embed_path
            pickle.dump(embeds.cpu(), open(embed_path,"wb"))
            print(f"dumped embeds to {embed_path}")

        triplet_idx = triplet_idx.long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        return triplets
    
    def setup(self, stage):
        self.triplets = np.array(pickle.load(open("/net/scratch/tianh/food100-dataset/triplets_idx.pkl", "rb")))

        # subset_idx = np.random.choice(len(self.triplets), len(self.triplets)//10, replace=False)
        # self.triplets = self.triplets[subset_idx]

        data_dir = '/net/scratch/tianh/food100-dataset/images'
        dataset = torchvision.datasets.ImageFolder(data_dir, transform=utils.food_transform())
        
        if self.hparams.split_by == "triplet":
            total_idx = np.arange(len(self.triplets))
            self.train_idx = np.random.choice(total_idx, int(len(total_idx)*0.8), replace=False)
            self.valid_idx = np.setdiff1d(total_idx, self.train_idx)
            self.train_triplets = self.triplets[self.train_idx]
            self.valid_triplets = self.triplets[self.valid_idx]

            dataset = torch.tensor(np.array([data[0].numpy() for data in dataset])).cuda()
            self.dataset = dataset

        elif self.hparams.split_by == "img":
            total_idx = np.arange(len(dataset))
            train_img_idx = np.random.choice(total_idx, int(len(total_idx)*self.hparams.img_split), replace=False)
            valid_img_idx = np.setdiff1d(total_idx, train_img_idx)

            train_label = torch.tensor(np.array([dataset[i][1] for i in train_img_idx]))
            valid_label = torch.tensor(np.array([dataset[i][1] for i in valid_img_idx]))
          
            train_triplets = []
            valid_triplets = []
            for t in self.triplets:
                if t[0] in valid_label and t[1] in train_label and t[2] in train_label:
                    valid_triplets.append(t)
                elif t[0] in train_label and t[1] in train_label and t[2] in train_label:
                    train_triplets.append(t)

            dataset = torch.tensor(np.array([data[0].numpy() for data in dataset])).cuda()

            self.dataset =  dataset
            self.train_triplets = np.array(train_triplets)
            self.valid_triplets = np.array(valid_triplets)
            
        self.test_triplets = self.triplets

    @staticmethod
    def add_model_specific_args(parser):
        return parser

def main():
    parser = argparse.ArgumentParser()
    TripletNet.add_generic_args(parser)
    parser = TN_food.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    model = TN_food(**dict_args)
    trainer = generic_train(model, args)

if __name__ == "__main__":
    main()
