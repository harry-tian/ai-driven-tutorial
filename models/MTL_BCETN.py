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

class MTL_BCETN(MTL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_data()

    def forward(self, triplet_idx, clf_idx, batch_idx):
        if self.trainer.training:
            inputs = self.train_inputs
        else:
            inputs = self.valid_inputs

        embeds = self.embed(inputs)
        if self.trainer.testing and self.hparams.do_embed and batch_idx==0:
            if not self.hparams.embed_path:
                embed_path = f"embeds/{self.hparams.wandb_project}.pkl"
            else:
                embed_path = self.hparams.embed_path
            pickle.dump(embeds.cpu(), open(embed_path,"wb"))
            print(f"\n dumped embeds to {embed_path}")

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

        probs = self.sigmoid(logits)

        if self.trainer.training:
            labels = self.train_labels
        else:
            labels = self.valid_labels
            
        if self.hparams.MTL_hparam:
            labels = labels[clf_idx]
        else:
            labels = labels

        clf_loss = self.criterion(logits, labels.type_as(logits).unsqueeze(1))
        triplet_loss = self.triplet_loss(*triplets)
        with torch.no_grad():
            m = utils.metrics(probs, labels.unsqueeze(1))
            d_ap = self.pdist(triplets[0], triplets[1])
            d_an = self.pdist(triplets[0], triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()

        total_loss = self.hparams.w1 * clf_loss + self.hparams.w2 * triplet_loss

        return clf_loss, m, triplet_loss, triplet_acc, total_loss
    
    def setup_data(self):
        train_dir = "/net/scratch/hanliu-shared/data/bm/train"
        valid_dir = "/net/scratch/hanliu-shared/data/bm/valid"
        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=utils.bm_transform())
        valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=utils.bm_transform())
        self.train_inputs = torch.tensor(np.array([data[0].numpy() for data in train_dataset])).cuda()
        self.valid_inputs = torch.tensor(np.array([data[0].numpy() for data in valid_dataset])).cuda()
        self.train_labels = torch.tensor(np.array([data[1] for data in train_dataset])).cuda()
        self.valid_labels = torch.tensor(np.array([data[1] for data in valid_dataset])).cuda()

        train_triplets = "/net/scratch/tianh/bm/triplets/train_triplets.pkl"
        valid_triplets = "/net/scratch/tianh/bm/triplets/valid_triplets.pkl"
        train_triplets = pickle.load(open(train_triplets, "rb"))
        valid_triplets = pickle.load(open(valid_triplets, "rb"))

        if self.hparams.subset:
            subset_idx = np.random.choice(len(train_triplets), len(train_triplets)//10, replace=False)
            train_triplets = train_triplets[subset_idx]

        test_triplets = valid_triplets
    
        self.train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_triplets))
        self.valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_triplets))
        self.test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_triplets))

    def get_datasets(self):
        return self.train_inputs, self.valid_inputs

    @staticmethod
    def add_model_specific_args(parser):
        return parser

def main():
    parser = argparse.ArgumentParser()
    MTL.add_generic_args(parser)
    parser = MTL_BCETN.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    model = MTL_BCETN(**dict_args)
    trainer = generic_train(model, args)

if __name__ == "__main__":
    main()
