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

class MTL_3(MTL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, triplet_idx, clf_idx, batch_idx):
        if self.trainer.training:
            inputs = self.train_inputs
            class_boundary = 80
        else:
            inputs = self.valid_inputs
            class_boundary = 20

        embeds = self.embed(inputs)
        if self.trainer.testing and self.hparams.do_embed and batch_idx==0:
            if not self.hparams.embed_path:
                embed_path = f"embeds/{self.hparams.wandb_project}.pkl"
            else:
                embed_path = self.hparams.embed_path
            pickle.dump(embeds.cpu(), open(embed_path,"wb"))
            print(f"\n dumped embeds to {embed_path}")

        all_triplets = torch.combinations(clf_idx, r=3)
        clf_triplet_idx = []
        for t in all_triplets:
            a, p, n = int(t[0]), int(t[1]), int(t[2])
            if a < class_boundary and p < class_boundary and n >= class_boundary:
                clf_triplet_idx.append(t.tolist())
                # print(t.tolist())
            elif a >= class_boundary and p >= class_boundary and n < class_boundary:
                clf_triplet_idx.append(t.tolist())
                # print(t.tolist())
        clf_triplet_idx = torch.tensor(clf_triplet_idx).long()
        # print(f"\n triplets_idx.shape: {triplet_idx.shape}")
        # print(f"\n clf_triplet_idx: {clf_triplet_idx}")
        # print(f"\n clf_triplet_idx.shape: {clf_triplet_idx.shape}")
        # quit()
        x1, x2, x3 = embeds[clf_triplet_idx[:,0]], embeds[clf_triplet_idx[:,1]], embeds[clf_triplet_idx[:,2]]
        clf_triplets = (x1, x2, x3)

        clf_data = embeds[clf_idx]
        logits = self.classifier(clf_data)
        
        triplet_idx = triplet_idx.long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        human_triplets = (x1, x2, x3)
        
        return logits, clf_triplets, human_triplets

    def get_loss_acc(self, batch, batch_idx):
        triplet_idx = batch[0]
        clf_idx = torch.unique(torch.flatten(triplet_idx))
        logits, clf_triplets, human_triplets = self(triplet_idx, clf_idx, batch_idx)

        probs = self.sigmoid(logits)

        if self.trainer.training:
            labels = self.train_labels
        else:
            labels = self.valid_labels
        labels = labels[clf_idx]

        clf_loss = self.triplet_loss(*clf_triplets)
        triplet_loss = self.triplet_loss(*human_triplets)
        with torch.no_grad():
            m = utils.metrics(probs, labels.unsqueeze(1))
            d_ap = self.pdist(human_triplets[0], human_triplets[1])
            d_an = self.pdist(human_triplets[0], human_triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()

        total_loss = clf_loss + triplet_loss

        return clf_loss, m, triplet_loss, triplet_acc, total_loss
    
    def setup(self, stage):
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
        # clf_train_triplets = "/net/scratch/tianh/bm/triplets/clf_train_triplets.pkl"
        # clf_valid_triplets = "/net/scratch/tianh/bm/triplets/clf_valid_triplets.pkl"

        train_triplets = pickle.load(open(train_triplets, "rb"))
        if self.hparams.subset:
            subset_idx = np.random.choice(len(train_triplets), len(train_triplets)//10, replace=False)
            train_triplets = train_triplets[subset_idx]
        valid_triplets = pickle.load(open(valid_triplets, "rb"))

        test_triplets = valid_triplets
    
        self.train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_triplets))
        self.valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_triplets))
        self.test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_triplets))

    @staticmethod
    def add_model_specific_args(parser):
        return parser

def main():
    parser = argparse.ArgumentParser()
    MTL.add_generic_args(parser)
    parser = MTL_3.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    model = MTL_3(**dict_args)
    trainer = generic_train(model, args)

if __name__ == "__main__":
    main()
