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
import sys
sys.path.insert(0,'..')
import algorithms.pdash as pdash

class MTL_proto(MTL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_data()

    def forward(self, inputs, batch_idx):
        if self.trainer.training:
            triplet_idx = self.train_triplets
        else:
            triplet_idx = self.valid_triplets

        embeds = self.embed(inputs)
        logits = self.classifier(embeds)

        if self.trainer.training:
            prototype_idx = pdash.proto_g(embeds.cpu().detach().numpy(), np.arange(len(embeds)), self.hparams.m)
            print("\n")
            print(prototype_idx)
            proto_triplet_idx = []
            for t in triplet_idx.cpu().detach().numpy():
                t = list(t)
                # for x in t:
                #     if int(x) in prototype_idx and t not in proto_triplet_idx: proto_triplet_idx.append(t)
                # if int(t[0]) in prototype_idx and t not in proto_triplet_idx: proto_triplet_idx.append(t)
                if int(t[0]) in prototype_idx and int(t[1]) in prototype_idx and int(t[2]) in prototype_idx and t not in proto_triplet_idx: proto_triplet_idx.append(t)
            
            print(len(proto_triplet_idx))
            proto_triplet_idx = torch.tensor(proto_triplet_idx).long()
            # print(proto_triplet_idx)
            x1, x2, x3 = embeds[proto_triplet_idx[:,0]], embeds[proto_triplet_idx[:,1]], embeds[proto_triplet_idx[:,2]]
            proto_triplets = (x1, x2, x3)
        else:
            proto_triplets = None

        triplet_idx = triplet_idx.long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        
        return logits, proto_triplets, triplets

    def get_loss_acc(self, batch, batch_idx):
        inputs, labels = batch
        logits, proto_triples, triplets = self(inputs, batch_idx)

        probs = self.sigmoid(logits)

        clf_loss = self.criterion(logits, labels.type_as(logits).unsqueeze(1))
        with torch.no_grad():
            m = utils.metrics(probs, labels.unsqueeze(1))

        if self.trainer.training:
            proto_triplet_loss = self.triplet_loss(*proto_triples)
            d_ap = self.pdist(proto_triples[0], proto_triples[1])
            d_an = self.pdist(proto_triples[0], proto_triples[2])
            proto_triplet_acc = (d_ap < d_an).float().mean()
            self.log('proto_triplet_loss', proto_triplet_loss, sync_dist=True)
            self.log('proto_triplet_acc', proto_triplet_acc, prog_bar=True, sync_dist=True)
        else:
            proto_triplet_loss = 0

        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * proto_triplet_loss

        triplet_loss = self.triplet_loss(*triplets)
        with torch.no_grad():
            m = utils.metrics(probs, labels.unsqueeze(1))
            d_ap = self.pdist(triplets[0], triplets[1])
            d_an = self.pdist(triplets[0], triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()

        return clf_loss, m, triplet_loss, triplet_acc, total_loss
    
    def setup_data(self):
        train_dir = "/net/scratch/hanliu-shared/data/bm/train"
        valid_dir = "/net/scratch/hanliu-shared/data/bm/valid"
        self.train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=utils.bm_transform())
        self.valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=utils.bm_transform())
        self.test_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=utils.bm_transform())
        # self.train_inputs = torch.tensor(np.array([data[0].numpy() for data in train_dataset])).cuda()
        # self.valid_inputs = torch.tensor(np.array([data[0].numpy() for data in valid_dataset])).cuda()
        # self.train_labels = torch.tensor(np.array([data[1] for data in train_dataset])).cuda()
        # self.valid_labels = torch.tensor(np.array([data[1] for data in valid_dataset])).cuda()

        # train_triplets = "/net/scratch/tianh/bm/triplets/train_triplets.pkl"
        # valid_triplets = "/net/scratch/tianh/bm/triplets/valid_triplets.pkl"
        # train_triplets = "../data/bm_triplets/3c2_unique=182/train_triplets.pkl"
        # valid_triplets = "../data/bm_triplets/3c2_unique=182/valid_triplets.pkl"
        train_triplets = pickle.load(open(self.hparams.train_triplets, "rb"))
        valid_triplets = pickle.load(open(self.hparams.valid_triplets, "rb"))

        if self.hparams.subset:
            subset_idx = np.random.choice(len(train_triplets), len(train_triplets)//10, replace=False)
            train_triplets = train_triplets[subset_idx]

        test_triplets = valid_triplets
    
        self.train_triplets = torch.tensor(train_triplets)
        self.valid_triplets = torch.tensor(valid_triplets)
        self.test_triplets = torch.tensor(test_triplets)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--train_triplets", default=None, type=str, required=True)
        parser.add_argument("--valid_triplets", default=None, type=str, required=True)  
        parser.add_argument("--m", default=10, type=int, required=False)   
        parser.add_argument("--ckpt", default=None, type=str, required=True)  
        return parser

def main():
    parser = argparse.ArgumentParser()
    MTL.add_generic_args(parser)
    parser = MTL_proto.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    model = MTL_proto.load_from_checkpoint(args.ckpt, **dict_args)
    trainer = generic_train(model, args)

if __name__ == "__main__":
    main()
