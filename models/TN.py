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

from omegaconf import OmegaConf as oc
from RESN import RESN
import trainer
import pandas as pd

class TN(RESN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_embeds = None
        self.summarize()

    # def train_triplets_step(self, triplet_idx):
    #     self.train_embeds = self(self.train_input)
    #     x1, x2, x3 = self.train_embeds[triplet_idx[:,0]], self.train_embeds[triplet_idx[:,1]], self.train_embeds[triplet_idx[:,2]]
    #     triplets = (x1, x2, x3)
    #     return self.triplet_loss_acc(triplets)

    def train_triplets_step(self, triplet_idx):
        uniques = np.unique(triplet_idx.cpu().detach().numpy().flatten())
        if len(uniques) < len(self.train_input):
            val2idx = {val:i for i,val in enumerate(uniques)}
            for i, triplet in enumerate(triplet_idx):
                for j, val in enumerate(triplet):
                    triplet_idx[i][j] = val2idx[int(val)]
            triplet_idx = triplet_idx.long()
            input = self.train_input[uniques]
        else: input = self.train_input

        embeds = self(input)
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)

        return self.triplet_loss_acc(triplets)

    def mixed_triplets_step(self, triplet_idx, input):
        self.train_embeds = self(self.train_input)
        
        embeds = self(input)
        x1, x2, x3 = embeds[triplet_idx[:,0]], self.train_embeds[triplet_idx[:,1]], self.train_embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        return self.triplet_loss_acc(triplets)

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

        knn_acc, ds_acc = self.test_evals()

        df = pd.read_csv("results.csv")
        df = pd.concat([df, pd.DataFrame({"wandb_group": [self.hparams.wandb_group], "wandb_name": [self.hparams.wandb_name],
            "test_clf_acc": [-1], "test_clf_loss": [-1], "test_1nn_acc": [knn_acc], "test_triplet_acc":[triplet_acc.item()], "decision_support": [ds_acc]})], sort=False)
        df.to_csv("results.csv", index=False)

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.train_triplets))
        print(f"\nlen_train:{len(dataset)}")
        return trainer.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.valid_triplets))
        print(f"\nlen_valid:{len(dataset)}")
        return trainer.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)

    def test_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.test_triplets))
        print(f"\nlen_test:{len(dataset)}")
        return trainer.get_dataloader(dataset, len(dataset), "test", self.hparams.dataloader_num_workers)

def main():
    parser = trainer.config_parser()
    config_files = parser.parse_args()
    configs = trainer.load_configs(config_files)

    print(configs)

    pl.seed_everything(configs["seed"])
    model = TN(**configs)
    monitor = "valid_triplet_loss"
    trainer.generic_train(model, configs, monitor)

if __name__ == "__main__":
    main()
