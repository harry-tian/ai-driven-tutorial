# -*- coding: utf-8 -*-
from email.policy import default
import os, pickle
import argparse
from torch import nn

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import warnings
from torchvision import  models
warnings.filterwarnings("ignore")
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from TN import TN
import trainer
from omegaconf import OmegaConf as oc

import pandas as pd

class MTL(TN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.loader_mode = 'max_size_cycle'

    def train_triplets_step(self, triplet_idx, clf_input, labels):
        clf_embeds = self(clf_input)
        clf_loss, m = self.clf_loss_acc(clf_embeds, labels)

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

        triplet_loss, triplet_acc = self.triplet_loss_acc(triplets)
        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss
        return clf_loss, m, triplet_loss, triplet_acc, total_loss

    def training_step(self, batch, batch_idx):
        triplet_idx = batch["triplet"][0]
        clf_input, labels = batch["clf"]

        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.train_triplets_step(triplet_idx, clf_input, labels)

        self.log('train_clf_loss', clf_loss, sync_dist=True)
        self.log('train_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_total_loss', total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        triplet_idx = batch["triplet"][0]
        clf_input, labels = batch["clf"]

        embeds = self(clf_input)
        clf_loss, m = self.clf_loss_acc(embeds, labels)

        train_embeds = self(self.train_input)
        valid_embeds = self(self.valid_input)
        x1, x2, x3 = valid_embeds[triplet_idx[:,0]], train_embeds[triplet_idx[:,1]], train_embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)

        triplet_loss, triplet_acc = self.triplet_loss_acc(triplets)
        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss

        self.log('valid_clf_loss', clf_loss, sync_dist=True)
        self.log('valid_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_total_loss', total_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        triplet_idx = batch["triplet"][0]
        clf_input, labels = batch["clf"]

        embeds = self(clf_input)
        clf_loss, m = self.clf_loss_acc(embeds, labels)

        train_embeds = self(self.train_input)
        test_embeds = self(self.test_input)
        x1, x2, x3 = test_embeds[triplet_idx[:,0]], train_embeds[triplet_idx[:,1]], train_embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)

        triplet_loss, triplet_acc = self.triplet_loss_acc(triplets)
        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss

        self.log('test_clf_loss', clf_loss, sync_dist=True)
        self.log('test_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('test_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('test_triplet_loss', triplet_loss, sync_dist=True)
        self.log('test_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('test_total_loss', total_loss, sync_dist=True)
        
        knn_acc, ds_acc = self.test_evals()

    def train_dataloader(self):
        triplet_dataset = torch.utils.data.TensorDataset(torch.tensor(self.train_triplets))
        clf_dataset = self.train_dataset
        print(f"\nlen_clf_train:{len(clf_dataset)}")
        print(f"\nlen_triplet_train:{len(triplet_dataset)}")
        triplet_loader = trainer.get_dataloader(triplet_dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)
        clf_loader = trainer.get_dataloader(clf_dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

        return CombinedLoader({"triplet": triplet_loader, "clf": clf_loader}, mode=self.loader_mode)

    def val_dataloader(self):
        triplet_dataset = torch.utils.data.TensorDataset(torch.tensor(self.valid_triplets))
        clf_dataset = self.valid_dataset
        print(f"\nlen_clf_train:{len(clf_dataset)}")
        print(f"\nlen_triplet_valid:{len(triplet_dataset)}")
        triplet_loader = trainer.get_dataloader(triplet_dataset, len(triplet_dataset), "valid", self.hparams.dataloader_num_workers)
        clf_loader = trainer.get_dataloader(clf_dataset, len(clf_dataset), "valid", self.hparams.dataloader_num_workers)

        return CombinedLoader({"triplet": triplet_loader, "clf": clf_loader}, mode=self.loader_mode)

    def test_dataloader(self):
        triplet_dataset = torch.utils.data.TensorDataset(torch.tensor(self.test_triplets))
        clf_dataset = self.test_dataset
        print(f"\nlen_clf_train:{len(clf_dataset)}")
        print(f"\nlen_triplet_test:{len(triplet_dataset)}")
        triplet_loader = trainer.get_dataloader(triplet_dataset, len(triplet_dataset), "test", self.hparams.dataloader_num_workers)
        clf_loader = trainer.get_dataloader(clf_dataset, len(clf_dataset), "test", self.hparams.dataloader_num_workers)
        return CombinedLoader({"triplet": triplet_loader, "clf": clf_loader}, mode=self.loader_mode)

def main():
    parser = trainer.config_parser()
    config_files = parser.parse_args()
    configs = trainer.load_configs(config_files)

    # wandb_name = "MTL_pretrained" if configs["pretrained"] else "MTL"
    # wandb_name = oc.create({"wandb_name": wandb_name}) 
    # configs = oc.merge(configs, wandb_name)
    print(configs)

    pl.seed_everything(configs["seed"])
    model = MTL(**configs)
    monitor = "valid_total_loss"
    trainer.generic_train(model, configs, monitor)

    # print(configs.early_stop_patience, configs.check_val_every_n_epoch)
    # early_stop_callback = EarlyStopping(monitor="valid_total_loss", min_delta=0.00, patience=args.early_stop_patience, verbose=True, mode="min")
    # trainer = Trainer(callbacks=[early_stop_callback])
    # trainer = trainer.generic_train(model, configs, monitor, callbacks = [early_stop_callback], check_val_every_n_epoch = args.check_val_every_n_epoch)

if __name__ == "__main__":
    main()
