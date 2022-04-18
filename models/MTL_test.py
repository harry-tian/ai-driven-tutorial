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
import utils
from pytorch_lightning.trainer.supporters import CombinedLoader
import sys
sys.path.insert(0, '..')
import evals.embed_evals as evals


class MTL(TN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.feature_extractor = models.resnet18(pretrained=True)

    def get_loss_acc(self, triplet_idx, input, labels, all_data):
        clf_embeds = self(input)
        logits = self.classifier(clf_embeds)
        probs = torch.sigmoid(logits)
        clf_loss = self.criterion(logits, labels.type_as(logits).unsqueeze(1))

        uniques = np.unique(triplet_idx.cpu().detach().numpy().flatten())
        val2idx = {val:i for i,val in enumerate(uniques)}
        for i, triplet in enumerate(triplet_idx):
            for j, val in enumerate(triplet):
                triplet_idx[i][j] = val2idx[int(val)]
        triplet_idx = triplet_idx.long()

        triplet_embeds = self(all_data[uniques])
        x1, x2, x3 = triplet_embeds[triplet_idx[:,0]], triplet_embeds[triplet_idx[:,1]], triplet_embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)

        triplet_loss = self.triplet_loss(*triplets)

        with torch.no_grad():
            m = utils.metrics(probs, labels.unsqueeze(1))
            d_ap = self.pdist(triplets[0], triplets[1])
            d_an = self.pdist(triplets[0], triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()

        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss

        return clf_loss, m, triplet_loss, triplet_acc, total_loss

    def training_step(self, batch, batch_idx):
        triplet_idx = batch["triplet"][0]
        input, labels = batch["clf"]
        all_data = self.train_input

        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(triplet_idx, input, labels, all_data)

        self.log('train_clf_loss', clf_loss, sync_dist=True)
        self.log('train_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_total_loss', total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        triplet_idx = batch["triplet"][0]
        input, labels = batch["clf"]
        all_data = self.train_input

        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(triplet_idx, input, labels, all_data)

        self.log('valid_clf_loss', clf_loss, sync_dist=True)
        self.log('valid_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_total_loss', total_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        triplet_idx = batch["triplet"][0]
        input, labels = batch["clf"]
        all_data = self.train_input

        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(triplet_idx, input, labels, all_data)

        self.log('test_clf_loss', clf_loss, sync_dist=True)
        self.log('test_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('test_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('test_triplet_loss', triplet_loss, sync_dist=True)
        self.log('test_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('test_total_loss', total_loss, sync_dist=True)

        train_x = self(self.train_input).cpu().detach().numpy()
        train_y = self.train_label.cpu().detach().numpy()
        test_x = self(self.test_input).cpu().detach().numpy()
        test_y = self.test_label.cpu().detach().numpy()
        knn_acc = evals.get_knn_score(train_x, train_y, test_x, test_y)
        self.log('test_1nn_acc', knn_acc, sync_dist=True)
        
        if self.hparams.syn:
            syn_x_train, syn_y_train = pickle.load(open(self.hparams.train_synthetic,"rb"))
            syn_x_test, syn_y_test = pickle.load(open(self.hparams.test_synthetic,"rb"))
            examples = evals.class_1NN_idx(train_x, train_y, test_x, test_y)
            ds_acc = evals.decision_support(syn_x_train, syn_y_train, syn_x_test, syn_y_test, examples)
            self.log('decision support', ds_acc, sync_dist=True)    

    def train_dataloader(self):
        triplet_dataset = torch.utils.data.TensorDataset(torch.tensor(self.train_triplets))
        clf_dataset = self.train_dataset
        print(f"\nlen_triplet_train:{len(triplet_dataset)}")
        triplet_loader = utils.get_dataloader(triplet_dataset, self.hparams.triplet_train_bs, "train", self.hparams.dataloader_num_workers)
        clf_loader = utils.get_dataloader(clf_dataset, self.hparams.clf_train_bs, "train", self.hparams.dataloader_num_workers)

        return CombinedLoader({"triplet": triplet_loader, "clf": clf_loader}, mode="max_size_cycle")

    def val_dataloader(self):
        triplet_dataset = torch.utils.data.TensorDataset(torch.tensor(self.valid_triplets))
        clf_dataset = self.valid_dataset
        print(f"\nlen_triplet_valid:{len(triplet_dataset)}")
        triplet_loader = utils.get_dataloader(triplet_dataset, min(len(triplet_dataset),self.hparams.triplet_valid_bs),
                "valid", self.hparams.dataloader_num_workers)
        clf_loader = utils.get_dataloader(clf_dataset, min(len(clf_dataset),self.hparams.clf_valid_bs), 
                "valid", self.hparams.dataloader_num_workers)

        return CombinedLoader({"triplet": triplet_loader, "clf": clf_loader}, mode="max_size_cycle")

    def test_dataloader(self):
        triplet_dataset = torch.utils.data.TensorDataset(torch.tensor(self.test_triplets))
        clf_dataset = self.test_dataset
        print(f"\nlen_triplet_test:{len(triplet_dataset)}")
        triplet_loader = utils.get_dataloader(triplet_dataset, min(len(triplet_dataset),self.hparams.triplet_test_bs),
                "test", self.hparams.dataloader_num_workers)
        clf_loader = utils.get_dataloader(clf_dataset, min(len(clf_dataset),self.hparams.clf_test_bs), 
                "test", self.hparams.dataloader_num_workers)

        return CombinedLoader({"triplet": triplet_loader, "clf": clf_loader}, mode="max_size_cycle")

    @staticmethod
    def add_model_specific_args(parser):
        parser = TN.add_model_specific_args(parser)
        parser.add_argument("--MTL_hparam", action="store_true")
        parser.add_argument("--lamda", default=0.5, type=float)
        parser.add_argument("--check_val_every_n_epoch", default = 1, type=int)
        parser.add_argument("--early_stop_patience", default = 10, type=int)

        parser.add_argument("--triplet_train_bs", default=20, type=int)
        parser.add_argument("--triplet_valid_bs", default=120, type=int)
        parser.add_argument("--triplet_test_bs", default=120, type=int)
        parser.add_argument("--clf_train_bs", default=20, type=int)
        parser.add_argument("--clf_valid_bs", default=120, type=int)
        parser.add_argument("--clf_test_bs", default=120, type=int)
        return parser

def main():
    parser = utils.add_generic_args()
    MTL.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    model = MTL(**dict_args)

    monitor = "valid_total_loss"
    print(args.early_stop_patience, args.check_val_every_n_epoch)
    early_stop_callback = EarlyStopping(monitor="valid_total_loss", min_delta=0.00, patience=args.early_stop_patience, verbose=True, mode="min")
    # trainer = Trainer(callbacks=[early_stop_callback])
    trainer = utils.generic_train(model, args, monitor, callbacks = [early_stop_callback], check_val_every_n_epoch = args.check_val_every_n_epoch)

if __name__ == "__main__":
    main()
