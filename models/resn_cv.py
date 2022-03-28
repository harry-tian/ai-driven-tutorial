# -*- coding: utf-8 -*-
import os
import time
import argparse
from sklearn import multiclass
import torch
import torchvision
from torch import nn
from torchvision import  models
import pytorch_lightning as pl
import wandb
import utils

import warnings
warnings.filterwarnings("ignore")


class RESN(pl.LightningModule):
    def __init__(self, split=None, verbose=False, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = models.resnet18(pretrained=True)#self.hparams.pretrained)
        num_features = 1000

        self.embed_dim = self.hparams.embed_dim
        if self.hparams.multiclass:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.fc = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, 256), 
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(), nn.Linear(256, self.embed_dim)
        )])
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Dropout(), nn.Linear(self.embed_dim, 4)
        )

        self.train_idx, self.valid_idx, self.test_idx = split
        transform = utils.bm_transform_aug() if self.hparams.transform == "bm" else utils.xray_transform_aug()
        self.dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=transform)

        self.summarize()

    def embed(self, x):
        embeds = self.feature_extractor(x)
        for layer in self.fc:
            embeds = layer(embeds)
        return embeds

    def forward(self, x):
        z = self.embed(x)
        x = self.classifier(z)
        return x

    def get_loss_acc(self, logits, target):
        prob = torch.nn.functional.softmax(logits)
        loss = self.criterion(logits, target)
        with torch.no_grad():
            acc = utils.get_acc(prob, target, multiclass=self.hparams.multiclass)
            # print(acc)
        return loss, {"acc":acc}

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss, m = self.get_loss_acc(logits, y)
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', m['acc'], prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss, m = self.get_loss_acc(logits, y)
        self.log('valid_loss', loss, sync_dist=True)
        self.log('valid_acc', m['acc'], prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss, m = self.get_loss_acc(logits, y)
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', m['acc'], prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.opt = optimizer
        return optimizer

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.hparams.train_batch_size, 
            num_workers=self.hparams.dataloader_num_workers, drop_last=True, sampler=self.train_idx)
        print(f"\ntrain:{len(list(dataloader)[0][0])}")
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.valid_idx), 
            num_workers=self.hparams.dataloader_num_workers, drop_last=False, sampler=self.valid_idx)
        print(f"\nvalid:{len(list(dataloader)[0][0])}")
        return dataloader


    def test_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.test_idx), 
            num_workers=self.hparams.dataloader_num_workers, drop_last=False, sampler=self.test_idx)
        print(f"\ntest:{len(list(dataloader)[0][0])}")
        return dataloader

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
        parser.add_argument("--transform", default="bm", type=str)
        return parser

def main():
    parser = utils.add_generic_args()
    parser.add_argument("--splits", default=None, type=str, required=True)
    RESN.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    
    import pickle
    splits = pickle.load(open(dict_args["splits"],"rb"))
    for split in splits:
        model = RESN(split, **dict_args)
        trainer = utils.generic_train(model, args, "valid_loss")
        break

if __name__ == "__main__":
    main()
