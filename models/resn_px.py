# -*- coding: utf-8 -*-
import os
import time
import argparse
from sklearn import multiclass
import torch
import torchvision
from torch import nn
from torchvision import  models,transforms
import pytorch_lightning as pl
import wandb
import utils
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class RESN(pl.LightningModule):
    def __init__(self, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = models.resnet18(pretrained=False)
        self.feature_extractor.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = 1000

        if self.hparams.num_class > 2:
            self.criterion = nn.CrossEntropyLoss()
            self.nonlinear = nn.Softmax()
            self.out_dim = self.hparams.num_class
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            self.nonlinear = nn.Sigmoid()
            self.out_dim = 1

        self.embed_dim = self.hparams.embed_dim
        self.fc = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, 256), 
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(), nn.Linear(256, self.embed_dim)
        )])
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Dropout(), nn.Linear(self.embed_dim, self.out_dim)
        )

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
        prob = self.nonlinear(logits)
        if self.hparams.num_class < 3:
            target = target.type_as(logits).unsqueeze(1)
        loss = self.criterion(logits, target)
        with torch.no_grad():
            m = utils.metrics(prob, target, num_class=self.hparams.num_class)
        return loss, m

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
        if self.hparams.num_class < 3:
            self.log('valid_auc', m['auc'], sync_dist=True)
            # self.log('valid_sensitivity', m['tpr'], sync_dist=True)
            # self.log('valid_specificity', m['tnr'], sync_dist=True)
            # self.log('valid_precision', m['ppv'], sync_dist=True)
            # self.log('valid_f1', m['f1'], sync_dist=True)
            # self.log('valid_ap', m['ap'], sync_dist=True)
            # self.log('valid_auprc', m['auprc'], sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss, m = self.get_loss_acc(logits, y)
        self.log('test_acc', m['acc'], prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.opt = optimizer
        return optimizer

    def train_dataloader(self):
        transform = transforms.get_transform(self.hparams.transform, aug=True) 
        # transform=transforms.ToTensor()
        dataset = torchvision.datasets.DatasetFolder(self.hparams.train_dir, extensions='npy', loader=np.load, transform=transform)
        print(f"\n train:{len(dataset)}")
        return utils.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        transform = transforms.get_transform(self.hparams.transform, aug=False)
        dataset = torchvision.datasets.DatasetFolder(self.hparams.valid_dir, extensions='npy', loader=np.load, transform=transform)
        print(f"\n valid:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)


    def test_dataloader(self):
        transform = transforms.get_transform(self.hparams.transform, aug=False)
        dataset = torchvision.datasets.DatasetFolder(self.hparams.test_dir, extensions='npy', loader=np.load, transform=transform)
        print(f"\n test:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "test", self.hparams.dataloader_num_workers)


    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
        parser.add_argument("--transform", default="bm", type=str)
        return parser

def main():
    parser = utils.add_generic_args()
    RESN.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)

    model = RESN(**dict_args)

    trainer = utils.generic_train(model, args, "valid_loss")

if __name__ == "__main__":
    main()
