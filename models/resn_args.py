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
    def __init__(self, verbose=False, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = models.resnet18(pretrained=True)#self.hparams.pretrained)
        num_features = 1000

        self.embed_dim = self.hparams.embed_dim
        self.criterion = nn.BCEWithLogitsLoss()
        self.fc = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, 256), 
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(), nn.Linear(256, self.embed_dim)
        )])
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Dropout(), nn.Linear(self.embed_dim, 1)
        )

        if verbose: 
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
        prob = torch.sigmoid(logits)
        loss = self.criterion(logits, target.type_as(logits).unsqueeze(1))
        with torch.no_grad():
            m = utils.metrics(prob, target.unsqueeze(1), multiclass=self.hparams.multiclass)
        return loss, m

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # prob = torch.sigmoid(logits)
        # loss = self.criterion(logits, y.type_as(logits).unsqueeze(1))
        # with torch.no_grad():
        #     m = utils.metrics(prob, y.unsqueeze(1))
        loss, m = self.get_loss_acc(logits, y)
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', m['acc'], prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # prob = torch.sigmoid(logits)
        # loss = self.criterion(logits, y.type_as(logits).unsqueeze(1))
        # m = utils.metrics(prob, y.unsqueeze(1))
        loss, m = self.get_loss_acc(logits, y)
        self.log('valid_loss', loss, sync_dist=True)
        self.log('valid_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_sensitivity', m['tpr'], sync_dist=True)
        self.log('valid_specificity', m['tnr'], sync_dist=True)
        self.log('valid_precision', m['ppv'], sync_dist=True)
        self.log('valid_f1', m['f1'], sync_dist=True)
        self.log('valid_ap', m['ap'], sync_dist=True)
        self.log('valid_auprc', m['auprc'], sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # prob = torch.sigmoid(logits)
        # loss = self.criterion(logits, y.type_as(logits).unsqueeze(1))
        # m = utils.metrics(prob, y.unsqueeze(1))
        loss, m = self.get_loss_acc(logits, y)
        self.log('test_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('test_auc', m['auc'], prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.opt = optimizer
        return optimizer

    def train_dataloader(self):
        transform = utils.bm_transform_aug() if self.hparams.transform == "bm" else utils.xray_transform_aug()
        dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=transform)
        return utils.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        transform = utils.bm_transform() if self.hparams.transform == "bm" else utils.xray_transform()
        dataset = torchvision.datasets.ImageFolder(self.hparams.valid_dir, transform=transform)
        return utils.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)


    def test_dataloader(self):
        transform = utils.bm_transform() if self.hparams.transform == "bm" else utils.xray_transform()
        dataset = torchvision.datasets.ImageFolder(self.hparams.test_dir, transform=transform)
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
