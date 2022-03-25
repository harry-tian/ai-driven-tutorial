# -*- coding: utf-8 -*-
import os
import time
import argparse
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
    def __init__(self, train_idx=None, valid_idx=None, verbose=False, **config_kwargs):
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        prob = torch.sigmoid(logits)
        loss = self.criterion(logits, y.type_as(logits).unsqueeze(1))
        with torch.no_grad():
            m = utils.metrics(prob, y.unsqueeze(1))
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', m['acc'], prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        prob = torch.sigmoid(logits)
        loss = self.criterion(logits, y.type_as(logits).unsqueeze(1))
        m = utils.metrics(prob, y.unsqueeze(1))
        self.log('valid_loss', loss, sync_dist=True)
        self.log('valid_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_sensitivity', m['tpr'], sync_dist=True)
        self.log('valid_specificity', m['tnr'], sync_dist=True)
        self.log('valid_precision', m['ppv'], sync_dist=True)
        self.log('valid_f1', m['f1'], sync_dist=True)
        self.log('valid_ap', m['ap'], sync_dist=True)
        self.log('valid_auprc', m['auprc'], sync_dist=True)
        return {'valid_loss': loss, 'valid_auc': m['auc']}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        prob = torch.sigmoid(logits)
        loss = self.criterion(logits, y.type_as(logits).unsqueeze(1))
        m = utils.metrics(prob, y.unsqueeze(1))
        self.log('test_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('test_auc', m['auc'], prog_bar=True, sync_dist=True)
        return {'test_loss': loss, 'test_auc': m['auc']}

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

def cross_validate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--dataloader_num_workers", default=4, type=int)
    parser.add_argument("--train_dir", default=None, type=str, required=True)
    parser.add_argument("--valid_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=False)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_mode", default="online", type=str)
    parser.add_argument("--wandb_project", default="?", type=str)
    parser.add_argument("--wandb_entity", default="harry-tian", type=str)
    parser.add_argument("--do_train", action="store_true")
    RESN.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{__name__}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)
    
    dict_args = vars(args)



    for split in splits:
        train_idx, valid_idx = split
        model = RESN(train_idx=train_idx, valid_idx=valid_idx, **dict_args)
        trainer = utils.generic_train(model, args, "valid_loss")

if __name__ == "__main__":
    main()
    # cross_validate()
