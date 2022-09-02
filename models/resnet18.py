# -*- coding: utf-8 -*-
import sys, pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision, os
from torchvision import  models
import pytorch_lightning as pl
import trainer, transforms
import pandas as pd

sys.path.insert(0, '..')
import evals.embed_evals as evals

import warnings
warnings.filterwarnings("ignore")


class resnet18(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters() 

        self.encoder = models.resnet18(
            pretrained=self.hparams.pretrained, zero_init_residual=not self.hparams.pretrained)
        num_features = self.encoder.fc.weight.shape[1]

        self.embed_dim = self.hparams.embed_dim

        if self.embed_dim != num_features:
            self.encoder.fc = nn.Sequential(nn.Linear(num_features, self.embed_dim, bias=False))
            self.classifier = nn.Sequential(nn.Linear(self.embed_dim, self.hparams.num_class))
        else:
            self.encoder.fc = nn.Identity()
            self.classifier = nn.Sequential(nn.Linear(num_features, self.hparams.num_class))

        self.clf_criterion = nn.CrossEntropyLoss()
        self.summarize()

    def forward(self, inputs):
        embeds = self.encoder(inputs)
        return embeds

    def classification_step(self, x, y, fold):
        embeds = self(x)
        logits = self.classifier(embeds)
        clf_acc = (logits.argmax(1) == y).float().mean()
        clf_loss = self.clf_criterion(logits, y)
        self.log(f'{fold}_clf_loss', clf_loss)
        self.log(f'{fold}_clf_acc', clf_acc)
        return clf_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.classification_step(x, y, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.classification_step(x, y, 'valid')
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.classification_step(x, y, 'test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self):
        transform = transforms.get_transform(self.hparams.transform, aug=False)
        dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=transform)
        dataloader = trainer.get_dataloader(dataset, 
            batch_size=self.hparams.train_batch_size, 
            split = 'train',
            num_workers=self.hparams.dataloader_num_workers, )
        return dataloader
        
    def val_dataloader(self):
        transform = transforms.get_transform(self.hparams.transform, aug=False)
        dataset = torchvision.datasets.ImageFolder(self.hparams.valid_dir, transform=transform)
        dataloader = trainer.get_dataloader(dataset, 
            batch_size=self.hparams.train_batch_size, 
            split = 'valid',
            num_workers=self.hparams.dataloader_num_workers, )
        return dataloader

    def test_dataloader(self): 
        transform = transforms.get_transform(self.hparams.transform, aug=False)
        dataset = torchvision.datasets.ImageFolder(self.hparams.test_dir, transform=transform)
        dataloader = trainer.get_dataloader(dataset, 
            batch_size=self.hparams.train_batch_size, 
            split = 'test',
            num_workers=self.hparams.dataloader_num_workers, )
        return dataloader


def main():
    parser = trainer.config_parser()
    config_files = parser.parse_args()
    configs = trainer.load_configs(config_files)
    print(configs)

    pl.seed_everything(configs["seed"])
    profiler = configs['profiler'] if 'profiler' in configs else None
    # from pytorch_lightning.profiler import SimpleProfiler
    # profiler = SimpleProfiler()

    model = resnet18(profiler=profiler, **configs)
    monitor = "valid_clf_loss"
    trainer.generic_train(model, configs, monitor, profiler=profiler)


if __name__ == "__main__":
    main()
