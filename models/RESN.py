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
import utils, pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.insert(0, '..')
import evals.embed_evals as evals

import warnings
warnings.filterwarnings("ignore")


class RESN(pl.LightningModule):
    def __init__(self, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.setup_data()

        self.feature_extractor = models.resnet18(pretrained=True)
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

        self.pdist = nn.PairwiseDistance()

        self.summarize()

    def forward(self, x):
        embeds = self.feature_extractor(x)
        for layer in self.fc:
            embeds = layer(embeds)
        return embeds

    def get_loss_metrics(self, input, target, triplets=None):
        embeds = self(input)

        logits = self.classifier(embeds)
        prob = self.nonlinear(logits)
        if self.hparams.num_class < 3:
            target = target.type_as(logits).unsqueeze(1)
        loss = self.criterion(logits, target)
        with torch.no_grad():
            m = utils.metrics(prob, target, num_class=self.hparams.num_class)

        if triplets is not None: m["triplet_acc"] = self.get_triplet_acc(embeds, triplets)
        else: m["triplet_acc"] = -1

        return loss, m

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, m = self.get_loss_metrics(x, y)
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', m['acc'], prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, m = self.get_loss_metrics(x, y, triplets=self.valid_triplets)
        self.log('valid_loss', loss, sync_dist=True)
        self.log('valid_acc', m['acc'], prog_bar=True, sync_dist=True)
        if self.hparams.num_class < 3: self.log('valid_auc', m['auc'], sync_dist=True)
        if m["triplet_acc"] > -1: self.log('valid_triplet_acc', m['triplet_acc'], sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, m = self.get_loss_metrics(x, y, triplets=self.test_triplets)
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', m['acc'], sync_dist=True)
        if m["triplet_acc"] > -1: self.log('test_triplet_acc', m['triplet_acc'], sync_dist=True)

        self.test_evals()

    def test_evals(self):
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
            ds_acc = evals.decision_support(syn_x_train, syn_y_train, syn_x_test, syn_y_test, examples, 
            [float(self.hparams.w1), float(self.hparams.w2)])
            self.log('decision support', ds_acc, sync_dist=True)  

    def get_triplet_acc(self, embeds, triplet_idx):
        triplet_idx = torch.tensor(triplet_idx).long()
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        
        d_ap = self.pdist(triplets[0], triplets[1])
        d_an = self.pdist(triplets[0], triplets[2])
        triplet_acc = (d_ap < d_an).float().mean()
        return triplet_acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.opt = optimizer
        return optimizer

    def setup_data(self):
        train_transform = utils.get_transform(self.hparams.transform, aug=True)
        valid_transform = utils.get_transform(self.hparams.transform, aug=False)
        self.train_dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=train_transform)
        self.valid_dataset = torchvision.datasets.ImageFolder(self.hparams.valid_dir, transform=valid_transform)
        self.test_dataset = torchvision.datasets.ImageFolder(self.hparams.test_dir, transform=valid_transform)
        self.train_input = torch.tensor(np.array([data[0].numpy() for data in self.train_dataset])).cuda()
        self.valid_input = torch.tensor(np.array([data[0].numpy() for data in self.valid_dataset])).cuda()
        self.test_input = torch.tensor(np.array([data[0].numpy() for data in self.test_dataset])).cuda()
        self.train_label = torch.tensor(np.array([data[1] for data in self.train_dataset])).cuda()
        self.valid_label = torch.tensor(np.array([data[1] for data in self.valid_dataset])).cuda()
        self.test_label = torch.tensor(np.array([data[1] for data in self.test_dataset])).cuda()

        self.train_triplets = np.array(pickle.load(open(self.hparams.train_triplets, "rb")))
        self.valid_triplets = np.array(pickle.load(open(self.hparams.valid_triplets, "rb")))
        self.test_triplets = np.array(pickle.load(open(self.hparams.test_triplets, "rb")))

    def train_dataloader(self):
        dataset = self.train_dataset
        print(f"\n train:{len(dataset)}")
        return utils.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        dataset = self.valid_dataset
        print(f"\n valid:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)

    def test_dataloader(self):
        dataset = self.test_dataset
        print(f"\n test:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "test", self.hparams.dataloader_num_workers)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
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
