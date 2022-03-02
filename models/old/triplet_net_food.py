# -*- coding: utf-8 -*-
from dataclasses import replace
import os, pickle
import time
import argparse
import shutil
from pathlib import Path

import numpy as np
from numpy import tri
from sklearn import datasets
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import pytorch_lightning as pl
from torchmetrics.functional.classification import auroc, stat_scores, average_precision, precision_recall_curve, auc
from pytorch_lightning.loggers import WandbLogger
import wandb

import warnings
warnings.filterwarnings("ignore")

class TripletNet(pl.LightningModule):

    def __init__(self, verbose=True, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = self.get_data()
        self.triplets = np.array(pickle.load(open("/net/scratch/tianh/food100-dataset/triplets_idx.pkl", "rb")))

        total_idx = np.arange(len(self.triplets))
        self.train_idx = np.random.choice(total_idx, len(total_idx)*4//5, replace=False)
        self.valid_idx = np.setdiff1d(total_idx, self.train_idx)
        self.train_triplets = self.triplets[self.train_idx]
        self.valid_triplets = self.triplets[self.valid_idx]

        self.feature_extractor = models.resnet18(pretrained=True)
        num_features = 1000

        self.embed_dim = self.hparams.embed_dim
        self.triplet_loss = nn.TripletMarginLoss()
        self.pdist = nn.PairwiseDistance()

        self.hidden_size = self.hparams.hidden_size
        self.fc = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, self.hidden_size), 
            nn.BatchNorm1d(self.hidden_size), nn.ReLU(), nn.Dropout(), nn.Linear(self.hidden_size, self.embed_dim)
        )])
        
        if verbose: 
            self.summarize()

    def embed(self, x):
        embeds = self.feature_extractor(x)
        # for layer in self.fc:
        #     embeds = layer(embeds)
        return embeds
        
    def forward(self, triplet_idx):
        dataset = self.dataset
        triplet_idx = triplet_idx.long()
        
        # x1, x2, x3 = dataset[triplet_idx[:,0]].cuda(), dataset[triplet_idx[:,1]].cuda(), dataset[triplet_idx[:,2]].cuda()
        # triplets = (self.embed(x1), self.embed(x2), self.embed(x3))
        # return triplets
        
        embeds = self.embed(dataset)
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        return triplets


    def training_step(self, batch, batch_idx):
        triplet_idx = batch[0]
        triplets = self(triplet_idx)

        triplet_loss = self.triplet_loss(*triplets)
        with torch.no_grad():
            d_ap = self.pdist(triplets[0], triplets[1])
            d_an = self.pdist(triplets[0], triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()
        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        return triplet_loss

    def validation_step(self, batch, batch_idx):
        triplet_idx = batch[0]
        triplets = self(triplet_idx)

        triplet_loss = self.triplet_loss(*triplets)
        with torch.no_grad():
            d_ap = torch.nn.functional.pairwise_distance(triplets[0], triplets[1])
            d_an = torch.nn.functional.pairwise_distance(triplets[0], triplets[2])
            triplet_acc = (d_ap <= d_an).float().mean()
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)
        return {'valid_triplet_loss': triplet_loss, 'valid_triplet_acc': triplet_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def get_data(self):
        transform = transforms.Compose([
            transforms.Resize([230,230]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data_dir = '/net/scratch/tianh/food100-dataset/images'
        dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
        dataset = torch.tensor(np.array([data[0].numpy() for data in dataset])).cuda()
        
        # dataset = torch.utils.data.TensorDataset(torch.from_numpy(input))
        print(f"\nlen_dataset:{len(dataset)}")
        return dataset

    def train_dataloader(self):
        # input = self.input[self.train_idx]
        # labels = self.labels[self.train_idx]
        # dataset = torch.utils.data.TensorDataset(torch.from_numpy(input), torch.from_numpy(labels))

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(self.train_triplets)))
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.hparams.train_batch_size, 
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=True, shuffle=True)
        print(f"\nlen_train:{len(dataset)}")
        return dataloader

    def val_dataloader(self):
        # input = self.input[self.valid_idx]
        # labels = self.labels[self.valid_idx]
        # dataset = torch.utils.data.TensorDataset(torch.from_numpy(input), torch.from_numpy(labels))
        
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(self.valid_triplets)))

        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.hparams.train_batch_size, 
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=False, shuffle=False)
        print(f"\nlen_valid:{len(dataset)}")
        return dataloader

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--train_pairwise_distance", default=None, type=str, required=True)
        parser.add_argument("--valid_pairwise_distance", default=None, type=str, required=True)
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
        parser.add_argument("--hidden_size", default=256, type=int, help="Embedding size")
        parser.add_argument('--kernel', type=str, default='gaussian', help='hparam for kernel [guassian|laplace|invquad]')
        parser.add_argument('--gamma', type=float, default=1.0, help='hparam for kernel')
        parser.add_argument('--eps', type=float, default=1e-12, help='label smoothing factor for learning')
        parser.add_argument("--horizontal_flip", default=0, type=float)
        parser.add_argument("--vertical_flip", default=0.5, type=float)
        parser.add_argument("--rotate", default=30, type=int)
        parser.add_argument("--translate", default=0, type=float)
        parser.add_argument("--scale", default=0.2, type=float)
        parser.add_argument("--shear", default=0, type=float)
        return parser


def train(model:TripletNet, args: argparse.Namespace, early_stopping_callback=False,  extra_callbacks=[],  checkpoint_callback=None,  logging_callback=None, **extra_train_kwargs):
    odir = Path(model.hparams.output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    experiment = wandb.init(
        mode=args.wandb_mode, 
        project=args.wandb_project,
        group=args.wandb_group,
        name=f"{time.strftime('%m/%d_%H:%M')}")

    logger = WandbLogger(project="imagenet_bm", experiment=experiment)
    logger.watch(model, log="all")
    ckpt_path = os.path.join(args.output_dir, logger.version, "checkpoints")
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{valid_loss:.2f}", monitor="valid_triplet_loss", mode="min", save_last=True, save_top_k=3, verbose=True)

    train_params = {}
    train_params["max_epochs"] = args.max_epochs
    if args.gpus == -1 or args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        auto_select_gpus=True,
        weights_summary=None,
        callbacks=extra_callbacks + [checkpoint_callback],
        logger=logger,
        **train_params)

    if args.do_train:
        trainer.fit(model)
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)
        # logger.unwatch(model)
    return trainer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--val_batch_size", default=64, type=int)
    parser.add_argument("--dataloader_num_workers", default=4, type=int)
    parser.add_argument("--train_dir", default=None, type=str, required=True)
    parser.add_argument("--valid_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_project", default=None, type=str)
    parser.add_argument("--wandb_mode", default="online", type=str)
    parser.add_argument("--do_train", action="store_true")
    TripletNet.add_model_specific_args(parser, os.getcwd())
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
    model = TripletNet(**dict_args)
    trainer = train(model, args)

if __name__ == "__main__":
    main()
