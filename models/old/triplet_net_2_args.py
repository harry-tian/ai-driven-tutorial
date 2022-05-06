# -*- coding: utf-8 -*-
from dataclasses import replace
import os, pickle
import time
import argparse
import shutil
from pathlib import Path

import numpy as np
from numpy import tri
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

    def __init__(self, verbose=False, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = self.get_train_dataset()
        self.valid_dataset = self.get_valid_dataset()

        self.train_pairwise_distance = torch.Tensor(pickle.load(open(self.hparams.train_pairwise_distance, "rb")), device=self.device)
        self.valid_pairwise_distance = torch.Tensor(pickle.load(open(self.hparams.valid_pairwise_distance, "rb")), device=self.device)

        self.feature_extractor = models.resnet18(pretrained=self.hparams.pretrained)
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
        
    def forward(self, batch):
        if self.training:
            dataset = self.train_dataset
            pairwise = self.train_pairwise_distance
        else:
            dataset = self.valid_dataset
            pairwise = self.valid_pairwise_distance

        triplet_idx = []
        for triplet in batch:
            anchor, pos, neg = triplet[0], triplet[1], triplet[2]
            if pairwise[anchor, pos] > pairwise[anchor, neg]:
                triplet_idx.append((anchor, neg, pos))
            else:
                triplet_idx.append((anchor, pos, neg))
        triplet_idx = torch.Tensor(triplet_idx).long()
        x1, x2, x3 = dataset[triplet_idx[:,0]][0].cuda(), dataset[triplet_idx[:,1]][0].cuda(), dataset[triplet_idx[:,2]][0].cuda()
        triplets = (self.embed(x1), self.embed(x2), self.embed(x3))
        return triplets

    def training_step(self, batch, batch_idx):
        triplets = self(batch[0])

        triplet_loss = self.triplet_loss(*triplets)
        with torch.no_grad():
            d_ap = self.pdist(triplets[0], triplets[1])
            d_an = self.pdist(triplets[0], triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()
        self.log('train_triplet_acc', triplet_acc, prog_bar=False)
        self.log('train_triplet_loss', triplet_loss)
        return triplet_loss

    def validation_step(self, batch, batch_idx):
        triplets = self(batch[0])

        triplet_loss = self.triplet_loss(*triplets)
        d_ap = torch.nn.functional.pairwise_distance(triplets[0], triplets[1])
        d_an = torch.nn.functional.pairwise_distance(triplets[0], triplets[2])
        triplet_acc = (d_ap <= d_an).float().mean()
        self.log('valid_triplet_acc', triplet_acc, prog_bar=False)
        self.log('valid_triplet_loss', triplet_loss)
        return {'valid_triplet_loss': triplet_loss, 'valid_triplet_acc': triplet_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def parse_augmentation(self):
        affine = {}
        affine["degrees"] = self.hparams.rotate
        if self.hparams.translate > 0: 
            translate = self.hparams.translate
            affine["translate"] = (translate, translate)
        if self.hparams.scale > 0: 
            scale = self.hparams.scale
            affine["scale"] = (1 - scale, 1 + scale)
        if self.hparams.shear > 0:
            shear = self.hparams.shear
            affine["shear"] = (-shear, shear, -shear, shear)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(self.hparams.horizontal_flip),
            transforms.RandomVerticalFlip(self.hparams.vertical_flip),
            transforms.RandomAffine(**affine)
        ])
        return transform

    def get_train_dataset(self):
        dataset = torchvision.datasets.ImageFolder(
            self.hparams.train_dir, transform=self.parse_augmentation()
            )
        a = [dataset[i][0].numpy() for i in range(len(dataset))]
        b = torch.utils.data.TensorDataset(torch.from_numpy(np.array(a)))
        return b

    def get_valid_dataset(self):
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.ImageFolder(
            self.hparams.valid_dir, transform=val_transform
            )
            
        a = [dataset[i][0].numpy() for i in range(len(dataset))]
        b = torch.utils.data.TensorDataset(torch.from_numpy(np.array(a)))
        return b

    def train_dataloader(self):
        total_combs = torch.combinations(torch.range(0, len(self.train_dataset)-1).int(), r=3)
        # dataset = torch.utils.data.TensorDataset(total_combs)
        subset = np.random.choice(len(total_combs), len(total_combs)//20, replace=False)
        dataset = torch.utils.data.TensorDataset(total_combs[subset])
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.hparams.train_batch_size, 
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=True, shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataset = torch.utils.data.TensorDataset(
        torch.combinations(torch.range(0, len(self.valid_dataset)-1).int(), r=3))
        

        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.hparams.val_batch_size, 
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=False, shuffle=False)
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

def dataset_with_indices(cls):

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def train(model:TripletNet, args: argparse.Namespace, early_stopping_callback=False,  extra_callbacks=[],  checkpoint_callback=None,  logging_callback=None, **extra_train_kwargs):
    odir = Path(model.hparams.output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    experiment = wandb.init(
        mode=args.wandb_mode, 
        group=args.wandb_group,
        name=f"{time.strftime('%m/%d_%H:%M')}")

    logger = WandbLogger(project="imagenet_bm", experiment=experiment)

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
