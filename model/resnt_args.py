# -*- coding: utf-8 -*-
import os, pickle
import time
import argparse
import shutil
from pathlib import Path

import numpy as np
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


class RESN(pl.LightningModule):

    def __init__(self, verbose=False, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_pairwise_distance = torch.Tensor(pickle.load(open(self.hparams.train_pairwise_distance, "rb")), device=self.device)
        self.valid_pairwise_distance = torch.Tensor(pickle.load(open(self.hparams.valid_pairwise_distance, "rb")), device=self.device)
        self.loss_lambda = self.hparams.loss_lambda

        self.feature_extractor = models.resnet18(pretrained=self.hparams.pretrained)
        num_features = 1000

        self.embed_dim = self.hparams.embed_dim
        self.criterion = nn.BCEWithLogitsLoss()
        self.triplet_loss = nn.TripletMarginLoss()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, 256), 
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(), nn.Linear(256, self.embed_dim)
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Dropout(), nn.Linear(self.embed_dim, 1)
        )

        if verbose: 
            self.summarize()


    def metrics(self, prob, target, threshold=0.5):
        pred = (prob >= threshold).long()
        tp, fp, tn, fn, sup = stat_scores(pred, target, ignore_index=0)
        if 0 < sup < len(target):
            precision, recall, _ = precision_recall_curve(pred, target)
            auprc = auc(recall, precision)
        m = {}
        m['pred'] = pred
        m['auc'] = auroc(prob, target) if 0 < sup < len(target) else None
        m['acc'] = (tp + tn) / (tp + tn + fp + fn)
        m['tpr'] = tp / (tp + fn)
        m['tnr'] = tn / (tn + fp)
        m['ppv'] = tp / (tp + fp)
        m['f1'] = 2 * tp / (2 * tp + fp + fn)
        m['ap'] = average_precision(prob, target)
        m['auprc'] = auprc if 0 < sup < len(target) else None
        return m

    def embed(self, x):
        embeds = self.feature_extractor(x)
        embeds = self.fc(embeds)
        return embeds

    def forward(self, x, i):
        z = self.embed(x)
        logits = self.classifier(z)
        pairwise = self.train_pairwise_distance[i][:, i]
        comb = torch.combinations(torch.range(0, len(z)-1).long(), r=3)
        triplet_idx = []
        for c in comb:
            anchor, pos, neg = c
            # print(pairwise.shape, anchor, pos)
            if pairwise[anchor, pos] > pairwise[anchor, neg]:
                triplet_idx.append((anchor, neg, pos))
            else:
                triplet_idx.append((anchor, pos, neg))
        triplet_idx = torch.Tensor(triplet_idx).long()
        # print(triplet_idx[:,0], triplet_idx[:,1], triplet_idx[:,2], triplet_idx[:,0].dtype)
        triplets = (z[triplet_idx[:,0]], z[triplet_idx[:,1]], z[triplet_idx[:,2]])
        return logits, triplets

    def training_step(self, batch, batch_idx):
        x, y, i = batch
        logits, triplets = self(x, i)
        prob = torch.sigmoid(logits)
        loss = self.criterion(logits, y.type_as(logits).unsqueeze(1))
        triplet_loss = self.triplet_loss(*triplets)
        total_loss = loss + self.loss_lambda * triplet_loss
        with torch.no_grad():
            m = self.metrics(prob, y.unsqueeze(1))
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        self.log('train_total_loss', total_loss, sync_dist=True)
        self.log('train_acc', m['acc'], prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, i = batch
        logits = self.classifier(self.embed(x))
        prob = torch.sigmoid(logits)
        loss = self.criterion(logits, y.type_as(logits).unsqueeze(1))
        m = self.metrics(prob, y.unsqueeze(1))
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

    def train_dataloader(self):
        ImageWithIndices = dataset_with_indices(torchvision.datasets.ImageFolder)
        dataset = ImageWithIndices(
            self.hparams.train_dir, transform=self.parse_augmentation()
            )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.hparams.train_batch_size, 
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=True, shuffle=True)
        return dataloader

    def val_dataloader(self):
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        ImageWithIndices = dataset_with_indices(torchvision.datasets.ImageFolder)
        dataset = ImageWithIndices(
            self.hparams.valid_dir, transform=val_transform
            )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=len(dataset), 
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=False, shuffle=False)
        return dataloader

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--train_pairwise_distance", default=None, type=str, required=True)
        parser.add_argument("--valid_pairwise_distance", default=None, type=str, required=True)
        parser.add_argument('--loss_lambda', type=float, default=0.5)
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
        parser.add_argument('--kernel', type=str, default='gaussian', help='hparam for kernel [guassian|laplace|invquad]')
        parser.add_argument('--gamma', type=float, default=1.0, help='hparam for kernel')
        parser.add_argument('--eps', type=float, default=1e-12, help='label smoothing factor for learning')
        parser.add_argument("--horizontal_flip", default=0, type=float)
        parser.add_argument("--vertical_flip", default=0, type=float)
        parser.add_argument("--rotate", default=0, type=int)
        parser.add_argument("--translate", default=0, type=float)
        parser.add_argument("--scale", default=0, type=float)
        parser.add_argument("--shear", default=0, type=float)
        return parser

def dataset_with_indices(cls):

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def train(
    model:RESN,
    args: argparse.Namespace,
    early_stopping_callback=False,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
    ):

    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    # build logger
    ## WandB logger
    experiment = wandb.init(
        mode=args.wandb_mode, 
        group=args.wandb_group
    )
    logger = WandbLogger(
        project="imagenet_bm",
        experiment=experiment
    )

    # add custom checkpoints
    ckpt_path = os.path.join(
        args.output_dir, logger.version, "checkpoints",
    )
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{valid_loss:.2f}", monitor="valid_loss", mode="min", save_last=True, save_top_k=3, verbose=True
        )

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
        **train_params,
    )

    if args.do_train:
        trainer.fit(model)
        # save best model to `best_model.ckpt`
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)

    return trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--dataloader_num_workers", default=4, type=int)
    parser.add_argument("--train_dir", default=None, type=str, required=True)
    parser.add_argument("--valid_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_mode", default="online", type=str)
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
    model = RESN(**dict_args)
    trainer = train(model, args)

if __name__ == "__main__":
    main()
