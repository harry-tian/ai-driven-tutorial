# -*- coding: utf-8 -*-
import sys, pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import  models
import pytorch_lightning as pl
import trainer, transforms

sys.path.insert(0, '..')
import evals.embed_evals as evals
from sklearn.metrics.pairwise import euclidean_distances as euc_dist

import warnings
warnings.filterwarnings("ignore")


def syn_trans(x, y, w1=1, w2=1, theta=0, s1=0, s2=0, center=True):
    u = x.mean(0); x = x - u if center else x
    u0, u1 = x[y==0].mean(0), x[y==1].mean(0); u01 = u1 - u0
    theta = 0.5 * np.pi - np.arctan(u01[1] / u01[0]) if theta == 'hard' else theta
    theta = 0 - np.arctan(u01[1] / u01[0]) if theta == 'easy' else theta
    apply = lambda t, x: t.dot(x.T).T
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    stretch = np.array([[w1, s1], [s2, w2]])
    # trans = lambda x: apply(apply(stretch, rot), x) # rotate first
    trans = lambda x: apply(stretch, apply(rot, x)) # stretch first
    return trans(x) + u if center else trans(x)

def ord_dist(a, b, order=2):
    order = np.array(order) if len(order) > 1 else order
    root = 2 if len(order) > 1 else order
    diff = a[:,np.newaxis].repeat(len(b),1) - b
    return (np.abs(diff)**order).sum(-1)**(1/root)

def get_nn_mat(dist, y_test, y_train):
    mask_train = np.tile(y_train, (len(y_test), 1))
    apply_mask = lambda x, m: x + (-(m - 1) * x.max())
    nn_mat = np.arange(len(y_test)).reshape(-1, 1)
    for label in np.sort(np.unique(y_train)):
        mask_in = label == mask_train
        in1nn = np.argmin(apply_mask(dist, mask_in), 1)
        nn_mat = np.hstack([nn_mat, in1nn.reshape(-1, 1)])
    return nn_mat

def eval_nn_mat(dist, nn_mat, y_test, y_train):
    dst = dist.take(nn_mat[:,0], 0)
    dnn = np.vstack([np.take_along_axis(dst, nn_mat[:,1+c].reshape(-1,1), 1).ravel() for c in np.sort(np.unique(y_train))])
    y_true = y_test.take(nn_mat[:,0])
    return dnn.argmin(0) == y_true


class MTLT(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters() 
        self.setup_data()
        self.backbone = models.resnet18(pretrained=self.hparams.pretrained)
        self.backbone.fc = nn.Identity()
        self.embed_dim = self.hparams.embed_dim
        num_features = 512
        if self.embed_dim:
            self.fc = nn.ModuleList([nn.Sequential(
                nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, self.embed_dim), 
            )])
        else:
            self.fc = nn.ModuleList([nn.Identity()])
        self.criterion = nn.TripletMarginLoss()
        if 'profiler' in kwargs:
            self.profiler = kwargs['profiler']

    def setup_data(self):
        if self.hparams.transform == 'bm':
            affine = {}
            degree, translate, scale = 30, 0.1, 0.2
            affine["degrees"] = degree
            affine["translate"] = (translate, translate)
            affine["scale"] = (1 - scale, 1 + scale)
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomVerticalFlip(0.5),
                torchvision.transforms.RandomAffine(**affine),
            ])
            valid_transform = torchvision.transforms.ToTensor()
        else:
            train_transform = transforms.get_transform(self.hparams.transform, aug=True)
            valid_transform = transforms.get_transform(self.hparams.transform, aug=False)
        self.train_dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=train_transform)
        self.valid_dataset = torchvision.datasets.ImageFolder(self.hparams.valid_dir, transform=valid_transform)
        self.test_dataset = torchvision.datasets.ImageFolder(self.hparams.test_dir, transform=valid_transform)
        self.ref_dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=valid_transform)
        if self.hparams.transform == 'wv':
            self.train_dataset = self.load_dataset_to_memory(self.train_dataset)
            self.valid_dataset = self.load_dataset_to_memory(self.valid_dataset)
            self.test_dataset = self.load_dataset_to_memory(self.test_dataset)
            self.ref_dataset = self.train_dataset
        self.train_triplets = np.array(pickle.load(open(self.hparams.train_triplets, "rb")))
        self.valid_triplets = np.array(pickle.load(open(self.hparams.valid_triplets, "rb")))
        self.test_triplets = np.array(pickle.load(open(self.hparams.test_triplets, "rb")))
        self.syn_x_train, self.syn_x_valid, self.syn_x_test = None, None, None
        if self.hparams.syn:
            self.syn_x_train = pickle.load(open(self.hparams.train_synthetic, "rb"))
            self.syn_x_valid = pickle.load(open(self.hparams.valid_synthetic, "rb"))
            self.syn_x_test = pickle.load(open(self.hparams.test_synthetic, "rb"))

    def load_dataset_to_memory(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, len(dataset), num_workers=1)
        batch = next(iter(loader))
        return (batch[0].to(self.device), batch[1].to(self.device))

    def sample_train_clf_trips(self, x_idx, ys):
        classes = torch.unique(ys)
        k_trips = self.hparams.train_batch_size // len(classes)
        clf_trips = []
        for c in classes:
            in_idx = x_idx[ys == c]
            out_idx = x_idx[ys != c]
            combs = torch.combinations(in_idx)
            combs_idx = torch.arange(len(combs)).to(combs.device)
            prods = torch.cartesian_prod(combs_idx, out_idx)
            combs = torch.cat([combs[prods[:, 0]], prods[:, 1].unsqueeze(1)], 1)
            trip = combs[torch.randperm(len(combs))[:k_trips]]
            clf_trips.append(trip)
        return torch.cat(clf_trips)

    def sample_test_clf_trips(self, an_idx, nn_idx, y_an, y_nn, n_trips):
        classes = torch.unique(y_an)
        k_trips = n_trips // len(classes)
        clf_trips = []
        for c in classes:
            a_idx = an_idx[y_an == c]
            in_idx = nn_idx[y_nn == c]
            out_idx = nn_idx[y_nn != c]
            combs = torch.cartesian_prod(a_idx, in_idx, out_idx)
            trip = combs[torch.randperm(len(combs))[:k_trips]]
            clf_trips.append(trip)
        return torch.cat(clf_trips)

    def sample_xs_ys(self, dataset, x_idx=None):
        if type(dataset) == tuple:
            data = (dataset[0][x_idx], dataset[1][x_idx]) if x_idx is not None else dataset
        else:
            subset = torch.utils.data.Subset(dataset, x_idx.cpu()) if x_idx is not None else dataset
            loader = torch.utils.data.DataLoader(
                subset, len(subset), num_workers=1)
            data = next(iter(loader))
        return data
    
    def forward(self, inputs):
        embeds = self.backbone(inputs)
        for layer in self.fc: embeds = layer(embeds)
        return embeds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_idx, train_idx = torch.unique(batch, sorted=False, return_inverse=True)
        train_batch_idx = torch.unique(train_idx, sorted=False)
        xs, ys = self.sample_xs_ys(self.train_dataset, batch_idx)
        xs, ys = xs.to(self.device), ys.to(self.device)
        clf_triplet_idx = self.sample_train_clf_trips(train_batch_idx, ys).to(self.device)
        zs = self(xs)
        ta, tp, tn = zs[train_idx[:, 0]], zs[train_idx[:, 1]], zs[train_idx[:, 2]]
        triplet_loss = self.criterion(ta, tp, tn)
        ca, cp, cn = zs[clf_triplet_idx[:, 0]], zs[clf_triplet_idx[:, 1]], zs[clf_triplet_idx[:, 2]]
        clf_triplet_loss = self.criterion(ca, cp, cn)
        total_loss = self.hparams.lamda * clf_triplet_loss + (1 - self.hparams.lamda) * triplet_loss
        triplet_acc = self.trips_acc(ta, tp, tn)
        clf_triplet_acc = self.trips_acc(ca, cp, cn)
        self.log('train_clf_loss', clf_triplet_loss)
        self.log('train_clf_triplet_acc', clf_triplet_acc, prog_bar=True)
        self.log('train_triplet_loss', triplet_loss)
        self.log('train_triplet_acc', triplet_acc, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        valid_batch, valid_idx = torch.unique(batch[:, 0], sorted=False, return_inverse=True)
        train_batch, train_idx = torch.unique(batch[:, 1:], sorted=False, return_inverse=True)
        valid_batch_idx = torch.unique(valid_idx, sorted=False)
        train_batch_idx = torch.unique(train_idx, sorted=False)
        x_valid, y_valid = self.sample_xs_ys(self.valid_dataset, valid_batch)
        x_valid, y_valid = x_valid.to(self.device), y_valid.to(self.device)
        x_train, y_train = self.sample_xs_ys(self.ref_dataset, train_batch)
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        clf_triplet_idx = self.sample_test_clf_trips(valid_batch_idx, train_batch_idx, y_valid, y_train, len(valid_idx))
        clf_triplet_idx = clf_triplet_idx.to(self.device)
        z_valid, z_train= self(x_valid), self(x_train)
        ta, tp, tn = z_valid[valid_idx], z_train[train_idx[:, 0]], z_train[train_idx[:, 1]]
        triplet_loss = self.criterion(ta, tp, tn)
        ca, cp, cn = z_valid[clf_triplet_idx[:, 0]], z_train[clf_triplet_idx[:, 1]], z_train[clf_triplet_idx[:, 2]]
        clf_triplet_loss = self.criterion(ca, cp, cn)
        total_loss = self.hparams.lamda * clf_triplet_loss + (1 - self.hparams.lamda) * triplet_loss
        triplet_acc = self.trips_acc(ta, tp, tn)
        clf_triplet_acc = self.trips_acc(ca, cp, cn)
        knn_acc, ds_acc = self.eval_knn_ds(
            self.valid_dataset, self.ref_dataset, self.syn_x_train, self.syn_x_valid, status='valid')
        self.log('valid_clf_loss', clf_triplet_loss)
        self.log('valid_clf_triplet_acc', clf_triplet_acc, prog_bar=True)
        self.log('valid_triplet_loss', triplet_loss)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True)
        self.log('valid_total_loss', total_loss, prog_bar=True)
        if knn_acc:
            self.log('valid_1nn_acc', knn_acc)
        if ds_acc:
            self.log('valid_decision_support', ds_acc)

    def test_step(self, batch, batch_idx):
        test_batch, test_idx = torch.unique(batch[:, 0], sorted=False, return_inverse=True)
        train_batch, train_idx = torch.unique(batch[:, 1:], sorted=False, return_inverse=True)
        test_batch_idx = torch.unique(test_idx, sorted=False)
        train_batch_idx = torch.unique(train_idx, sorted=False)
        x_test, y_test = self.sample_xs_ys(self.test_dataset, test_batch)
        x_test, y_test = x_test.to(self.device), y_test.to(self.device)
        x_train, y_train = self.sample_xs_ys(self.ref_dataset, train_batch)
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        clf_triplet_idx = self.sample_test_clf_trips(test_batch_idx, train_batch_idx, y_test, y_train, len(test_idx))
        clf_triplet_idx = clf_triplet_idx.to(self.device)
        z_test, z_train= self(x_test), self(x_train)
        ta, tp, tn = z_test[test_idx], z_train[train_idx[:, 0]], z_train[train_idx[:, 1]]
        triplet_loss = self.criterion(ta, tp, tn)
        ca, cp, cn = z_test[clf_triplet_idx[:, 0]], z_train[clf_triplet_idx[:, 1]], z_train[clf_triplet_idx[:, 2]]
        clf_triplet_loss = self.criterion(ca, cp, cn)
        total_loss = self.hparams.lamda * clf_triplet_loss + (1 - self.hparams.lamda) * triplet_loss
        triplet_acc = self.trips_acc(ta, tp, tn)
        clf_triplet_acc = self.trips_acc(ca, cp, cn)
        knn_acc, ds_acc = self.eval_knn_ds(
            self.test_dataset, self.ref_dataset, self.syn_x_train, self.syn_x_test, status='test')
        self.log('test_clf_loss', clf_triplet_loss)
        self.log('test_clf_triplet_acc', clf_triplet_acc)
        self.log('test_triplet_loss', triplet_loss)
        self.log('test_triplet_acc', triplet_acc)
        self.log('test_total_loss', total_loss)
        if knn_acc:
            self.log('test_1nn_acc', knn_acc)
        if ds_acc:
            self.log('test_decision_support', ds_acc)

    def eval_knn_ds(self, test_ds, train_ds, syn_x_train=None, syn_x_test=None, status=None):
        x_train, y_train = self.sample_xs_ys(train_ds)
        x_test, y_test = self.sample_xs_ys(test_ds)
        z_train = self.backbone(x_train.to(self.device)).cpu().detach().numpy()
        z_test = self.backbone(x_test.to(self.device)).cpu().detach().numpy()
        y_train, y_test = y_train.numpy(), y_test.numpy()
        knn_acc = evals.get_knn_score(z_train, y_train, z_test, y_test)
        ds_acc = None
        if self.hparams.syn:
            w1, w2 = self.hparams.weights
            syn_z_train = syn_trans(syn_x_train, y_train, w1=w1, w2=w2)
            syn_z_test = syn_trans(syn_x_test, y_test, w1=w1, w2=w2)
            syn_dst = ord_dist(syn_z_test, syn_z_train, order=self.hparams.powers)
            nn_mat = get_nn_mat(euc_dist(z_test, z_train), y_test, y_train)
            ds_acc = eval_nn_mat(syn_dst, nn_mat, y_test, y_train).mean()
            results = evals.syn_evals(z_train, y_train, z_test, y_test, syn_x_train, syn_x_test, 
            self.hparams.weights, self.hparams.powers)
            to_log = ["NINO_ds_acc", "rNINO_ds_acc", "NIFO_ds_acc"]
            to_print = ["NINO_ds_err","rNINO_ds_err","NIFO_ds_err","NIs"]
            for key in to_log: self.log(status + "_" + key, results[key])
            if status == 'test':
                for key in to_print: print(f"\n{status}_{key}: {results[key]}")
        return knn_acc, ds_acc

    def trips_acc(self, a, p, n):
        dap = F.pairwise_distance(a, p)
        dan = F.pairwise_distance(a, n)
        return (dap < dan).float().mean()

    def train_dataloader(self):
        triplet_loader = torch.utils.data.DataLoader(
            torch.Tensor(self.train_triplets).long(), 
            batch_size=self.hparams.train_batch_size, 
            num_workers=self.hparams.dataloader_num_workers,
            drop_last=True, shuffle=True)
        return triplet_loader
        
    def val_dataloader(self):
        triplet_loader = torch.utils.data.DataLoader(
            torch.Tensor(self.valid_triplets).long(), 
            batch_size=len(self.valid_triplets), 
            num_workers=self.hparams.dataloader_num_workers)
        return triplet_loader

    def test_dataloader(self):
        triplet_loader = torch.utils.data.DataLoader(
            torch.Tensor(self.test_triplets).long(), 
            batch_size=len(self.test_triplets), 
            num_workers=self.hparams.dataloader_num_workers)
        return triplet_loader


def main():
    parser = trainer.config_parser()
    config_files = parser.parse_args()
    configs = trainer.load_configs(config_files)
    print(configs)

    pl.seed_everything(configs["seed"])
    profiler = configs['profiler'] if 'profiler' in configs else None
    # from pytorch_lightning.profiler import SimpleProfiler
    # profiler = SimpleProfiler()

    model = MTLT(profiler=profiler, **configs)
    monitor = "valid_total_loss"
    trainer.generic_train(model, configs, monitor, profiler=profiler)


if __name__ == "__main__":
    main()
