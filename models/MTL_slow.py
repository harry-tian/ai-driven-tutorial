# -*- coding: utf-8 -*-
from email.policy import default
import os, pickle
import argparse
from torch import nn

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import warnings
from torchvision import  models
warnings.filterwarnings("ignore")
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from TN import TN
import trainer, transforms
from sklearn.metrics.pairwise import euclidean_distances
from pytorch_lightning.trainer.supporters import CombinedLoader
import wandb, time, random,sys
import pandas as pd
sys.path.insert(0, '..')
import evals.embed_evals as evals

class MTL(TN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.loader_mode = 'max_size_cycle'

    def setup_data(self):
        train_transform = transforms.get_transform(self.hparams.transform, aug=True)
        valid_transform = transforms.get_transform(self.hparams.transform, aug=False)
        self.train_dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=train_transform)
        self.valid_dataset = torchvision.datasets.ImageFolder(self.hparams.valid_dir, transform=valid_transform)
        self.test_dataset = torchvision.datasets.ImageFolder(self.hparams.test_dir, transform=valid_transform)

        self.train_input = torch.tensor(np.array([data[0].numpy() for data in self.train_dataset])).cuda()
        self.valid_input = torch.tensor(np.array([data[0].numpy() for data in self.valid_dataset])).cuda()
        self.test_input = torch.tensor(np.array([data[0].numpy() for data in self.test_dataset])).cuda()
        self.train_label = torch.tensor(np.array([data[1] for data in self.train_dataset]))#.cuda()
        self.valid_label = torch.tensor(np.array([data[1] for data in self.valid_dataset]))#.cuda()
        self.test_label = torch.tensor(np.array([data[1] for data in self.test_dataset]))#.cuda()

        self.train_triplets = np.array(pickle.load(open(self.hparams.train_triplets, "rb")))
        self.valid_triplets = np.array(pickle.load(open(self.hparams.valid_triplets, "rb")))
        self.test_triplets = np.array(pickle.load(open(self.hparams.test_triplets, "rb")))

        if self.hparams.syn:
            self.syn_x_train = pickle.load(open(self.hparams.train_synthetic, "rb"))
            self.syn_x_valid = pickle.load(open(self.hparams.valid_synthetic, "rb"))
            self.syn_x_test = pickle.load(open(self.hparams.test_synthetic, "rb"))

    def train_triplets_step(self, triplet_idx, clf_idx, labels):
        uniques = torch.unique(torch.concat([clf_idx, triplet_idx.flatten()]))
        if len(uniques) < len(self.train_input):
            val2idx = {val.item():i for i,val in enumerate(uniques)}
            for i, triplet in enumerate(triplet_idx):
                for j, val in enumerate(triplet):
                    triplet_idx[i][j] = val2idx[val.item()]
            triplet_idx = triplet_idx.long()
            for i, val in enumerate(clf_idx):
                clf_idx[i] = val2idx[val.item()]
            input = self.train_input[uniques]
        else: input = self.train_input
        embeds = self(input)
        
        x1, x2, x3 = embeds[triplet_idx[:,0]], embeds[triplet_idx[:,1]], embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)

        triplet_loss, triplet_acc = self.triplet_loss_acc(triplets)
        clf_loss, m = self.clf_loss_acc(embeds[clf_idx], labels)

        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss
        return clf_loss, m, triplet_loss, triplet_acc, total_loss

    def mixed_triplets_step(self, triplet_idx, clf_idx, labels, mode):
        train_uniques = torch.unique(triplet_idx[:,1:])
        if len(train_uniques) < len(self.train_input):
            val2idx = {val.item():i for i,val in enumerate(train_uniques)}
            for i, triplet in enumerate(triplet_idx):
                for j, val in enumerate(triplet[1:]):
                    triplet_idx[i][j+1] = val2idx[val.item()]
            triplet_idx = triplet_idx.long()
            input = self.train_input[train_uniques]
        else: input = self.train_input
        train_embeds = self(input)
        embeds = self(self.valid_input) if mode == "valid" else self(self.test_input)
        
        x1, x2, x3 = embeds[triplet_idx[:,0]], train_embeds[triplet_idx[:,1]], train_embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)

        triplet_loss, triplet_acc = self.triplet_loss_acc(triplets)
        clf_loss, m = self.clf_loss_acc(embeds[clf_idx], labels)

        total_loss = self.hparams.lamda * clf_loss + (1-self.hparams.lamda) * triplet_loss
        return clf_loss, m, triplet_loss, triplet_acc, total_loss

    def training_step(self, batch, batch_idx):
        triplet_idx = batch["triplet"][0]
        clf_idx, labels = batch["clf"]

        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.train_triplets_step(triplet_idx, clf_idx, labels)

        self.log('train_clf_loss', clf_loss)
        self.log('train_clf_acc', m['acc'], prog_bar=False)
        self.log('train_triplet_loss', triplet_loss)
        self.log('train_triplet_acc', triplet_acc, prog_bar=False)
        self.log('train_total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        triplet_idx = batch["triplet"][0]
        clf_idx, labels = batch["clf"]

        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.mixed_triplets_step(triplet_idx, clf_idx, labels, "valid")

        self.log('valid_clf_loss', clf_loss)
        self.log('valid_clf_acc', m['acc'], prog_bar=False)
        # self.log('valid_auc', m['auc'], prog_bar=False)
        self.log('valid_triplet_loss', triplet_loss)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=False)
        self.log('valid_total_loss', total_loss)

    def test_step(self, batch, batch_idx):
        triplet_idx = batch["triplet"][0]
        clf_idx, labels = batch["clf"]

        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.mixed_triplets_step(triplet_idx, clf_idx, labels, "test")

        self.log('test_clf_loss', clf_loss)
        self.log('test_clf_acc', m['acc'], prog_bar=False)
        # self.log('test_auc', m['auc'], prog_bar=False)
        self.log('test_triplet_loss', triplet_loss)
        self.log('test_triplet_acc', triplet_acc, prog_bar=False)
        self.log('test_total_loss', total_loss)
        return m['acc'], triplet_acc

    def embed_dataset(self, dataset):
        self.eval()
        # dataset = torch.utils.data.TensorDataset(*dataset) if self.in_memeory_dataset else dataset
        zs, dl = [], torch.utils.data.DataLoader(dataset, batch_size=self.hparams.train_batch_size)
        for x, _ in iter(dl): 
            zs.append(self(x.to(self.device)).cpu())
        return torch.cat(zs)

    def test_epoch_end(self, outputs):
        # clf_acc, triplet_acc = zip(*outputs)
        # if len(triplet_acc) > 1: triplet_acc = triplet_acc.mean()
        # else: triplet_acc = triplet_acc[0].mean()
        # if len(clf_acc) > 1: clf_acc = clf_acc.mean()
        # else: clf_acc = clf_acc[0].mean()

        results = self.eval_knn_ds(
            self.test_dataset, self.train_dataset, self.syn_x_train, self.syn_x_test)
        for k,v in results.items(): self.log(k,v)
        csv = {
            "wandb_project": self.hparams.wandb_project,
            "wandb_group": self.hparams.wandb_group,
            "wandb_name": self.hparams.wandb_name,
            "seed": self.hparams.seed,
            "weights": self.hparams.weights,
            "embed_dim": self.hparams.embed_dim,
            "lamda": self.hparams.lamda,
            # "test_clf_acc": clf_acc.cpu().detach().numpy(),
            # "test_triplet_acc": triplet_acc.cpu().detach().numpy(),
            }
        csv.update(results)
        csv = {k:[v] for k,v in csv.items()}
        if self.hparams.out_csv is not None: out_csv = self.hparams.out_csv 
        else: out_csv = "out.csv"
        out_csv = f"results/{out_csv}"
        if not os.path.isfile(out_csv): df = pd.DataFrame()
        else: df = pd.read_csv(out_csv)
        time.sleep(random.randint(0,20))
        df = pd.concat([df,pd.DataFrame(csv)])
        df.to_csv(out_csv,index=False)

    def eval_knn_ds(self, test_ds, train_ds, syn_x_train=None, syn_x_test=None):
        z_train = self.embed_dataset(train_ds).numpy()
        z_test = self.embed_dataset(test_ds).numpy()
        y_train, y_test = self.train_label.numpy(), self.test_label.numpy()
        knn_acc = evals.get_knn_score(z_train, y_train, z_test, y_test)
        results = {"test_1nn_acc":knn_acc}
        if self.hparams.syn:
            to_log = ["NINO_ds_acc", "rNINO_ds_acc", "NIFO_ds_acc"]
            RESN_d512_dir = "../embeds/wv_3d/RESN_d=512"
            RESN_d50_dir = "../embeds/wv_3d/RESN_d=512"

            for k in [1,3,5]:
                syn_evals = evals.syn_evals(z_train, y_train, z_test, y_test, syn_x_train, syn_x_test, 
                self.hparams.weights, self.hparams.powers, k=k)

                if self.hparams.model != "RESN":
                    h2h_50 = []
                    h2h_512 = []
                    for seed in range(5):
                        RESN_train_50 = pickle.load(open(f"{RESN_d50_dir}/RESN_train_seed{seed}.pkl","rb"))
                        RESN_test_50 = pickle.load(open(f"{RESN_d50_dir}/RESN_test_seed{seed}.pkl","rb"))
                        euc_dist_M = euclidean_distances(RESN_test_50,RESN_train_50)
                        RESN_NIs = evals.get_NI(euc_dist_M, y_train, y_test, k=k)
                        wins, errs, ties = evals.nn_comparison(syn_x_train, syn_x_test, syn_evals["NIs"], RESN_NIs, self.hparams.weights, self.hparams.powers)
                        h2h_50.append((wins + ties*0.5)/len(y_test))

                        RESN_train_512 = pickle.load(open(f"{RESN_d512_dir}/RESN_train_seed{seed}.pkl","rb"))
                        RESN_test_512 = pickle.load(open(f"{RESN_d512_dir}/RESN_test_seed{seed}.pkl","rb"))
                        euc_dist_M = euclidean_distances(RESN_test_512,RESN_train_512)
                        RESN_NIs = evals.get_NI(euc_dist_M, y_train, y_test, k=k)
                        wins, errs, ties = evals.nn_comparison(syn_x_train, syn_x_test, syn_evals["NIs"], RESN_NIs, self.hparams.weights, self.hparams.powers)
                        h2h_512.append((wins + ties*0.5)/len(y_test))
                    results["h2h_50"] = np.array(h2h_50).mean()
                    results["h2h_512"] = np.array(h2h_512).mean()

                for eval in to_log: results[f"{eval}_k={k}"] = syn_evals[eval]
        return results

    def train_dataloader(self):
        triplet_dataset = torch.utils.data.TensorDataset(torch.tensor(self.train_triplets))
        clf_dataset = torch.utils.data.TensorDataset(torch.tensor(np.arange(len(self.train_dataset))), self.train_label)
        print(f"\nlen_clf_train:{len(clf_dataset)}")
        print(f"\nlen_triplet_train:{len(triplet_dataset)}")
        triplet_loader = trainer.get_dataloader(triplet_dataset, self.hparams.triplet_batch_size, "train", self.hparams.dataloader_num_workers)
        clf_loader = trainer.get_dataloader(clf_dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

        return CombinedLoader({"triplet": triplet_loader, "clf": clf_loader}, mode=self.loader_mode)

    def val_dataloader(self):
        triplet_dataset = torch.utils.data.TensorDataset(torch.tensor(self.valid_triplets))
        clf_dataset = torch.utils.data.TensorDataset(torch.tensor(np.arange(len(self.valid_dataset))), self.valid_label)
        print(f"\nlen_clf_valid:{len(clf_dataset)}")
        print(f"\nlen_triplet_valid:{len(triplet_dataset)}")
        triplet_loader = trainer.get_dataloader(triplet_dataset, len(triplet_dataset), "valid", self.hparams.dataloader_num_workers)
        clf_loader = trainer.get_dataloader(clf_dataset, len(clf_dataset), "valid", self.hparams.dataloader_num_workers)

        return CombinedLoader({"triplet": triplet_loader, "clf": clf_loader}, mode=self.loader_mode)

    def test_dataloader(self):
        triplet_dataset = torch.utils.data.TensorDataset(torch.tensor(self.test_triplets))
        clf_dataset = torch.utils.data.TensorDataset(torch.tensor(np.arange(len(self.test_dataset))), self.test_label)
        print(f"\nlen_clf_test:{len(clf_dataset)}")
        print(f"\nlen_triplet_test:{len(triplet_dataset)}")
        triplet_loader = trainer.get_dataloader(triplet_dataset, len(triplet_dataset), "test", self.hparams.dataloader_num_workers)
        clf_loader = trainer.get_dataloader(clf_dataset, len(clf_dataset), "test", self.hparams.dataloader_num_workers)
        return CombinedLoader({"triplet": triplet_loader, "clf": clf_loader}, mode=self.loader_mode)

def main():
    parser = trainer.config_parser()
    config_files = parser.parse_args()
    configs = trainer.load_configs(config_files)

    print(configs)

    pl.seed_everything(configs["seed"])
    model = MTL(**configs)
    monitor = "valid_total_loss"
    trainer.generic_train(model, configs, monitor)

if __name__ == "__main__":
    main()
