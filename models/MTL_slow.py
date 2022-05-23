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
import trainer, transforms
from sklearn.metrics.pairwise import euclidean_distances
from pytorch_lightning.trainer.supporters import CombinedLoader
import wandb, time, random,sys
import pandas as pd
sys.path.insert(0, '..')
import evals.embed_evals as evals

class MTL(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.setup_data()

        self.encoder = models.resnet18(pretrained=self.hparams.pretrained)
        num_features = 512

        self.pdist = nn.PairwiseDistance()
        self.triplet_loss = nn.TripletMarginLoss()

        self.criterion = nn.CrossEntropyLoss()
        self.nonlinear = nn.Softmax()

        num_features = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(num_features, self.hparams.embed_dim, bias=False))
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.hparams.embed_dim), nn.ReLU(), nn.Linear(self.hparams.embed_dim, self.hparams.num_class))

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

    def forward(self, inputs):
        inputs = inputs.to(self.device)        
        embeds = self.encoder(inputs)
        return embeds
        
    def clf_loss_acc(self, embeds, labels):
        logits = self.classifier(embeds)
        # probs = self.nonlinear(logits)
        # if self.hparams.num_class < 3:
        #     labels = labels.type_as(logits).unsqueeze(1)
        preds = (logits.argmax(1) == labels).float()
        clf_loss = self.criterion(logits, labels)
        # m = trainer.metrics(probs, labels, num_class=self.hparams.num_class)
        m = {"pred":preds}
        return clf_loss, m
        
    def triplet_loss_acc(self, triplets):
        triplet_loss = self.triplet_loss(*triplets)
        d_ap = self.pdist(triplets[0], triplets[1])
        d_an = self.pdist(triplets[0], triplets[2])
        triplet_acc = (d_ap < d_an).float().mean()
        return triplet_loss, triplet_acc

    def test_mixed_triplets(self):
        triplet_idx = torch.tensor(self.test_triplets).long()
        train_embeds, test_embeds = self(self.train_input), self(self.test_input)
        x1, x2, x3 = test_embeds[triplet_idx[:,0]], train_embeds[triplet_idx[:,1]], train_embeds[triplet_idx[:,2]]
        triplets = (x1, x2, x3)
        return self.triplet_loss_acc(triplets)[1]

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
        # self.log('train_clf_acc', m['acc'], prog_bar=False)
        self.log('train_triplet_loss', triplet_loss)
        self.log('train_triplet_acc', triplet_acc, prog_bar=False)
        self.log('train_total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        triplet_idx = batch["triplet"][0]
        clf_idx, labels = batch["clf"]

        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.mixed_triplets_step(triplet_idx, clf_idx, labels, "valid")

        self.log('valid_clf_loss', clf_loss)
        # self.log('valid_clf_acc', m['acc'], prog_bar=False)
        # self.log('valid_auc', m['auc'], prog_bar=False)
        self.log('valid_triplet_loss', triplet_loss)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=False)
        self.log('valid_total_loss', total_loss)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.test_corrects = torch.zeros(len(self.test_input) )
        triplet_idx = batch["triplet"][0]
        clf_idx, labels = batch["clf"]

        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.mixed_triplets_step(triplet_idx, clf_idx, labels, "test")
        self.test_corrects[clf_idx] = m["pred"].reshape(-1).float().cpu().detach()
        self.log('test_clf_loss', clf_loss)
        # self.log('test_clf_acc', m['acc'], prog_bar=False)
        # self.log('test_auc', m['auc'], prog_bar=False)
        self.log('test_triplet_loss', triplet_loss)
        self.log('test_triplet_acc', triplet_acc, prog_bar=False)
        self.log('test_total_loss', total_loss)
        return triplet_acc

    def embed_dataset(self, dataset):
        self.eval()
        zs, dl = [], torch.utils.data.DataLoader(dataset, batch_size=self.hparams.train_batch_size)
        for x, _ in iter(dl): 
            zs.append(self(x.to(self.device)).cpu())
        return torch.cat(zs)

    def test_epoch_end(self, outputs):
        results = self.eval_knn_ds(
            self.test_dataset, self.train_dataset, self.syn_x_train, self.syn_x_test)
        for k,v in results.items(): self.log(k,v)
        # csv = {
        #     "wandb_project": self.hparams.wandb_project,
        #     "wandb_group": self.hparams.wandb_group,
        #     "wandb_name": self.hparams.wandb_name,
        #     "seed": self.hparams.seed,
        #     "weights": self.hparams.weights,
        #     "embed_dim": self.hparams.embed_dim,
        #     "lamda": self.hparams.lamda,
        #     }
        # csv.update(results)
        # csv = {k:[v] for k,v in csv.items()}
        # if self.hparams.out_csv is not None: out_csv = self.hparams.out_csv 
        # else: out_csv = "out.csv"
        # out_csv = f"results/{out_csv}"
        # if not os.path.isfile(out_csv): df = pd.DataFrame()
        # else: df = pd.read_csv(out_csv)
        # df = pd.concat([df,pd.DataFrame(csv)])
        # df.to_csv(out_csv,index=False)

        # df = pd.read_csv("results/out.csv")
        # df = pd.concat([df,pd.DataFrame(csv)])
        # df.to_csv("results/out.csv",index=False)


    def eval_knn_ds(self, test_ds, train_ds, syn_x_train=None, syn_x_test=None):
        z_train = self.embed_dataset(train_ds).numpy()
        z_test = self.embed_dataset(test_ds).numpy()
        y_train, y_test = self.train_label.numpy(), self.test_label.numpy()

        ## predicted labels
        y_pred = np.array([not y if not m else y for y, m in zip(y_test, self.test_corrects)])
        if self.hparams.model == "TN":
            y_pred = evals.get_knn_score(z_train, y_train, z_test, y_test, metric="preds")


        knn_acc = evals.get_knn_score(z_train, y_train, z_test, y_test)
        results = {"test_1nn_acc":knn_acc}
        if self.hparams.syn:
            to_log = ["NINO_ds_acc", "rNINO_ds_acc", "NIFO_ds_acc"]
            RESN_d512_dir = "../embeds/wv_3d_linear_RESN"

            syn_evals = evals.syn_evals(z_train, y_train, z_test, y_test, y_pred, syn_x_train, syn_x_test, 
            self.hparams.weights, self.hparams.powers, k=1)

            if self.hparams.model != "RESN":
                NI_h2h = []
                NO_h2h = []
                for seed in range(3):
                    RESN_train_512 = pickle.load(open(f"{RESN_d512_dir}/RESN_train_d512_seed{seed}.pkl","rb"))
                    RESN_test_512 = pickle.load(open(f"{RESN_d512_dir}/RESN_test_d512_seed{seed}.pkl","rb"))
                    RESN_pred_512 = pickle.load(open(f"{RESN_d512_dir}/RESN_preds_d512_seed{seed}.pkl","rb"))
                    euc_dist_M = euclidean_distances(RESN_test_512,RESN_train_512)
                    RESN_NINOs = evals.get_NINO(euc_dist_M, y_train, RESN_pred_512, k=1)
                    RESN_NIs = RESN_NINOs[:,0]
                    RESN_NOs = RESN_NINOs[:,1]
                    wins, errs, ties = evals.nn_comparison(syn_x_train, syn_x_test, syn_evals["NINOs"][:,0], RESN_NIs, self.hparams.weights, self.hparams.powers)
                    NI_h2h.append((wins + ties*0.5)/len(y_test))
                    wins, errs, ties = evals.nn_comparison(syn_x_train, syn_x_test, syn_evals["NINOs"][:,1], RESN_NOs, self.hparams.weights, self.hparams.powers)
                    NO_h2h.append((wins + ties*0.5)/len(y_test))
                results["NI_h2h"] = np.array(NI_h2h).mean()
                results["NO_h2h"] = np.array(NO_h2h).mean()

            for eval in to_log: results[eval] = syn_evals[eval]
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.opt = optimizer
        return optimizer

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
