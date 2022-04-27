# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn
from torchvision import  models
import pytorch_lightning as pl
import trainer, pickle, transforms
import numpy as np
import sys

from omegaconf import OmegaConf as oc
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


sys.path.insert(0, '..')
import evals.embed_evals as evals

class RESN(pl.LightningModule):
    def __init__(self, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.setup_data()

        self.feature_extractor = models.resnet18(pretrained=self.hparams.pretrained)
        # num_features = 1000

        self.pdist = nn.PairwiseDistance()
        self.triplet_loss = nn.TripletMarginLoss()

        if self.hparams.num_class > 2:
            self.criterion = nn.CrossEntropyLoss()
            self.nonlinear = nn.Softmax()
            self.out_dim = self.hparams.num_class
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            self.nonlinear = nn.Sigmoid()
            self.out_dim = 1

        self.embed_dim = self.hparams.embed_dim

    ###### old architectur: final linear layer, d=embed_dim
        # self.fc = nn.ModuleList([nn.Sequential(
        #     nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, 256), 
        #     nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(), nn.Linear(256, self.embed_dim)
        # )])
        # self.classifier = nn.Sequential(
        #     nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Dropout(), nn.Linear(self.embed_dim, self.out_dim)
        # )

    ###### new architectur: no linear layer, d=512
        self.feature_extractor.fc = nn.Identity()
        num_features = 512
        # self.fc = nn.ModuleList([nn.Sequential(nn.Dropout(), nn.Linear(512,512))])
        self.fc = nn.ModuleList([nn.Sequential(nn.Dropout())])
        self.dropout = nn.Dropout()
        self.classifier = nn.Sequential(nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, self.out_dim))


        self.summarize()

    def forward(self, x):
        embeds = self.feature_extractor(x)
        for layer in self.fc: embeds = layer(embeds)
        return embeds

    def clf_loss_acc(self, embeds, labels):
        # embeds = self.dropout(embeds)
        logits = self.classifier(embeds)
        probs = self.nonlinear(logits)
        if self.hparams.num_class < 3:
            labels = labels.type_as(logits).unsqueeze(1)

        clf_loss = self.criterion(logits, labels)
        m = trainer.metrics(probs, labels, num_class=self.hparams.num_class)
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeds = self(x)
        loss, m = self.clf_loss_acc(embeds, y)
        self.log('train_clf_loss', loss)
        self.log('train_clf_acc', m['acc'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeds = self(x)
        loss, m = self.clf_loss_acc(embeds, y)
        self.log('valid_clf_loss', loss, sync_dist=True)
        self.log('valid_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        if self.hparams.num_class < 3: self.log('valid_auc', m['auc'], sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        embeds = self(x)
        loss, m = self.clf_loss_acc(embeds, y)
        self.log('test_clf_loss', loss, sync_dist=True)
        self.log('test_clf_acc', m['acc'], sync_dist=True)
        # return

        triplet_acc = self.test_mixed_triplets()
        self.log('test_triplet_acc', triplet_acc, sync_dist=True)

        knn_acc, ds_acc = self.test_evals()

        df = pd.read_csv("results.csv")
        df = pd.concat([df, pd.DataFrame({"wandb_group": [self.hparams.wandb_group], "wandb_name": [self.hparams.wandb_name],
            "test_clf_acc": [m['acc'].item()], "test_clf_loss": [loss.item()], "test_1nn_acc": [knn_acc], "test_triplet_acc":[triplet_acc.item()], "decision_support": [ds_acc]})], sort=False)
        df.to_csv("results.csv", index=False)

    def test_evals(self):
        train_x = self(self.train_input).cpu().detach().numpy()
        train_y = self.train_label.cpu().detach().numpy()
        test_x = self(self.test_input).cpu().detach().numpy()
        test_y = self.test_label.cpu().detach().numpy()
        knn_acc = evals.get_knn_score(train_x, train_y, test_x, test_y)
        self.log('test_1nn_acc', knn_acc, sync_dist=True)
        
        if self.hparams.syn:
            syn_x_train  = pickle.load(open(self.hparams.train_synthetic,"rb"))
            syn_x_test = pickle.load(open(self.hparams.test_synthetic,"rb"))
            examples = evals.class_1NN_idx(train_x, train_y, test_x, test_y)
            ds_acc = evals.decision_support(syn_x_train, train_y, syn_x_test, test_y, examples, self.hparams.weights, self.hparams.powers)
            self.log('decision support', ds_acc, sync_dist=True)  
        else: ds_acc = 0

        return knn_acc, ds_acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.opt = optimizer
        return optimizer

    def setup_data(self):
        train_transform = transforms.get_transform(self.hparams.transform, aug=True)
        valid_transform = transforms.get_transform(self.hparams.transform, aug=False)
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
        return trainer.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        dataset = self.valid_dataset
        print(f"\n valid:{len(dataset)}")
        return trainer.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)

    def test_dataloader(self):
        dataset = self.test_dataset
        print(f"\n test:{len(dataset)}")
        return trainer.get_dataloader(dataset, len(dataset), "test", self.hparams.dataloader_num_workers)

def main():
    parser = trainer.config_parser()
    config_files = parser.parse_args()
    configs = trainer.load_configs(config_files)

    # wandb_name = "RESN_pretrained" if configs["pretrained"] else "RESN"
    # wandb_name = oc.create({"wandb_name": wandb_name}) 
    # configs = oc.merge(configs, wandb_name)
    print(configs)

    pl.seed_everything(configs["seed"])
    model = RESN(**configs)
    monitor = "valid_clf_loss"
    trainer.generic_train(model, configs, monitor)

if __name__ == "__main__":
    main()
