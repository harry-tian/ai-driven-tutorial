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
warnings.filterwarnings("ignore")

sys.path.insert(0, '..')
import evals.embed_evals as evals

class RESN(pl.LightningModule):
    def __init__(self, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.setup_data()

        self.feature_extractor = models.resnet18(pretrained=self.hparams.pretrained)
        num_features = 512

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
        self.feature_extractor.fc = nn.Identity()

    ###### old architectur: final linear layer, d=embed_dim
        self.fc = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, self.embed_dim), 
            # nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(), nn.Linear(256, self.embed_dim)
        )])
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Dropout(), nn.Linear(self.embed_dim, self.out_dim)
        )

    ###### new architectur: no linear layer, d=512
        # num_features = 512
        # self.fc = nn.ModuleList([nn.Sequential(nn.Dropout())])
        # self.classifier = nn.Sequential(nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, self.out_dim))

        self.summarize()

    def forward(self, x):
        embeds = self.feature_extractor(x)
        for layer in self.fc: embeds = layer(embeds)
        return embeds

    def clf_loss_acc(self, embeds, labels):
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

    def valid_ds(self):
        triplet_idx = self.valid_triplets
        y_train = self.train_label.cpu().detach().numpy()
        y_valid = self.valid_label.cpu().detach().numpy()
        ds_triplets = []
        for triplet in triplet_idx:
            a, p, n = triplet[0], triplet[1], triplet[2]
            if y_train[p] != y_train[n]:
                temp = [a,p,n] if y_valid[a] == y_train[p] else [a,n,p]
                ds_triplets.append(temp)

        triplet_idx = torch.tensor(ds_triplets).long()
        train_embeds, valid_embeds = self(self.train_input), self(self.valid_input)
        x1, x2, x3 = valid_embeds[triplet_idx[:,0]], train_embeds[triplet_idx[:,1]], train_embeds[triplet_idx[:,2]]
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

        triplet_acc = self.test_mixed_triplets()
        self.log('test_triplet_acc', triplet_acc, sync_dist=True)

    def test_epoch_end(self, outputs):
        z_train = self(self.train_input).cpu().detach().numpy()
        y_train = self.train_label.detach().numpy()
        z_test = self(self.test_input).cpu().detach().numpy()
        y_test = self.test_label.detach().numpy()

        knn_acc = evals.get_knn_score(z_train, y_train, z_test, y_test)
        self.log('test_1nn_acc', knn_acc)

        if self.hparams.syn: 
            syn_x_train  = pickle.load(open(self.hparams.train_synthetic,"rb"))
            syn_x_test = pickle.load(open(self.hparams.test_synthetic,"rb"))
            results = evals.syn_evals(z_train, y_train, z_test, y_test, syn_x_train, syn_x_test, 
            self.hparams.weights, self.hparams.powers)

            to_log = ["NINO_ds_acc", "rNINO_ds_acc", "NIFO_ds_acc"]
            to_print = ["NINO_ds_err","rNINO_ds_err","NIFO_ds_err","NIs"]
            # to_print = ["NINO_ds_acc", "NIFO_ds_acc"]
            for key in to_log: self.log(key, results[key])
            for key in to_print: print(f"\n{key}: {results[key]}")

        # knn_acc, ds_acc, ds_err = self.test_evals()
        # df = pd.read_csv("results.csv")
        # df = pd.concat([df, pd.DataFrame({"wandb_group": [self.hparams.wandb_group], "wandb_name": [self.hparams.wandb_name],
        #     "test_clf_acc": [m['acc'].item()], "test_clf_loss": [loss.item()], "test_1nn_acc": [knn_acc], "test_triplet_acc":[triplet_acc.item()], "decision_support": [ds_acc]})], sort=False)
        # df.to_csv("results.csv", index=False)

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
        self.train_label = torch.tensor(np.array([data[1] for data in self.train_dataset]))#.cuda()
        self.valid_label = torch.tensor(np.array([data[1] for data in self.valid_dataset]))#.cuda()
        self.test_label = torch.tensor(np.array([data[1] for data in self.test_dataset]))#.cuda()

        self.train_triplets = np.array(pickle.load(open(self.hparams.train_triplets, "rb")))
        self.valid_triplets = np.array(pickle.load(open(self.hparams.valid_triplets, "rb")))
        self.test_triplets = np.array(pickle.load(open(self.hparams.test_triplets, "rb")))

        # train_idx = np.random.choice(len(self.train_triplets), 2400, replace=False)
        # self.train_triplets = self.train_triplets[train_idx]
        # valid_idx = np.random.choice(len(self.valid_triplets), 800, replace=False)
        # self.test_triplets = self.test_triplets[valid_idx]
        # self.valid_triplets = self.valid_triplets[valid_idx]

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

    print(configs)

    pl.seed_everything(configs["seed"])
    model = RESN(**configs)
    monitor = "valid_clf_loss"
    trainer.generic_train(model, configs, monitor)

if __name__ == "__main__":
    main()


    # def test_evals(self):
    #     z_train = self(self.train_input)
    #     y_train = self.train_label
    #     z_test = self(self.test_input)
    #     y_test = self.test_label
    #     # _, m = self.clf_loss_acc(z_train, y_train)
    #     # train_preds = m["pred"].cpu().detach().numpy()
    #     # _, m = self.clf_loss_acc(z_test, y_test)
    #     # test_preds = m["pred"].cpu().detach().numpy()

    #     z_train = z_train.cpu().detach().numpy()
    #     y_train = y_train.detach().numpy()
    #     z_test = z_test.cpu().detach().numpy()
    #     y_test = y_test.detach().numpy()


    #     # knn_acc = evals.get_knn_score(train_x, train_y, test_x, test_y)
    #     # self.log('test_1nn_acc', knn_acc, sync_dist=True)

    #     # # valid_ds = self.valid_ds()
    #     # # self.log('valid_decision_support', valid_ds, sync_dist=True)
        
    #     # if self.hparams.syn:
    #     #     syn_x_train  = pickle.load(open(self.hparams.train_synthetic,"rb"))
    #     #     syn_x_test = pickle.load(open(self.hparams.test_synthetic,"rb"))

    #     #     # examples = evals.NINO(train_x, train_y, test_x, test_y)
    #     #     # ds_acc, ds_err = evals.decision_support(syn_x_train, train_y, syn_x_test, test_y, examples, self.hparams.weights, self.hparams.powers)
    #     #     # self.log('decision support', ds_acc, sync_dist=True)  
    #     #     # print(f"\ndecision support errors{ds_err}")
    #     #     # print(examples)

    #     #     examples = evals.NIFO(train_x, train_y, test_x, test_preds)
    #     #     # print(examples)
    #     #     ds_acc, ds_err = evals.decision_support(syn_x_train, train_y, syn_x_test, test_y, examples, self.hparams.weights, self.hparams.powers)
    #     #     self.log('decision support', ds_acc, sync_dist=True)  
    #     #     print(f"\ndecision support errors{ds_err}")
    #     # else:  ds_acc, ds_err = 0, []

    #     # print(f"\ntest_predictions{test_preds}")
    #     return knn_acc, ds_acc, ds_err
