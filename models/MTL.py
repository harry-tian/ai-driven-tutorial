# -*- coding: utf-8 -*-
import pathlib, time, random
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

from sklearn.metrics.pairwise import euclidean_distances
sys.path.insert(0, '..')
import evals.embed_evals as evals

import warnings
warnings.filterwarnings("ignore")


class MTL(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters() 
        self.setup_data()
        self.encoder = models.resnet18(
            pretrained=self.hparams.pretrained, zero_init_residual=not self.hparams.pretrained)
        num_features = self.encoder.fc.weight.shape[1]
        self.embed_dim = self.hparams.embed_dim
        if self.embed_dim!=512:
            self.encoder.fc = nn.Sequential(nn.Linear(num_features, self.embed_dim, bias=False))
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.hparams.num_class))
        else:
            self.encoder.fc = nn.Identity()
            self.classifier = nn.Sequential(nn.Linear(num_features, self.hparams.num_class))
        self.clf_criterion = nn.CrossEntropyLoss()
        self.criterion = nn.TripletMarginLoss()
        self.summarize()

    def setup_data(self):
        train_transform = transforms.get_transform(self.hparams.transform, aug=True)
        valid_transform = transforms.get_transform(self.hparams.transform, aug=False)
        self.train_dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=train_transform)
        self.valid_dataset = torchvision.datasets.ImageFolder(self.hparams.valid_dir, transform=valid_transform)
        self.test_dataset = torchvision.datasets.ImageFolder(self.hparams.test_dir, transform=valid_transform)
        self.ref_dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=valid_transform)
        self.in_memeory_dataset = False
        transform = self.hparams.transform
        if transform == 'wv' or transform == 'bm' or transform == 'wv_3d':
            self.in_memeory_dataset = True
            self.train_dataset = self.load_dataset_to_memory(self.train_dataset)
            self.valid_dataset = self.load_dataset_to_memory(self.valid_dataset)
            self.test_dataset = self.load_dataset_to_memory(self.test_dataset)
            self.ref_dataset = self.train_dataset
        if self.hparams.transform == 'bm':
            affine = {}
            degree, translate, scale = 10, 0.1, 0.1
            affine["degrees"] = degree
            affine["translate"] = (translate, translate)
            affine["scale"] = (1 - scale, 1 + scale)
            self.augmentation = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomAffine(**affine),
            ])
        self.train_triplets = np.array(pickle.load(open(self.hparams.train_triplets, "rb")))
        self.valid_triplets = np.array(pickle.load(open(self.hparams.valid_triplets, "rb")))
        self.test_triplets = np.array(pickle.load(open(self.hparams.test_triplets, "rb")))
        self.syn_x_train, self.syn_x_valid, self.syn_x_test = None, None, None
        if self.hparams.syn:
            self.syn_x_train = pickle.load(open(self.hparams.train_synthetic, "rb"))
            self.syn_x_valid = pickle.load(open(self.hparams.valid_synthetic, "rb"))
            self.syn_x_test = pickle.load(open(self.hparams.test_synthetic, "rb"))
        self.valid_embeds = None
        self.test_embeds = None

    def load_dataset_to_memory(self, dataset):
        num_workers = 4 if len(dataset) > 500 else 1
        loader = torch.utils.data.DataLoader(dataset, len(dataset), num_workers=num_workers)
        batch = next(iter(loader))
        return batch[0].to(self.device), batch[1].to(self.device)

    def sample_xs_ys(self, dataset, x_idx=None, aug=False):
        if type(dataset) == tuple:
            data = (dataset[0][x_idx], dataset[1][x_idx]) if x_idx is not None else dataset
        else:
            x_idx = x_idx.to(dataset[0][0].device) if x_idx is not None else None
            subset = torch.utils.data.Subset(dataset, x_idx) if x_idx is not None else dataset
            loader = torch.utils.data.DataLoader(
                subset, len(subset), num_workers=1)
            data = next(iter(loader))
        if aug:
            x = self.augmentation(data[0])
            data = (x, data[1])
        return data
    
    def forward(self, inputs):
        embeds = self.encoder(inputs)
        return embeds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        clf_idx = torch.LongTensor([]).to(self.device)
        if self.hparams.lamda > 0:
            len_train_dataset = len(self.train_dataset[0]) if self.in_memeory_dataset else len(self.train_dataset)
            clf_idx = torch.randperm(len_train_dataset)[:self.hparams.train_batch_size].to(self.device)
            uniques = clf_idx
            _, batch_clf_idx = torch.unique(clf_idx, sorted=False, return_inverse=True)
        if self.hparams.lamda < 1:
            trip_shape, trip_numel = batch.shape, batch.numel()
            all_flatten = torch.cat([batch.flatten(), clf_idx])
            uniques, inverse = torch.unique(all_flatten, sorted=False, return_inverse=True)
            batch_trip_idx = inverse[:trip_numel].view(trip_shape)
            batch_clf_idx = inverse[trip_numel+1:]
        xs, ys = self.sample_xs_ys(self.train_dataset, uniques, aug=self.hparams.aug)
        xs, ys = xs.to(self.device), ys.to(self.device)
        zs = self(xs)
        total_loss = torch.zeros(1).to(self.device)
        if self.hparams.lamda < 1:
            ta, tp, tn = zs[batch_trip_idx[:, 0]], zs[batch_trip_idx[:, 1]], zs[batch_trip_idx[:, 2]]
            triplet_loss = self.criterion(ta, tp, tn)
            total_loss += (1 - self.hparams.lamda) * triplet_loss
            triplet_acc = self.trips_corr(ta, tp, tn).mean()
            self.log('train_triplet_loss', triplet_loss)
            self.log('train_triplet_acc', triplet_acc, prog_bar=True)
        if self.hparams.lamda > 0:
            logits = self.classifier(zs[batch_clf_idx])
            clf_loss = self.clf_criterion(logits, ys[batch_clf_idx])
            clf_acc = (logits.argmax(1) == ys[batch_clf_idx]).float().mean()
            total_loss += self.hparams.lamda * clf_loss
            self.log('train_clf_loss', clf_loss)
            self.log('train_clf_acc', clf_acc, prog_bar=True)
        self.log('train_total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: 
            len_valid_dataset = len(self.valid_dataset[0]) if self.in_memeory_dataset else len(self.valid_dataset)
            self.valid_losses = torch.zeros(len_valid_dataset)
            self.valid_corres = torch.zeros(len_valid_dataset)
        batch = batch.cpu()
        valid_batch, valid_idx = torch.unique(batch[:, 0], sorted=False, return_inverse=True)
        train_batch, train_idx = torch.unique(batch[:, 1:], sorted=False, return_inverse=True)
        x_valid, y_valid = self.sample_xs_ys(self.valid_dataset, valid_batch)
        x_valid, y_valid = x_valid.to(self.device), y_valid.to(self.device)
        x_train, y_train = self.sample_xs_ys(self.ref_dataset, train_batch)
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        z_valid, z_train= self(x_valid), self(x_train)
        ta, tp, tn = z_valid[valid_idx], z_train[train_idx[:, 0]], z_train[train_idx[:, 1]]
        triplet_loss = F.triplet_margin_loss(ta, tp, tn, reduction='none')
        triplet_corr = self.trips_corr(ta, tp, tn)
        logits = self.classifier(z_valid)
        clf_loss = F.cross_entropy(logits, y_valid, reduction='none')
        clf_corr = (logits.argmax(1) == y_valid).float()
        self.valid_losses[valid_batch] = clf_loss.cpu()
        self.valid_corres[valid_batch] = clf_corr.cpu()
        return triplet_loss, triplet_corr

    def validation_epoch_end(self, validation_step_outputs):
        all_triplet_loss, all_triplet_corr = zip(*validation_step_outputs)
        if len(all_triplet_loss) > 1:
            triplet_loss = torch.cat(all_triplet_loss).mean()
            triplet_acc = torch.cat(all_triplet_corr).mean()
        else:
            triplet_loss, triplet_acc = all_triplet_loss[0], all_triplet_corr[0].mean()
        total_loss = (1 - self.hparams.lamda) * triplet_loss
        clf_loss = self.valid_losses.mean()
        clf_acc = self.valid_corres.mean()
        total_loss += self.hparams.lamda * clf_loss
        self.log('valid_clf_loss', clf_loss)
        self.log('valid_clf_acc', clf_acc, prog_bar=True)
        self.log('valid_triplet_loss', triplet_loss)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True)
        self.log('valid_total_loss', total_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: 
            len_test_dataset = len(self.test_dataset[0]) if self.in_memeory_dataset else len(self.test_dataset)
            self.test_losses = torch.zeros(len_test_dataset)
            self.test_corres = torch.zeros(len_test_dataset)
        batch = batch.cpu()
        test_batch, test_idx = torch.unique(batch[:, 0], sorted=False, return_inverse=True)
        train_batch, train_idx = torch.unique(batch[:, 1:], sorted=False, return_inverse=True)
        x_test, y_test = self.sample_xs_ys(self.test_dataset, test_batch)
        x_test, y_test = x_test.to(self.device), y_test.to(self.device)
        x_train, y_train = self.sample_xs_ys(self.ref_dataset, train_batch)
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        z_test, z_train= self(x_test), self(x_train)
        ta, tp, tn = z_test[test_idx], z_train[train_idx[:, 0]], z_train[train_idx[:, 1]]
        triplet_loss = F.triplet_margin_loss(ta, tp, tn, reduction='none')
        triplet_corr = self.trips_corr(ta, tp, tn)
        logits = self.classifier(z_test)
        clf_loss = F.cross_entropy(logits, y_test, reduction='none')
        clf_corr = (logits.argmax(1) == y_test).float()
        self.test_losses[test_batch] = clf_loss.cpu()
        self.test_corres[test_batch] = clf_corr.cpu()
        return triplet_loss, triplet_corr

    def test_epoch_end(self, test_step_outputs):
        all_triplet_loss, all_triplet_corr = zip(*test_step_outputs)
        if len(all_triplet_loss) > 1:
            triplet_loss = torch.cat(all_triplet_loss).mean()
            triplet_acc = torch.cat(all_triplet_corr).mean()
        else:
            triplet_loss, triplet_acc = all_triplet_loss[0], all_triplet_corr[0].mean()
        total_loss = (1 - self.hparams.lamda) * triplet_loss
        clf_loss = self.test_losses.mean()
        clf_acc = self.test_corres.mean()
        total_loss += self.hparams.lamda * clf_loss
        results = self.eval_knn_ds(
            self.test_dataset, self.ref_dataset, self.syn_x_train, self.syn_x_test, status='test')
        for k,v in results.items(): self.log(k,v)
        self.log('test_clf_loss', clf_loss)
        self.log('test_clf_acc', clf_acc, prog_bar=True)
        self.log('test_triplet_loss', triplet_loss)
        self.log('test_triplet_acc', triplet_acc, prog_bar=True)
        self.log('test_total_loss', total_loss, prog_bar=True)

        csv = {
            "wandb_project": self.hparams.wandb_project,
            "wandb_group": self.hparams.wandb_group,
            "wandb_name": self.hparams.wandb_name,
            "seed": self.hparams.seed,
            "weights": self.hparams.weights,
            "embed_dim": self.hparams.embed_dim,
            "lamda": self.hparams.lamda,
            "test_clf_acc": clf_acc.cpu().detach().numpy(),
            "test_triplet_acc": triplet_acc.cpu().detach().numpy(),
            }
        csv.update(results)
        csv = {k:[v] for k,v in csv.items()}
        if self.hparams.out_csv is not None: out_csv = self.hparams.out_csv 
        else: out_csv = "out.csv"
        out_csv = f"results/{out_csv}"
        if not os.path.isfile(out_csv): df = pd.DataFrame()
        else: df = pd.read_csv(out_csv)
        df = pd.concat([df,pd.DataFrame(csv)])
        df.to_csv(out_csv,index=False)
        
        df = pd.read_csv("results/out.csv")
        df = pd.concat([df,pd.DataFrame(csv)])
        df.to_csv("results/out.csv",index=False)

        if self.hparams.embeds_output_dir is not None:
            self.save_embeds()

    def embed_dataset(self, dataset):
        self.eval()
        dataset = torch.utils.data.TensorDataset(*dataset) if self.in_memeory_dataset else dataset
        zs, dl = [], torch.utils.data.DataLoader(dataset, batch_size=self.hparams.train_batch_size)
        for x, _ in iter(dl): 
            zs.append(self(x.to(self.device)).cpu())
        return torch.cat(zs)

    def save_embeds(self):
        self.eval()
        datasets = [self.ref_dataset, self.valid_dataset, self.test_dataset]
        z_train, z_valid, z_test = [self.embed_dataset(ds) for ds in datasets]
        for fold, emb in zip(['train', 'valid', 'test'], [z_train, z_valid, z_test]):
            name = f"{self.hparams.wandb_name}_{fold}_d{self.hparams.embed_dim}_seed{self.hparams.seed}.pkl"
            path = '/'.join([
                self.hparams.embeds_output_dir, 
                self.hparams.wandb_project,
                self.hparams.wandb_group])
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            f_name = path + '/' + name
            print("Saving embeds at:", f_name)
            pickle.dump(emb, open(f_name, 'wb'))

    def eval_knn_ds(self, test_ds, train_ds, syn_x_train=None, syn_x_test=None, status=None):
        _, y_train = self.sample_xs_ys(train_ds)
        _, y_test = self.sample_xs_ys(test_ds)
        z_train = self.embed_dataset(train_ds).numpy()
        z_test = self.embed_dataset(test_ds).numpy()
        y_train, y_test = y_train.numpy(), y_test.numpy()
        knn_acc = evals.get_knn_score(z_train, y_train, z_test, y_test)
        results = {"test_1nn_acc":knn_acc}
        if self.hparams.syn:
            to_log = ["NINO_ds_acc", "rNINO_ds_acc", "NIFO_ds_acc"]
            to_print = ["NINO_ds_err", "rNINO_ds_err", "NIFO_ds_err", "NIs"]
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

    def trips_corr(self, a, p, n):
        dap = F.pairwise_distance(a, p)
        dan = F.pairwise_distance(a, n)
        return (dap < dan).float()

    def train_dataloader(self):
        triplets = torch.Tensor(self.train_triplets).long()
        print(f"\n len_train: {len(triplets)}")
        triplet_loader = torch.utils.data.DataLoader(
            triplets, 
            batch_size=self.hparams.triplet_batch_size, 
            num_workers=self.hparams.dataloader_num_workers,
            drop_last=True, shuffle=True)
        return triplet_loader
        
    def val_dataloader(self):
        triplets = torch.Tensor(self.valid_triplets).long()
        print(f"\n len_valid: {len(triplets)}")
        triplet_loader = torch.utils.data.DataLoader(
            triplets, 
            batch_size=self.hparams.triplet_batch_size, 
            num_workers=self.hparams.dataloader_num_workers)
        return triplet_loader

    def test_dataloader(self):
        triplets = torch.Tensor(self.test_triplets).long()
        print(f"\n len_test: {len(triplets)}")
        triplet_loader = torch.utils.data.DataLoader(
            triplets, 
            batch_size=self.hparams.triplet_batch_size, 
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

    model = MTL(profiler=profiler, **configs)
    monitor = "valid_total_loss"
    trainer.generic_train(model, configs, monitor, profiler=profiler)


if __name__ == "__main__":
    main()
