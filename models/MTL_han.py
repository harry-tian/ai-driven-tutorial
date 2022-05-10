# -*- coding: utf-8 -*-
import pathlib
import sys, pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import  models
import pytorch_lightning as pl
import trainer, transforms

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
        if self.embed_dim:
            self.encoder.fc = nn.Sequential(nn.Linear(num_features, self.embed_dim, bias=False))
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.hparams.num_class))
        else:
            self.encoder.fc = nn.Identity()
            self.classifier = nn.Sequential(nn.Linear(num_features, self.hparams.num_class))
        self.clf_criterion = nn.CrossEntropyLoss()
        self.criterion = nn.TripletMarginLoss()
        if 'profiler' in kwargs:
            self.profiler = kwargs['profiler']

    def setup_data(self):
        train_transform = transforms.get_transform(self.hparams.transform, aug=True)
        valid_transform = transforms.get_transform(self.hparams.transform, aug=False)
        self.train_dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=train_transform)
        self.valid_dataset = torchvision.datasets.ImageFolder(self.hparams.valid_dir, transform=valid_transform)
        self.test_dataset = torchvision.datasets.ImageFolder(self.hparams.test_dir, transform=valid_transform)
        self.ref_dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=valid_transform)
        self.in_memeory_dataset = False
        if self.hparams.transform == 'wv' or self.hparams.transform == 'bm':
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
        loader = torch.utils.data.DataLoader(dataset, len(dataset), num_workers=1)
        batch = next(iter(loader))
        return batch[0].to(self.device), batch[1].to(self.device)

    def sample_train_clf_trips(self, x_idx, ys):
        classes = torch.unique(ys)
        k_trips = self.hparams.triplet_batch_size // len(classes)
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

    def sample_xs_ys(self, dataset, x_idx=None, aug=False):
        if type(dataset) == tuple:
            data = (dataset[0][x_idx], dataset[1][x_idx]) if x_idx is not None else dataset
        else:
            x_idx = x_idx.to(dataset[0][0].device)
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
            triplet_acc = self.trips_acc(ta, tp, tn)
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
        if self.valid_embeds is None: 
            embed_dim = self.embed_dim if self.embed_dim else self.classifier[0].in_features
            len_valid_dataset = len(self.valid_dataset[0]) if self.in_memeory_dataset else len(self.valid_dataset)
            self.valid_embeds = torch.zeros(len_valid_dataset, embed_dim)
        valid_batch, valid_idx = torch.unique(batch[:, 0], sorted=False, return_inverse=True)
        train_batch, train_idx = torch.unique(batch[:, 1:], sorted=False, return_inverse=True)
        x_valid, y_valid = self.sample_xs_ys(self.valid_dataset, valid_batch)
        x_valid, y_valid = x_valid.to(self.device), y_valid.to(self.device)
        x_train, y_train = self.sample_xs_ys(self.ref_dataset, train_batch)
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        z_valid, z_train= self(x_valid), self(x_train)
        self.valid_embeds[valid_batch.cpu()] = z_valid[valid_batch].cpu()
        ta, tp, tn = z_valid[valid_idx], z_train[train_idx[:, 0]], z_train[train_idx[:, 1]]
        triplet_loss = self.criterion(ta, tp, tn)
        triplet_acc = self.trips_acc(ta, tp, tn)
        return triplet_loss, triplet_acc

    def validation_epoch_end(self, validation_step_outputs):
        all_triplet_loss, all_triplet_acc = zip(*validation_step_outputs)
        if len(all_triplet_loss) > 1:
            triplet_loss = torch.cat(list(all_triplet_loss)).sum() / len(all_triplet_loss)
            triplet_acc = torch.cat(list(all_triplet_acc)).sum() / len(all_triplet_loss)
        else:
            triplet_loss, triplet_acc = all_triplet_loss[0], all_triplet_acc[0]
        total_loss = (1 - self.hparams.lamda) * triplet_loss
        z_valid = self.valid_embeds
        _, y_valid = self.sample_xs_ys(self.valid_dataset)
        z_valid, y_valid = z_valid.to(self.device), y_valid.to(self.device)
        logits = self.classifier(z_valid)
        clf_loss = self.clf_criterion(logits, y_valid)
        clf_acc = (logits.argmax(1) == y_valid).float().mean()
        total_loss += self.hparams.lamda * clf_loss
        self.valid_embeds = None
        knn_acc, ds_acc = self.eval_knn_ds(
            self.valid_dataset, self.ref_dataset, self.syn_x_train, self.syn_x_valid, status='valid')
        self.log('valid_clf_loss', clf_loss)
        self.log('valid_clf_acc', clf_acc, prog_bar=True)
        self.log('valid_triplet_loss', triplet_loss)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True)
        self.log('valid_total_loss', total_loss, prog_bar=True)
        if knn_acc:
            self.log('valid_1nn_acc', knn_acc)
        if ds_acc:
            self.log('valid_decision_support', ds_acc)

    def test_step(self, batch, batch_idx):
        if self.test_embeds is None: 
            embed_dim = self.embed_dim if self.embed_dim else self.classifier[0].in_features
            len_test_dataset = len(self.test_dataset[0]) if self.in_memeory_dataset else len(self.test_dataset)
            self.test_embeds = torch.zeros(len_test_dataset, embed_dim)
        test_batch, test_idx = torch.unique(batch[:, 0], sorted=False, return_inverse=True)
        train_batch, train_idx = torch.unique(batch[:, 1:], sorted=False, return_inverse=True)
        x_test, y_test = self.sample_xs_ys(self.test_dataset, test_batch)
        x_test, y_test = x_test.to(self.device), y_test.to(self.device)
        x_train, y_train = self.sample_xs_ys(self.ref_dataset, train_batch)
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        z_test, z_train= self(x_test), self(x_train)
        self.test_embeds[test_idx.cpu()] = z_test[test_idx].cpu()
        ta, tp, tn = z_test[test_idx], z_train[train_idx[:, 0]], z_train[train_idx[:, 1]]
        triplet_loss = self.criterion(ta, tp, tn)
        triplet_acc = self.trips_acc(ta, tp, tn)
        return triplet_loss, triplet_acc

    def test_epoch_end(self, test_step_outputs):
        all_triplet_loss, all_triplet_acc = zip(*test_step_outputs)
        if len(all_triplet_loss) > 1:
            triplet_loss = torch.cat(list(all_triplet_loss)).sum() / len(all_triplet_loss)
            triplet_acc = torch.cat(list(all_triplet_acc)).sum() / len(all_triplet_loss)
        else:
            triplet_loss, triplet_acc = all_triplet_loss[0], all_triplet_acc[0]
        total_loss = (1 - self.hparams.lamda) * triplet_loss
        z_test = self.test_embeds
        _, y_test = self.sample_xs_ys(self.test_dataset)
        z_test, y_test = z_test.to(self.device), y_test.to(self.device)
        logits = self.classifier(z_test)
        clf_loss = self.clf_criterion(logits, y_test)
        clf_acc = (logits.argmax(1) == y_test).float().mean()
        total_loss += self.hparams.lamda * clf_loss
        knn_acc, ds_acc = self.eval_knn_ds(
            self.test_dataset, self.ref_dataset, self.syn_x_train, self.syn_x_test, status='test')
        self.log('test_clf_loss', clf_loss)
        self.log('test_clf_acc', clf_acc, prog_bar=True)
        self.log('test_triplet_loss', triplet_loss)
        self.log('test_triplet_acc', triplet_acc, prog_bar=True)
        self.log('test_total_loss', total_loss, prog_bar=True)
        if knn_acc:
            self.log('test_1nn_acc', knn_acc)
        if ds_acc:
            self.log('test_decision_support', ds_acc)
        if self.hparams.embeds_output_dir is not None:
            self.save_embeds()

    def save_embeds(self):
        self.eval()
        if self.in_memeory_dataset:
            z_train = self(self.ref_dataset[0].to(self.device)).cpu().detach().numpy()
            z_valid = self(self.valid_dataset[0].to(self.device)).cpu().detach().numpy()
            z_test = self(self.test_dataset[0].to(self.device)).cpu().detach().numpy()
        else:
            z_train, z_valid, z_test = [], [], []
            tdl = torch.utils.data.DataLoader(self.ref_dataset, batch_size=self.hparams.train_batch_size)
            vdl = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.hparams.train_batch_size)
            sdl = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.hparams.train_batch_size)
            for x, _ in iter(tdl): z_train.append(self(x).cpu().detach().numpy())
            for x, _ in iter(vdl): z_valid.append(self(x).cpu().detach().numpy())
            for x, _ in iter(sdl): z_test.append(self(x).cpu().detach().numpy())
            z_train = torch.cat(z_train)
            z_valid = torch.cat(z_valid)
            z_test = torch.cat(z_test)
        for fold, emb in zip(['train', 'valid', 'test'], [z_train, z_valid, z_test]):
            name = f"MTL_han_{fold}_emb{self.embed_dim}_s{self.hparams.seed}.pkl"
            path = '/'.join([
                self.hparams.embeds_output_dir, 
                self.hparams.wandb_project,
                self.hparams.wandb_group,
                self.hparams.wandb_name.split('s')[0]
                ])
            print("Saving embeds at:", path)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            pickle.dump(emb, open(path + '/' + name, 'wb'))

    def eval_knn_ds(self, test_ds, train_ds, syn_x_train=None, syn_x_test=None, status=None):
        x_train, y_train = self.sample_xs_ys(train_ds)
        x_test, y_test = self.sample_xs_ys(test_ds)
        z_train = self.encoder(x_train.to(self.device)).cpu().detach().numpy()
        z_test = self.encoder(x_test.to(self.device)).cpu().detach().numpy()
        y_train, y_test = y_train.numpy(), y_test.numpy()
        knn_acc = evals.get_knn_score(z_train, y_train, z_test, y_test)
        ds_acc = None
        if self.hparams.syn:
            results = evals.syn_evals(z_train, y_train, z_test, y_test, syn_x_train, syn_x_test, 
            self.hparams.weights, self.hparams.powers)
            to_log = ["NINO_ds_acc", "rNINO_ds_acc", "NIFO_ds_acc"]
            to_print = ["NINO_ds_err", "rNINO_ds_err", "NIFO_ds_err", "NIs"]
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
            batch_size=self.hparams.triplet_batch_size, 
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

    model = MTL(profiler=profiler, **configs)
    monitor = "valid_total_loss"
    trainer.generic_train(model, configs, monitor, profiler=profiler, 
        num_sanity_val_steps=4)


if __name__ == "__main__":
    main()
