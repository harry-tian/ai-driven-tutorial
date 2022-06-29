# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn
from torchvision import  models
import pytorch_lightning as pl
import trainer, pickle, transforms
import numpy as np
import sys, pathlib

# from omegaconf import OmegaConf as oc
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '..')
import evals.embed_evals as evals

class RESN(pl.LightningModule):
    def __init__(self, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = models.resnet18(pretrained=self.hparams.pretrained)
        num_features = 512

        self.criterion = nn.CrossEntropyLoss()
        self.nonlinear = nn.Softmax()

        self.embed_dim = self.hparams.embed_dim

        self.encoder.fc = nn.Sequential(nn.Linear(num_features, self.embed_dim, bias=False))

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.hparams.num_class)
        )

        self.summarize()

    def forward(self, inputs):        
        embeds = self.encoder(inputs)
        return embeds

    def clf_loss_acc(self, embeds, labels):
        logits = self.classifier(embeds)

        clf_loss = self.criterion(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        return clf_loss, acc
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        embeds = self(x)
        loss, acc = self.clf_loss_acc(embeds, y)
        self.log('train_clf_loss', loss)
        self.log('train_clf_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeds = self(x)
        loss, acc = self.clf_loss_acc(embeds, y)
        self.log('valid_clf_loss', loss)
        self.log('valid_clf_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        embeds = self(x)
        loss, acc = self.clf_loss_acc(embeds, y)
        self.log('test_clf_loss', loss)
        self.log('test_clf_acc', acc)

    def embed_dataset(self, dataloader):
        embed = [self(x.to(self.device)).cpu() for x, _ in iter(dataloader)]
        return np.array(torch.cat(embed))

    def save_embeds(self):
        self.eval()
        z_train = self.embed_dataset(self.train_dataloader())
        z_valid = self.embed_dataset(self.val_dataloader())
        z_test = self.embed_dataset(self.test_dataloader())

        embed_dir = "../data/embeds/" + self.hparams.resn_embed_dir
        for fold, emb in zip(['train', 'valid', 'test'], [z_train, z_valid, z_test]):
            path = '/'.join([embed_dir, self.hparams.dataset])
            print("Saving embeds at:", path)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

            name = f"{self.hparams.dataset}_{fold}_emb{self.embed_dim}.pkl"
            pickle.dump(emb, open(path + '/' + name, 'wb'))
    
    def test_epoch_end(self, test_step_outputs):
        self.save_embeds()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.opt = optimizer
        return optimizer

    def train_dataloader(self):
        transform = transforms.get_transform(self.hparams.transform, aug=True)
        dataset = torchvision.datasets.ImageFolder(self.hparams.train_dir, transform=transform)

        print(f"\n train:{len(dataset)}")
        return trainer.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        transform = transforms.get_transform(self.hparams.transform, aug=False)
        dataset = torchvision.datasets.ImageFolder(self.hparams.valid_dir, transform=transform)

        print(f"\n valid:{len(dataset)}")
        return trainer.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)

    def test_dataloader(self):
        transform = transforms.get_transform(self.hparams.transform, aug=False)
        dataset = torchvision.datasets.ImageFolder(self.hparams.test_dir, transform=transform)

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
