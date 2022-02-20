import torch
import argparse
import os
from pathlib import Path
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pathlib import Path
from pytorch_lightning import loggers as pl_loggers
from torchvision import models
from torch import nn
import time
import shutil
from pytorch_lightning.loggers import WandbLogger
import wandb

class RESN_TN(pl.LightningModule):
    def __init__(self,**config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = models.resnet18(pretrained=True)
        num_features = 1000

        self.embed_dim = self.hparams.embed_dim
        self.triplet_loss = nn.TripletMarginLoss()
        self.pdist = nn.PairwiseDistance()

        self.hidden_size = self.hparams.hidden_size
        self.fc = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, self.hidden_size), 
            nn.BatchNorm1d(self.hidden_size), nn.ReLU(), nn.Dropout(), nn.Linear(self.hidden_size, self.embed_dim)
        )])

        self.summarize()

    def embed(self, x):
        embeds = self.feature_extractor(x)
        # for layer in self.fc:
        #     embeds = layer(embeds)
        return embeds

    def forward(self, **inputs):
        raise NotImplementedError

    def triplet_loss_acc(self, triplet_idx, batch_idx=0):
        triplets = self(triplet_idx, batch_idx)
        
        triplet_loss = self.triplet_loss(*triplets)
        with torch.no_grad():
            d_ap = self.pdist(triplets[0], triplets[1])
            d_an = self.pdist(triplets[0], triplets[2])
            triplet_acc = (d_ap < d_an).float().mean()

        return triplet_loss, triplet_acc

    def training_step(self, batch, batch_idx):
        triplet_loss, triplet_acc = self.triplet_loss_acc(batch[0], batch_idx)

        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        return triplet_loss

    def validation_step(self, batch, batch_idx):
        triplet_loss, triplet_acc = self.triplet_loss_acc(batch[0], batch_idx)

        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        triplet_loss, triplet_acc = self.triplet_loss_acc(batch[0], batch_idx)

        self.log('test_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('test_triplet_loss', triplet_loss, sync_dist=True)

    def configure_optimizers(self):
        # optimizer = SGD(model.parameters(), lr=self.hparams.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.opt = optimizer
        return optimizer

    def setup(self, stage):
        raise NotImplementedError

    def get_dataloader(self, dataset, batch_size, split):
        drop_last = True if split == "train" else False
        shuffle = True if split == "train" else False
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
            num_workers=self.hparams.dataloader_num_workers, drop_last=drop_last, shuffle=shuffle)
        return dataloader

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.train_triplets))
        print(f"\nlen_train:{len(dataset)}")
        return self.get_dataloader(dataset, self.hparams.train_batch_size, "train")

    def val_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.valid_triplets))
        print(f"\nlen_valid:{len(dataset)}")
        return self.get_dataloader(dataset, self.hparams.train_batch_size, "valid")

    def test_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.test_triplets))
        print(f"\nlen_test:{len(dataset)}")
        return self.get_dataloader(dataset, self.hparams.train_batch_size, "test")

    @staticmethod
    def add_generic_args(parser) -> None:
        parser.add_argument("--gpus", default=1, type=int)
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--max_epochs", default=200, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--train_batch_size", default=16, type=int)
        parser.add_argument("--eval_batch_size", default=64, type=int)
        parser.add_argument("--dataloader_num_workers", default=4, type=int)
        parser.add_argument("--train_dir", default=None, type=str, required=False)
        parser.add_argument("--valid_dir", default=None, type=str, required=False)
        parser.add_argument("--wandb_group", default=None, type=str)
        parser.add_argument("--wandb_mode", default="online", type=str)
        parser.add_argument("--wandb_project", default="?", type=str)
        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
        parser.add_argument("--hidden_size", default=256, type=int, help="Embedding size")
        parser.add_argument("--add_linear", action="store_true")
        parser.add_argument("--split_by", default=None, type=str, required=True)
        parser.add_argument("--img_split", default=0.6, type=float)

def generic_train(model: RESN_TN, args: argparse.Namespace,early_stopping_callback=False, extra_callbacks=[], checkpoint_callback=None, logging_callback=None,  **extra_train_kwargs):
    output_dir = os.path.join("results", model.hparams.wandb_project)
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    experiment = wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode, 
        group=args.wandb_group,
        name=f"{time.strftime('%m/%d_%H:%M')}")

    logger = WandbLogger(project="imagenet_bm", experiment=experiment)

    ckpt_path = os.path.join(output_dir, logger.version, "checkpoints")
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{valid_loss:.2f}", monitor="valid_triplet_loss", mode="min", save_last=True, save_top_k=2, verbose=True)

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
        check_val_every_n_epoch=1,
        **train_params)

    if args.do_train:
        trainer.fit(model)
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)

    if args.do_test:
        # best_model_path = os.path.join(ckpt_path, "best_model.ckpt")
        # model = model.load_from_checkpoint(best_model_path)
        trainer.test(model)

    return trainer