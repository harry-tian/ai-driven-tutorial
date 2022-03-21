import torch
import pytorch_lightning as pl
from torchvision import models
from torch import nn
import utils

class TripletNet(pl.LightningModule):
    def __init__(self,**config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = models.resnet18(pretrained=True)
        num_features = 1000

        self.triplet_loss = nn.TripletMarginLoss()
        self.pdist = nn.PairwiseDistance()

        self.embed_dim = self.hparams.embed_dim
        self.hidden_size = 256
        self.fc = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, self.hidden_size), 
            nn.BatchNorm1d(self.hidden_size), nn.ReLU(), nn.Dropout(), nn.Linear(self.hidden_size, self.embed_dim)
        )])

        self.summarize()

    def embed(self, inputs):
        embeds = self.feature_extractor(inputs)
        for layer in self.fc:
            embeds = layer(embeds)
        return embeds

    def forward(self, **inputs):
        raise NotImplementedError

    def triplet_loss_acc(self, triplet_idx, batch_idx):
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

    def setup_data(self):
        raise NotImplementedError

    def train_dataloader(self):
        dataset = self.train_dataset
        print(f"\nlen_train:{len(dataset)}")
        return utils.get_dataloader(dataset, self.hparams.train_batch_size, "train", self.hparams.dataloader_num_workers)

    def val_dataloader(self):
        dataset = self.valid_dataset
        print(f"\nlen_valid:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "valid", self.hparams.dataloader_num_workers)

    def test_dataloader(self):
        dataset = self.test_dataset
        print(f"\nlen_test:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "test", self.hparams.dataloader_num_workers)

    @staticmethod
    def add_generic_args(parser) -> None:
        parser.add_argument("--gpus", default=1, type=int)
        parser.add_argument("--seed", default=42, type=int)

        parser.add_argument("--max_epochs", default=45, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--train_batch_size", default=64, type=int)
        parser.add_argument("--eval_batch_size", default=64, type=int)
        parser.add_argument("--dataloader_num_workers", default=4, type=int)

        parser.add_argument("--wandb_group", default=None, type=str)
        parser.add_argument("--wandb_mode", default="online", type=str)
        parser.add_argument("--wandb_project", default="?", type=str)
        parser.add_argument("--wandb_entity", default="harry-tian", type=str)
        parser.add_argument("--wandb_name", default=None, type=str)

        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument("--do_embed", action="store_true")

        parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
        parser.add_argument("--hidden_size", default=256, type=int, help="Embedding size")
        parser.add_argument("--add_linear", action="store_true")

        parser.add_argument("--img_split", default=0.6, type=float)
        parser.add_argument("--subset", action="store_true")
        parser.add_argument("--embed_path", default=None, type=str, required=False)

def generic_train(model, args):
    moniter = "valid_triplet_loss"
    return utils.generic_train(model, args, moniter)