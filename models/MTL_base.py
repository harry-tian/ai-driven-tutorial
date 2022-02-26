import torch
import pytorch_lightning as pl
from torchvision import models
from torch import nn
import utils

class MTL(pl.LightningModule):
    def __init__(self,**config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = models.resnet18(pretrained=self.hparams.pretrained)
        num_features = 1000

        self.embed_dim = self.hparams.embed_dim
        self.triplet_loss = nn.TripletMarginLoss()
        self.pdist = nn.PairwiseDistance()

        self.hidden_size = self.hparams.hidden_size
        self.fc = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(), nn.Linear(num_features, self.hidden_size), 
            nn.BatchNorm1d(self.hidden_size), nn.ReLU(), nn.Dropout(), nn.Linear(self.hidden_size, self.embed_dim)
        )])

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Dropout(), nn.Linear(self.embed_dim, 1)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.summarize()

    def embed(self, inputs):
        embeds = self.feature_extractor(inputs)
        for layer in self.fc:
            embeds = layer(embeds)
        return embeds

    def forward(self, **inputs):
        raise NotImplementedError

    def setup_data(self):
        raise NotImplementedError

    def get_loss_acc(self, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(batch, batch_idx)

        self.log('train_clf_loss', clf_loss, sync_dist=True)
        self.log('train_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('train_triplet_loss', triplet_loss, sync_dist=True)
        self.log('train_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('train_total_loss', total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(batch, batch_idx)

        self.log('valid_clf_loss', clf_loss, sync_dist=True)
        self.log('valid_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_triplet_loss', triplet_loss, sync_dist=True)
        self.log('valid_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('valid_total_loss', total_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        clf_loss, m, triplet_loss, triplet_acc, total_loss = self.get_loss_acc(batch, batch_idx)

        self.log('test_clf_loss', clf_loss, sync_dist=True)
        self.log('test_clf_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('test_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('test_triplet_loss', triplet_loss, sync_dist=True)
        self.log('test_triplet_acc', triplet_acc, prog_bar=True, sync_dist=True)
        self.log('test_total_loss', total_loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.opt = optimizer
        return optimizer

    # def get_dataloader(self, dataset, batch_size, split):
    #     drop_last = True if split == "train" else False
    #     shuffle = True if split == "train" else False
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
    #         num_workers=self.hparams.dataloader_num_workers, drop_last=drop_last, shuffle=shuffle)
    #     return dataloader
    # def get_datasets(self):
    #     return self.train_inputs, self.valid_inputs

    def train_dataloader(self):
        dataset = self.train_dataset
        print(f"\nlen_train:{len(dataset)}")
        return utils.get_dataloader(dataset, self.hparams.train_batch_size, "train")

    def val_dataloader(self):
        dataset = self.valid_dataset
        print(f"\nlen_valid:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "valid")

    def test_dataloader(self):
        dataset = self.test_dataset
        print(f"\nlen_test:{len(dataset)}")
        return utils.get_dataloader(dataset, len(dataset), "test")

    @staticmethod
    def add_generic_args(parser) -> None:
        parser.add_argument("--gpus", default=1, type=int)
        parser.add_argument("--seed", default=42, type=int)

        parser.add_argument("--max_epochs", default=200, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--train_batch_size", default=16, type=int)
        parser.add_argument("--eval_batch_size", default=64, type=int)
        parser.add_argument("--dataloader_num_workers", default=4, type=int)

        parser.add_argument("--wandb_group", default=None, type=str)
        parser.add_argument("--wandb_mode", default="offline", type=str)
        parser.add_argument("--wandb_project", default="?", type=str)

        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument("--do_embed", action="store_true")

        parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
        parser.add_argument("--hidden_size", default=256, type=int, help="Embedding size")
        parser.add_argument("--add_linear", action="store_true")
        
        # parser.add_argument("--split_by", default="img", type=str, required=False)
        # parser.add_argument("--img_split", default=0.6, type=float)
        parser.add_argument("--embed_path", default=None, type=str, required=False)
        parser.add_argument("--subset", action="store_true")
        parser.add_argument("--MTL_hparam", action="store_true")
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--w1", default=0.5, type=float)
        parser.add_argument("--w2", default=1, type=float)

def generic_train(model, args):
    moniter = "valid_total_loss"
    return utils.generic_train(model, args, moniter)