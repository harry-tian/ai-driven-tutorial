
from omegaconf import OmegaConf as oc
# import torchvision
import pytorch_lightning as pl
import torch, argparse
from pytorch_lightning.loggers import WandbLogger
import wandb
import time, pickle, os
import numpy as np
from pathlib import Path
from torchmetrics.functional.classification import auroc, stat_scores, average_precision, precision_recall_curve, auc
# from zmq import device
import shutil


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default='/net/scratch/tianh/explain_teach/models/configs/base.yaml', type=str, required=False)
    parser.add_argument("--model_config", default=None, type=str, required=True)
    parser.add_argument("--dataset_config", default=None, type=str, required=False)
    parser.add_argument("--triplet_config", default=None, type=str, required=False)
    return parser

def test_configs(configs):
    required_args = ["gpus", "seed", "dataloader_num_workers",
    "max_epochs", "learning_rate", "train_batch_size", "embed_dim",
    "num_class" ,"train_dir", "valid_dir", "test_dir", "transform", 
    "wandb_group", "wandb_mode", "wandb_project", "wandb_entity",  "wandb_name",
    "do_train", "do_test",
    "train_triplets", "valid_triplets", "test_triplets",
    "pretrained", "lamda", "syn"]
    syn_args = ["syn", "train_synthetic", "test_synthetic", "weights", "powers"]
    
    if "syn" in configs:
        if configs["syn"]: 
            for arg in syn_args: assert(arg in configs)
            required_args += syn_args
            
    if set(configs) != set(required_args):
        print("\n WARNING: Missing args:")
        print(np.setdiff1d(required_args, configs))
        print("WARNING: Unrecognized args:")
        print(np.setdiff1d(configs, required_args))

def load_configs(config_files):
    base_config = oc.load(config_files.base_config)
    model_config = oc.load(config_files.model_config)
    dataset_config = oc.load(config_files.dataset_config) if config_files.dataset_config else {}
    triplet_config = oc.load(config_files.triplet_config) if config_files.triplet_config else {}
    configs = oc.merge(base_config, dataset_config,  model_config, triplet_config)
    test_configs(configs)
    return configs

def generic_train(model, args, monitor, 
                    early_stopping_callback=False, extra_callbacks=[], checkpoint_callback=None, logging_callback=None,  **extra_train_kwargs):
    output_dir = os.path.join("results", model.hparams.wandb_project)
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    wandb_name = f"{time.strftime('%m/%d_%H:%M')}" if not args["wandb_name"] else args["wandb_name"]
    experiment = wandb.init(
        entity=args["wandb_entity"],
        project=args["wandb_project"],
        mode=args["wandb_mode"], 
        group=args["wandb_group"],
        name=wandb_name)

    logger = WandbLogger(project="imagenet_bm", experiment=experiment)

    ckpt_path = os.path.join(output_dir, logger.version, "checkpoints")
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{valid_loss:.2f}", monitor=monitor, mode="min", save_last=True, save_top_k=3, verbose=True)

    train_params = {}
    train_params["max_epochs"] = args["max_epochs"]
    if args["gpus"] == -1 or args["gpus"] > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(
        auto_select_gpus=True,
        gpus=args["gpus"],
        weights_summary=None,
        callbacks=extra_callbacks + [checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=1,
        **train_params)

    if args["do_train"]:
        trainer.fit(model)
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)


    if args["do_test"]:
        trainer.test(model, ckpt_path='best')
        
    ckpts = [f for f in os.listdir(ckpt_path)]
    for ckpt in ckpts:
        if ckpt != "best_model.ckpt":
            os.remove(os.path.join(ckpt_path, ckpt))

    return trainer

def do_test(model, args, ckpt_path):
    wandb_name = f"{time.strftime('%m/%d_%H:%M')}" if not args["wandb_name"] else args["wandb_name"]
    experiment = wandb.init(
        entity=args["wandb_entity"],
        project=args["wandb_project"],
        mode=args["wandb_mode"], 
        group=args["wandb_group"],
        name=wandb_name)

    logger = WandbLogger(experiment=experiment)

    train_params = {}
    train_params["max_epochs"] = args["max_epochs"]
    if args["gpus"] == -1 or args["gpus"] > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(
        gpus=args["gpus"],
        auto_select_gpus=True, ## true
        weights_summary=None,
        logger=logger,
        check_val_every_n_epoch=1,
        **train_params)
        
    trainer.test(model, ckpt_path=ckpt_path)

    return trainer

def get_dataloader(dataset, batch_size, split, num_workers):
    drop_last = True if split == "train" else False
    shuffle = True if split == "train" else False
    batch_size = min(len(dataset), batch_size)
    # batch_size = min(128,batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        num_workers=num_workers, drop_last=drop_last, shuffle=shuffle)
    return dataloader

def sample_dataset(dataset, index):
    input = torch.tensor(np.array([data[0].numpy() for data in dataset]))[index]
    label = torch.tensor(np.array([data[1] for data in dataset]))[index]
    dataset = torch.utils.data.TensorDataset(input, label)
    return dataset

def metrics(probs, target, threshold=0.5, num_class=2):
    if num_class > 2:
        pred = torch.argmax(probs, dim=1)
    else:
        pred = (probs >= threshold).long()
    target = target.int()

    tp, fp, tn, fn, sup = stat_scores(pred, target, ignore_index=0)
    if 0 < sup < len(target):
        precision, recall, _ = precision_recall_curve(pred, target)
        auprc = auc(recall, precision)
    m = {}
    m['pred'] = pred
    m['acc'] = (tp + tn) / (tp + tn + fp + fn)
    if num_class < 3:
        m['auc'] = auroc(probs, target) if 0 < sup < len(target) else None
        # m['tpr'] = tp / (tp + fn)
        # m['tnr'] = tn / (tn + fp)
        # m['ppv'] = tp / (tp + fp)
        # m['f1'] = 2 * tp / (2 * tp + fp + fn)
        # m['ap'] = average_precision(probs, target)
        # m['auprc'] = auprc if 0 < sup < len(target) else None
    return m   

# def get_acc(probs, target, threshold=0.5, multiclass=False, verbose=False): 
#     if multiclass:
#         pred = torch.argmax(probs, dim=1)
#     else:
#         pred = (probs >= threshold).long()

#     correct = (pred == target).sum().item()
#     total = len(target)

#     return correct/total


def dataset_with_indices(cls):

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })



# def add_generic_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gpus", default=1, type=int)
#     parser.add_argument("--seed", default=42, type=int)

#     parser.add_argument("--max_epochs", default=200, type=int)
#     parser.add_argument("--learning_rate", default=1e-4, type=float)
#     parser.add_argument("--train_batch_size", default=16, type=int)
#     parser.add_argument("--valid_batch_size", default=64, type=int)
#     parser.add_argument("--test_batch_size", default=64, type=int)
#     parser.add_argument("--dataloader_num_workers", default=4, type=int)
#     parser.add_argument("--num_class", default=2, type=int)

#     parser.add_argument("--train_dir", default=None, type=str, required=True)
#     parser.add_argument("--valid_dir", default=None, type=str, required=False)
#     parser.add_argument("--test_dir", default=None, type=str, required=False)

#     parser.add_argument("--wandb_group", default=None, type=str)
#     parser.add_argument("--wandb_mode", default="offline", type=str)
#     parser.add_argument("--wandb_project", default="?", type=str)
#     parser.add_argument("--wandb_entity", default="ai-driven-tutorial", type=str)
#     parser.add_argument("--wandb_name", default=None, type=str)

#     parser.add_argument("--do_train", action="store_true")
#     parser.add_argument("--do_test", action="store_true")
#     parser.add_argument("--ckpt_path", default=None, type=str, required=False)
    
#     parser.add_argument("--transform", default="bm", type=str)

#     parser.add_argument("--train_triplets", default=None, type=str, required=False)
#     parser.add_argument("--valid_triplets", default=None, type=str, required=False) 
#     parser.add_argument("--test_triplets", default=None, type=str, required=False) 

#     parser.add_argument("--train_synthetic", default=None, type=str, required=False) 
#     parser.add_argument("--test_synthetic", default=None, type=str, required=False) 

#     parser.add_argument("--syn", action="store_true") 
#     parser.add_argument("--w1", default=1, type=float)
#     parser.add_argument("--w2", default=1, type=float)

#     return parser



# def old_train(model, args, monitor, 
#                     early_stopping_callback=False, extra_callbacks=[], checkpoint_callback=None, logging_callback=None,  **extra_train_kwargs):
#     output_dir = os.path.join("results", model.hparams.wandb_project)
#     odir = Path(output_dir)
#     odir.mkdir(parents=True, exist_ok=True)
#     log_dir = Path(os.path.join(output_dir, 'logs'))
#     log_dir.mkdir(parents=True, exist_ok=True)

#     wandb_name = f"{time.strftime('%m/%d_%H:%M')}" if not args.wandb_name else args.wandb_name
#     experiment = wandb.init(
#         entity=args.wandb_entity,
#         project=args.wandb_project,
#         mode=args.wandb_mode, 
#         group=args.wandb_group,
#         name=wandb_name)

#     logger = WandbLogger(project="imagenet_bm", experiment=experiment)

#     ckpt_path = os.path.join(output_dir, logger.version, "checkpoints")
#     if checkpoint_callback is None:
#         checkpoint_callback = pl.callbacks.ModelCheckpoint(
#             dirpath=ckpt_path, filename="{epoch}-{valid_loss:.2f}", monitor=monitor, mode="min", save_last=True, save_top_k=3, verbose=True)

#     train_params = {}
#     train_params["max_epochs"] = args.max_epochs
#     if args.gpus == -1 or args.gpus > 1:
#         train_params["distributed_backend"] = "ddp"

#     trainer = pl.Trainer.from_argparse_args(
#         args,
#         auto_select_gpus=True, ## true
#         weights_summary=None,
#         callbacks=extra_callbacks + [checkpoint_callback],
#         logger=logger,
#         check_val_every_n_epoch=1,
#         **train_params)

#     if args.do_train:
#         trainer.fit(model)
#         target_path = os.path.join(ckpt_path, 'best_model.ckpt')
#         print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
#         shutil.copy(checkpoint_callback.best_model_path, target_path)

#     if args.do_test:
#         trainer.test(model, ckpt_path='best')

#     return trainer
