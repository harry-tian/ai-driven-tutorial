
from torchvision import transforms
from torchmetrics.functional.classification import auroc, stat_scores, average_precision, precision_recall_curve, auc

import time
import shutil
from pytorch_lightning.loggers import WandbLogger
import wandb
import os, pickle
from pathlib import Path
import pytorch_lightning as pl
import torch
import numpy as np
from pydoc import locate
def food_transform():
    transform = transforms.Compose([
        transforms.Resize([230,230]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def bm_transform_aug(hparams=None):
    affine = {}
    affine["degrees"] = 30 #hparams.rotate
    # if hparams.translate > 0: 
    #     translate = hparams.translate
    #     affine["translate"] = (translate, translate)
    if 0.2 > 0: 
        scale = 0.2
        affine["scale"] = (1 - scale, 1 + scale)
    # if hparams.shear > 0:
    #     shear = hparams.shear
    #     affine["shear"] = (-shear, shear, -shear, shear)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(0),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomAffine(**affine)
    ])
    return transform

def bm_transform():
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return val_transform

def metrics(probs, target, threshold=0.5):
    pred = (probs >= threshold).long()
    tp, fp, tn, fn, sup = stat_scores(pred, target, ignore_index=0)
    if 0 < sup < len(target):
        precision, recall, _ = precision_recall_curve(pred, target)
        auprc = auc(recall, precision)
    m = {}
    m['pred'] = pred
    m['auc'] = auroc(probs, target) if 0 < sup < len(target) else None
    m['acc'] = (tp + tn) / (tp + tn + fp + fn)
    m['tpr'] = tp / (tp + fn)
    m['tnr'] = tn / (tn + fp)
    m['ppv'] = tp / (tp + fp)
    m['f1'] = 2 * tp / (2 * tp + fp + fn)
    m['ap'] = average_precision(probs, target)
    m['auprc'] = auprc if 0 < sup < len(target) else None
    return m

def dataset_with_indices(cls):

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def generic_train(model, args, moniter, 
                    early_stopping_callback=False, extra_callbacks=[], checkpoint_callback=None, logging_callback=None,  **extra_train_kwargs):
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
            dirpath=ckpt_path, filename="{epoch}-{valid_loss:.2f}", monitor=moniter, mode="min", save_last=True, save_top_k=2, verbose=True)

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

def get_dataloader(dataset, batch_size, split, num_workers):
    drop_last = True if split == "train" else False
    shuffle = True if split == "train" else False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        num_workers=num_workers, drop_last=drop_last, shuffle=shuffle)
    return dataloader

def get_embeds(model_path, args, ckpt, split, embed_path=None):
    model = locate(model_path)
    model = model.load_from_checkpoint(ckpt, **vars(args)).to("cuda")
    model.eval()
    train_dataset, valid_dataset = model.get_datasets()
    if split == "train":
        dataset = train_dataset
    elif split == "val" or split == "valid":
        dataset = valid_dataset
    else:
        print("???")
        quit()
    
    # embeds = model.embed(dataset)
    embeds = model.feature_extractor(dataset)
    for layer in model.fc:
        embeds = layer(embeds)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=4, shuffle=False)
    # embeds = []
    # for batch_idx, batch in enumerate(dataloader):
    #     if type(batch) == list:
    #         batch = batch[0]
    #     if type(batch) != torch.Tensor:
    #         print("???")
    #         quit()
    
    #     # print(batch)
    #     # print(batch.shape)
    #     # print(type(batch))
    #     # quit()
    #     embeds.append(model.embed(batch))

    # embeds = np.asarray([e.squeeze().detach().numpy() for e in embeds])[0]
    embeds = embeds.cpu().detach().numpy()
    print(f"embeds.shape:{embeds.shape}")

    if not embed_path:
        embed_path = f"{model_path}_{split}.pkl"
    pickle.dump(embeds, open(embed_path, "wb"))
    print(f"dumped to {embed_path}")

    return embeds