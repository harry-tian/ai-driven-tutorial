# -*- coding: utf-8 -*-
from pydoc import locate
import pytorch_lightning as pl
import argparse, wandb, trainer
import warnings
from pytorch_lightning.loggers import WandbLogger
warnings.filterwarnings("ignore")


model_path_dict ={
    'TN': "TN.TN",
    'MTL': "MTL.MTL",
    'RESN': "RESN.RESN"
}


# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default=None, type=str, required=True)
# parser.add_argument("--ckpt_path", default=None, type=str, required=True)
# parser.add_argument("--seed", default=42, type=int, required=False)
# parser.add_argument("--wandb_mode", default="online", type=str)
# parser.add_argument("--wandb_entity", default="ai-driven-tutorial", type=str)
# parser.add_argument("--wandb_project", default="?", type=str)
# parser.add_argument("--wandb_group", default=None, type=str)
# parser.add_argument("--wandb_name", default=None, type=str)
# args = parser.parse_args()

# pl.seed_everything(args.seed)
# experiment = wandb.init(
#     entity=args.wandb_entity,
#     project=args.wandb_project,
#     mode=args.wandb_mode, 
#     group=args.wandb_group,
#     name=args.wandb_name,
#     config=args)


# logger = WandbLogger(project="imagenet_bm", experiment=experiment)


# model_path = model_path_dict[args.model]
# model = locate(model_path)
# model = model.load_from_checkpoint(args.ckpt_path)

# model.eval()
# trainer = pl.Trainer(gpus=1,
#         logger=logger,
#         enable_progress_bar=False)
# trainer.test(model, ckpt_path=args.ckpt_path)



parser = trainer.config_parser()
args = parser.parse_args()
configs = trainer.load_configs(args)
pl.seed_everything(configs["seed"], workers=True)
model_path = model_path_dict[configs["model"]]

model = locate(model_path)
model = model.load_from_checkpoint(args.ckpt_path)

trainer.do_test(model, configs, args.ckpt_path)





