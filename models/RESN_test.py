# -*- coding: utf-8 -*-
from dataclasses import replace
from email.headerregistry import UniqueSingleAddressHeader
import os, pickle
import argparse

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import warnings
from torchvision import  models
warnings.filterwarnings("ignore")

import utils
from RESN import RESN
from omegaconf import OmegaConf as oc

ckpt_path = "results/wv/2zupostk/checkpoints/best_model.ckpt"

def main():
    parser = utils.config_parser()
    config_files = parser.parse_args()
    configs = utils.load_configs(config_files)

    wandb_name = oc.create({"wandb_group": "RESN_test",
    "wandb_name": configs["wandb_group"]}) 
    configs = oc.merge(configs, wandb_name)
    print(configs)


    model = RESN.load_from_checkpoint(ckpt_path, **configs)

    trainer = utils.do_test(model, configs, ckpt_path)

if __name__ == "__main__":
    main()

