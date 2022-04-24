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
from torch import nn
from RESN import RESN

def main():
    parser = utils.add_generic_args()
    RESN.add_model_specific_args(parser)
    args = parser.parse_args()
    # print(args)

    pl.seed_everything(args.seed)
    
    dict_args = vars(args)
    ckpt_path = dict_args["ckpt_path"]


    model = RESN.load_from_checkpoint(ckpt_path, **dict_args)

    trainer = utils.do_test(model, args, ckpt_path)

if __name__ == "__main__":
    main()

