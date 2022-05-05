# -*- coding: utf-8 -*-
from pydoc import locate
import pytorch_lightning as pl
import warnings
warnings.filterwarnings("ignore")


model_path_dict ={
    'TN': "TN.TN",
    'MTL': "MTL.MTL",
    'RESN': "RESN.RESN"
}



model = "RESN"
ckpt_path = "results/wv_2d/2wql6bo5/checkpoints/best_model.ckpt"








model_path = model_path_dict[model]
model = locate(model_path)
model = model.load_from_checkpoint(ckpt_path)

model.eval()
trainer = pl.Trainer(gpus=1)
trainer.test(model, ckpt_path=ckpt_path)


