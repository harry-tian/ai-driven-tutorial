from turtle import mode
import trainer
import pytorch_lightning as pl
from pydoc import locate

model_path_dict ={
    'TN': "TN.TN",
    'MTL': "MTL.MTL",
    'RESN': "RESN.RESN"
}

model_monitor_dict ={
    'TN': "valid_triplet_loss",
    'MTL': "valid_total_loss",
    'RESN': "valid_clf_loss"
}

def main():
    parser = trainer.config_parser()
    config_files = parser.parse_args()
    configs = trainer.load_configs(config_files)

    print(configs)

    pl.seed_everything(configs["seed"], workers=True)
    model_path = model_path_dict[configs["model"]]
    
    model = locate(model_path)
    model = model(**configs)

    monitor = model_monitor_dict[configs["model"]]
    trainer.generic_train(model, configs, monitor)

if __name__ == "__main__":
    main()
