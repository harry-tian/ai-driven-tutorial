# how to use

### configs
configurate hyperparamters using `.yaml` in `configs/`. Read more about this in `trainer.py`.

### embeddings
generatre embeddings using `gen_embeds.py`


# models

### dwac
automatic evaluation for embeddings. metrics include triplet accuracy, model-1NN accuracy and human_1NN_alignment. Use function like `bm_eval_human` to get all evaluation metrics.

### ResNET
uses ResNET18

### TripletNet(TN)
uses triplet margin loss

### MTL:BCE+TN
Loss = lambda * ResNET.BCE + (1-lambda) * TN