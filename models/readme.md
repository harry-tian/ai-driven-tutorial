# experiment procedures

### hyperparameters

lamda={0.2,0.5,0.8}
* filtered, d=50
* filtered, d=512
* unfiltered, d=50
* unfiltered, d=512
### settings
1. aligns
2. noise
3. number of triplets









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