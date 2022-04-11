# AI driven tutorial

`algorithms` contains the teaching example selection algorithms  
`code` contains ?  
`data` contains mostly triplets  
`embeds` contains generated embeddings or distance functions.  
`evals` packages for evaluating embeddings and teaching algorithms  
`models` contains models used to generated embeddings  

## Evaluation steps
1. run a model in the `models` dir and save the best checkpoint
2. use `models/get_embeds.py` and the checkpoint to generate an embedding
3. use functions in `evals/embed_eval.py` to evaluate the embedding



## Reference
If you find our work useful in your research please consider citing our paper.  
```
@inproceedings{,
  title     = {},
  author    = {},
  booktitle = {},
  year = {}
}
```
