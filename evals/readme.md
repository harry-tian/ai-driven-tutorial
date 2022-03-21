# evaluation packages

### embed_evals
automatic evaluation for embeddings. metrics include triplet accuracy, model-1NN accuracy and human_1NN_alignment. Use function like `bm_eval_human` to get all evaluation metrics.

### teaching_evals
automatic evaluation for the entire teaching framework by using KNN to simulate human learnes. KNNs use distances such as TripletNet trained on human triplets.  