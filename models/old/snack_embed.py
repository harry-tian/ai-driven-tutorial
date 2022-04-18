import numpy as np
import snack

model, name = 'dwac', 'emb10.merged2'
title = "prostatex"
train_path = 'embeds/{}_train_{}.npz'.format(model, name)
train_data = np.load(train_path)
valid_path = 'embeds/{}_valid_{}.npz'.format(model, name)
valid_data = np.load(valid_path)
features = np.concatenate((train_data["arr_2"],valid_data["arr_2"]))

triplets_path = 'embeds/triplets.px.train+valid.npy'
triplets = np.load(triplets_path)

tsne_weight = 500.0 # btw 100 and 5000
tste_weight = 0.05 # btw 0.1 and 5
theta = 0.0 # 0.0 is exact solution, higher --> faster
max_iter = 500

# Standard embedding
embeds = snack.snack_embed(
    features.astype('float'), 
    tsne_weight,
    triplets, 
    tste_weight,
    theta = theta,
    max_iter=max_iter
)
# fig,ax = subplots(figsize=(10, 10))
# ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
# ax.scatter(*Y.T, lw=0, s=1)
# _=ax.axis('off')

embeds_file = 'embeds/snack_dwac+lpips.npy'
np.save(embeds_file, embeds)