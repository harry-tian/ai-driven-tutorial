import numpy as np
import snack
DATASET_HOME = "/Users/michael/food-10k/release/Food-10k/"
dset = simplejson.load(open(DATASET_HOME+"/dataset.json"))
features = np.load(DATASET_HOME+"/features.npy")

uuid_map = {uuid: i for i,uuid in enumerate(dset['image_uuids'])}
triplets = []
for line in open(DATASET_HOME+"/all-triplets.txt").readlines():
    (a,b,c) = line.replace("\n","").split(" ")
    triplets.append( (uuid_map[a], uuid_map[b], uuid_map[c]) )
triplets = np.array(triplets)

tsne_weight = 500.0 # btw 100 and 5000
tste_weight = 0.05 # btw 0.1 and 5
theta = 0.5 # 0.0 is exact solution, higher --> faster

# Standard embedding
embeds = snack.snack_embed(
    features.astype('float'), 
    tsne_weight,
    triplets, 
    tste_weight,
    theta = theta,
)
# fig,ax = subplots(figsize=(10, 10))
# ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
# ax.scatter(*Y.T, lw=0, s=1)
# _=ax.axis('off')

np.save('snack_dawc+lpips.npy', embeds)