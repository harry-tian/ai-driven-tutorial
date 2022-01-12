import pickle
import utils


distance_matrix = pickle.load(open("lpips.prostatex.train+valid.pkl", "rb"))
# distance_matrix.shape

triplets_fname = "data/triplets.px.train+valid.pkl"
tste_fname = "data/tste.px.train+valid.pkl"

embeds = utils.get_tste(distance_matrix, triplets_fname, tste_fname, max_iter=1000)