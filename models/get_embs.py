import utils
import argparse, pickle
from pydoc import locate

def get_embeds(model_path, args, ckpt, split, embed_path=None):
    model = locate(model_path)
    model = model.load_from_checkpoint(ckpt, **vars(args)).to("cuda")
    model.eval()
    train_dataset, valid_dataset = utils.get_bm_datasets()

    if split == "train":
        dataset = train_dataset.cuda()
    elif split == "val" or split == "valid":
        dataset = valid_dataset.cuda()
    else:
        print("???")
        quit()
    
    embeds = model.embed(dataset)
    # embeds = model.feature_extractor(dataset)
    # for layer in model.fc:
    #     embeds = layer(embeds)
    
    embeds = embeds.cpu().detach().numpy()
    print(f"embeds.shape:{embeds.shape}")

    if not embed_path:
        embed_path = f"{model_path}_{split}.pkl"
    pickle.dump(embeds, open(embed_path, "wb"))
    print(f"dumped to {embed_path}")

    return embeds

model_path = "resn_args.RESN"

args = argparse.Namespace(embed_dim=10)
ckpt = 'resn-emb2/1v1hnx3o'
ckpt = f"results/{ckpt}/checkpoints/best_model.ckpt" 



split = "valid"
name = "resn_bm"











embed_path = f"../embeds/{name}_{split}.pkl"
get_embeds(model_path, args, ckpt, split, embed_path)