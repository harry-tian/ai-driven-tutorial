
from torchvision import transforms

def food_transform():
    transform = transforms.Compose([
        transforms.Resize([230,230]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def bm_train_transform(hparams=None):
    affine = {}
    affine["degrees"] = 30 #hparams.rotate
    # if hparams.translate > 0: 
    #     translate = hparams.translate
    #     affine["translate"] = (translate, translate)
    if 0.2 > 0: 
        scale = 0.2
        affine["scale"] = (1 - scale, 1 + scale)
    # if hparams.shear > 0:
    #     shear = hparams.shear
    #     affine["shear"] = (-shear, shear, -shear, shear)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(0),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomAffine(**affine)
    ])
    return transform

def bm_valid_transform():
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return val_transform