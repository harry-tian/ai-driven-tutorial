import torch
import torchvision
import lpips
import numpy as np
import pickle
from torchvision import transforms
loss_fn = lpips.LPIPS(net='alex')
loss_fn.cuda()

transform = transforms.Compose([
    transforms.Resize([230,230]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_dir = '/net/scratch/tianh/food100-dataset/images'
dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
data = next(iter(dataloader))[0].cuda()

with torch.no_grad():
    distances = []
    for i in data:
        distances.append(loss_fn.forward(i, data, normalize=True).cpu().numpy().ravel())

distances_matrix = np.vstack(distances)
print(distances_matrix.shape)
pickle.dump(distances_matrix, open("lpips.food.pkl", "wb"))