import pickle
import numpy as np
import torch

x_train = pickle.load(open('../data/datasets/wv_3d/train_features.pkl', 'rb')).astype(np.float32)
x_test = pickle.load(open('../data/datasets/wv_3d/test_features.pkl', 'rb')).astype(np.float32)
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
# print(x_train.shape, x_test.shape)

# from torchvision.datasets import ImageFolder
# train_ds = ImageFolder('../data/datasets/wv_3d/train')
# test_ds = ImageFolder('../data/datasets/wv_3d/test')
# y_train = torch.tensor([b[1] for b in train_ds])
# y_test = torch.tensor([b[1] for b in test_ds])
# y_train = np.array([b[1] for b in train_ds])
# y_test = np.array([b[1] for b in test_ds])
# pickle.dump(y_train, (open('../data/datasets/wv_3d/train_labels.pkl', 'wb')))
# pickle.dump(y_test, (open('../data/datasets/wv_3d/test_labels.pkl', 'wb')))
# print(np.all(a.numpy() == y_train), np.all(b.numpy() == y_test))

y_train = pickle.load(open('../data/datasets/wv_3d/train_labels.pkl', 'rb'))
y_test = pickle.load(open('../data/datasets/wv_3d/test_labels.pkl', 'rb'))
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
# print(y_train.dtype, y_test.shape)

CUDA = False
M = 100
# scale = torch.Tensor([0, 0, 1, 1])
scales = [torch.Tensor([0, i, 1, 1]) for i in [1, 0.1, 0.01, 0.005, 0.0025, 0.001, 0.0001]]

# for s in [0]:
s=0
for scale in scales:
    print(scale)
    torch.manual_seed(s)

    w = torch.rand(M, 4)
    w = w * 0 + scale
    w = w / w.sum(1, keepdim=True)
    # print(w[:5])

    wd = torch.diag_embed(w)

    if CUDA:
        wd = wd.cuda()
        x_train = x_train.cuda()
        x_test = x_test.cuda()

    with torch.no_grad():
        z_train = x_train @ wd
        z_test = x_test @ wd
        # print(z_train.shape, z_test.shape)

        dst = torch.cdist(z_test, z_train).cpu()
    # print(dst.shape)

    y_pred = dst.argmin(-1)
    y_pred = torch.take(y_train, y_pred)
    y_corr = y_pred == y_test
    y_nacc = y_corr.float().mean(1)
    # print(y_corr.shape, y_nacc.shape)

    # print(y_nacc[:5])
    # print(scale, y_nacc.mean(), y_nacc.std())

    # torch.manual_seed(s)
    # p = torch.randint(0, y_train.size(0), (M, y_test.size(0),))
    # n = torch.randint(0, y_train.size(0), (M, y_test.size(0),))
    # dap = torch.take(dst, p)
    # dan = torch.take(dst, n)

    # y_pred = torch.zeros(M, y_test.size(0)).long()
    # y_pred[dap < dan] = y_train.take(p[dap < dan])
    # y_pred[dap >= dan] = y_train.take(n[dap >= dan])
    # y_racc = (y_pred == y_test).float().mean(1)

    y_pred = dst.argmin(-1)
    y_pred = torch.take(y_train, y_pred)
    y_nacc = (y_pred == y_test).float().mean(1)

    # print(y_racc.mean(), y_nacc.mean(), y_nacc.std())
    print(y_nacc.mean(), y_nacc.std())
    # results = torch.cat([w, y_racc.unsqueeze(1), y_nacc.unsqueeze(1)], 1)
    # results = results.numpy()
    # pickle.dump(results, open(f'ws/M={M}_s={s}.pkl', 'wb'))