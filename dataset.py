# -*- coding: utf-8 -*-
import os
import cv2
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import random
from sklearn.model_selection import train_test_split
import joblib
import torch
import config as cfg


class MyDataset(Dataset):

    def __init__(self, mode=cfg.mode, feat_path=cfg.feat_path):
        super(MyDataset, self).__init__()
        self.seed = cfg.seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.mode = mode
        self.feat_path = feat_path
        self.load_data = self.load_phase()

    def load_phase(self):
        feat_path = os.path.join(self.feat_path)
        load_data = joblib.load(feat_path)

        return load_data

    def class_choose(self, c_idx, data):
        self.cur_c_data = []
        for (img_tensor, label, detect) in data:
            if label == c_idx:
                self.cur_c_data.append((img_tensor, detect))

        return self.cur_c_data

    def split_phase(self, data):
        train_cs_data, valid_cs_data = train_test_split(data, test_size=0.2, random_state=self.seed)

        return train_cs_data, valid_cs_data

    def data_choose(self, mode=cfg.mode, train_data=None, valid_data=None, test_data=None):
        feats = {'train': train_data, 'valid': valid_data} if mode in ['train', 'valid'] else {'test':test_data}
        self.data = feats[mode]

        return self.data

    def convert_to_tensor_dataset(self, data):
        feats, labels, detects = [], [], []
        for img_tensor, label, detect in data:
            feats.append(img_tensor)
            labels.append(label)
            detects.append(detect)

        feats = torch.tensor(np.array(feats), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        detects = torch.tensor(detects, dtype=torch.long)
        return TensorDataset(feats, labels, detects)

    def __getitem__(self, index):
        img_tensor, detect = self.data[index]

        return img_tensor, detect

    def __len__(self):
        return len(self.data)


class DataAug(Dataset):
    def __init__(self, ):
        ### CutMix,mixup,gaussian,specaug
        super(DataAug, self).__init__()
        self.seed = cfg.seed
        np.random.seed(cfg.seed)
        random.seed(self.seed)

    # cutmix
    def rand_bbox(self, size, lam):
        W = size[3]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix_batch(self, x, beta=1):
        B = x.size(0)
        x_anchor = x.clone()
        x_positive = x.clone()
        index = torch.randperm(B)

        for i in range(B):
            lam = np.random.beta(beta, beta)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)

            x_positive[i, :, bby1:bby2, bbx1:bbx2] = x[index[i], :, bby1:bby2, bbx1:bbx2]

        return x_positive

    def mixup(self, x, alpha=0.5):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]

        return mixed_x

    def randomgaussian_batch(self, x, max_ksize=9, stdev_x=20):  # max_ksize=5
        x_np = x.squeeze(1).cpu().numpy()  # B, W, H
        blurred = []
        for i in range(x_np.shape[0]):
            kernel_size = tuple(2 * np.random.randint(0, max_ksize // 2 + 1, 2) + 1)
            blurred_img = cv2.GaussianBlur(x_np[i], kernel_size, stdev_x)
            blurred.append(torch.tensor(blurred_img))
        blurred = torch.stack(blurred).unsqueeze(1).to(x.device)  # B, 1, W, H
        return blurred

    def spec_augment_batch(self, x, F=20, T=20, m_f=1, m_t=1, mask_val='zero', reduce_mask_range=0):
        B, C, W, H = x.shape
        x_aug = x.clone()

        for i in range(B):
            x_aug[i, 0] = self.freq_mask(x_aug[i, 0], F=F, n_masks=m_f, mask_val=mask_val,
                                         reduce_mask_range=reduce_mask_range)
            x_aug[i, 0] = self.time_mask(x_aug[i, 0], T=T, n_masks=m_t, mask_val=mask_val)

        return x_aug

    def freq_mask(self, _spec, F=30, n_masks=1, mask_val='zero', reduce_mask_range=0):
        W, H = _spec.shape
        for i in range(n_masks):
            bw = int(np.random.uniform(low=0.0, high=F))
            if reduce_mask_range == 0:
                f0 = random.randint(0, H - bw)
            else:
                f0 = random.randint(0, int(H * reduce_mask_range) - bw)

            if mask_val == 'zero':
                _spec[:, f0:f0 + bw] = 0
            elif mask_val == 'min':
                _spec[:, f0:f0 + bw] = _spec.min()
            elif mask_val == 'mean':
                _spec[:, f0:f0 + bw] = _spec.mean()
            elif mask_val == 'max':
                _spec[:, f0:f0 + bw] = _spec.max()
            elif mask_val == 'noise':
                _spec[:, f0:f0 + bw] = np.random.normal(_spec.mean(), _spec.std(), size=(W, bw))
        return _spec

    def time_mask(self, _spec, T=40, n_masks=1, mask_val='zero'):
        W, H = _spec.shape
        for i in range(n_masks):
            deltat = int(np.random.uniform(low=0.0, high=T))
            t0 = random.randint(0, W - deltat)

            if mask_val == 'zero':
                _spec[t0: t0 + deltat, :] = 0
            elif mask_val == 'min':
                _spec[t0: t0 + deltat, :] = _spec.min()
            elif mask_val == 'mean':
                _spec[t0: t0 + deltat, :] = _spec.mean()
            elif mask_val == 'max':
                _spec[t0: t0 + deltat, :] = _spec.max()
            elif mask_val == 'noise':
                _spec[t0: t0 + deltat, :] = np.random.normal(_spec.mean(), _spec.std(), size=(deltat, H))
        return _spec

    def batch_topk_patch_mixup(self, R_a, patch_size=(32, 32), topk=4):
        B, C, T, F = R_a.shape
        pT, pF = patch_size
        index = torch.randperm(B)
        R_b = R_a[index]

        A_a = torch.mean(torch.abs(R_a), dim=1, keepdim=True)
        A_b = torch.mean(torch.abs(R_b), dim=1, keepdim=True)

        patches_a = A_a.unfold(2, pT, pT).unfold(3, pF, pF)
        patches_b = A_b.unfold(2, pT, pT).unfold(3, pF, pF)
        Nt, Nf = patches_a.shape[2], patches_a.shape[3]
        scores = (patches_a + patches_b).mean(dim=(-1, -2)).view(B, -1)

        _, topk_idx = torch.topk(scores, k=topk, dim=-1)

        R_mix = R_a.clone()
        for b in range(B):
            for k in range(topk):
                idx = topk_idx[b, k].item()
                i, j = idx // Nf, idx % Nf
                t0, f0 = i * pT, j * pF
                R_mix[b, :, t0:t0 + pT, f0:f0 + pF] = 0.5 * R_a[b, :, t0:t0 + pT, f0:f0 + pF] + \
                                                      0.5 * R_b[b, :, t0:t0 + pT, f0:f0 + pF]

        return R_mix, index

    def soft_attention_mixup(self, R_a, R_b):
        A_a = torch.mean(torch.abs(R_a), dim=1, keepdim=True)  # [B, 1, T, F]
        A_b = torch.mean(torch.abs(R_b), dim=1, keepdim=True)

        alpha = A_a / (A_a + A_b + 1e-6)  # attention-based weight
        R_mix = alpha * R_a + (1 - alpha) * R_b
        return R_mix


if __name__ == '__main__':
    index = 0
    c_idx = 0

    dataset = MyDataset(mode='test')
    test_c_data = dataset.class_choose(c_idx=c_idx, data=dataset.load_data)
    test_data = dataset.data_choose(mode='test', test_data=test_c_data)
    img_tensor_test, detect_test = test_data[index]

    dataset = MyDataset(mode='train')
    load_c_data = dataset.class_choose(c_idx=c_idx, data=dataset.load_data)
    train_c_data, valid_c_data = dataset.split_phase(data=load_c_data)

    train_data = dataset.data_choose(mode='train', train_data=train_c_data)
    img_tensor_train, detect_train = train_data[index]

    valid_data = dataset.data_choose(mode='valid', valid_data=valid_c_data)
    img_tensor_valid, detect_valid = valid_data[index]

    print('Right')
