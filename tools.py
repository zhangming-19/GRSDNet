# -*- coding: utf-8 -*-
import numpy as np
import csv
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from config import *


class ModelTrainer(object):

    @staticmethod
    def compute_losses(embedi, embedj, labels, criterion_cls, criterion_contra, model_head, loss_mode, contra_loss_weight):
        if loss_mode == 'cls':
            # Compute ArcFace loss for classification
            loss_arcface = criterion_cls(embedi, labels)  # Subcon embedi.unsqueeze(1),ArcFace embedi
            return loss_arcface, loss_arcface, 0

    @staticmethod
    def train(data_loader, gre_model, cnn_model, model_head,
              criterion_cls, criterion_contra, contra_loss_weight, loss_mode,
              augmethod, dataaug, optimizer, epoch_id, device, max_epoch):

        use_gre = False
        if gre_model not in ['', None, False]:
            use_gre = True
            gre_model.train()
        cnn_model.train()

        conf_mat = np.zeros((class_num, class_num))
        loss_sigma, cls_losses, contra_losses = [], [], []

        model_type = 'Train'
        msf = f'{model_type:<8}(Epoch {epoch_id + 1}/{max_epoch})'
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=msf)

        for _, data in progress_bar:
            inputs, labels, _ = data
            inputs = inputs.to(torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()

            if use_gre:
                _, _, _, _, inputs = gre_model(inputs)  # B_map, R, R_excited, R_feat

            # 输入模型 优化 损失
            if loss_mode == 'cls':
                resulti, embedi = cnn_model(inputs)  # embedding i (for ex1)
                resultj, embedj = None, None
            elif learn_method == 'Contrastive':
                if 'cutmix' in augmethod:
                    ex1 = dataaug.cutmix_batch(inputs)
                    ex2 = dataaug.cutmix_batch(inputs)
                elif 'mixup' in augmethod:
                    ex1 = dataaug.mixup(inputs)
                    ex2 = dataaug.mixup(inputs)
                elif 'gaussian' in augmethod:
                    ex1 = dataaug.randomgaussian_batch(inputs)
                    ex2 = dataaug.randomgaussian_batch(inputs)
                elif 'specaug' in augmethod:
                    ex1 = dataaug.spec_augment_batch(inputs)
                    ex2 = dataaug.spec_augment_batch(inputs)
                resulti, embedi = cnn_model(ex1)  # embedding i (for ex1)
                resultj, embedj = cnn_model(ex2)  # embedding j (for ex2)  resultj, embedj = model(ex2)

            losses, loss_arcface, loss_contra = ModelTrainer.compute_losses(embedi, embedj, labels, criterion_cls, criterion_contra, model_head, loss_mode, contra_loss_weight)

            losses.backward()
            optimizer.step()

            if loss_mode == 'cls':
                losses = loss_arcface
            elif loss_mode == 'contra':
                losses = loss_contra
            elif loss_mode == 'clscontra':
                losses = (1-contra_loss_weight) * loss_arcface + contra_loss_weight * loss_contra
            else:
                return ('Wrong')

            if 'cls' in loss_mode:
                cls_losses.append(loss_arcface.item())
            elif 'contra' in loss_mode:
                contra_losses.append(loss_contra.item())
            loss_sigma.append(losses.item())

            # Predictions and metrics
            result = (resulti+resultj)/2 if learn_method == 'clscontra' else resulti
            _, predicted = torch.max(result, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            acc_avg = conf_mat.trace() / conf_mat.sum()


            progress_bar.set_postfix({'AllLoss': f'{losses:.6f}', 'Cls': f'{loss_arcface:.6f}',
                                      'Contra': f'{loss_contra:.6f}','Acc': f'{acc_avg:.4f}'})
            #break
        progress_bar.close()
        return np.mean(loss_sigma)


def calculate_pauc(class_real_detects, class_distances):
    # ROC
    pauc = roc_auc_score(class_real_detects,class_distances, max_fpr=pauc_value)
    return pauc

# Mixup
def mixup_data(x, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x



def kmeans_test(self, test_data, epoch_num=None, counts=None):

    model_name = self.model_pklpath.split('_')[-2]

    test_feats = [[] for _ in range(class_num)]

    for embed, label, detect in test_data:
        test_feats[label].append((embed, label, detect))

    avg_auc_helps = []
    avg_pauc_helps = []
    per_class_results = {}

    for idx, test_feat in enumerate(test_feats):

        cur_model_pklpath = self.model_pklpath.replace('.pkl', f'_{idx}.pkl')
        kmeans_model = CustomKMeans.load_model(cur_model_pklpath)

        test_features = []
        test_real_detects = []

        for feat, label, detect in test_feat:
            test_features.append(feat)
            test_real_detects.append(detect)

        test_real_detects = np.asarray(test_real_detects)

        test_distances = kmeans_model._compute_distances(
            test_features
        )

        test_weighted_distances = test_distances

        class_distances = np.asarray([
            torch.min(distance).cpu().numpy()
            for distance in test_weighted_distances
        ])

        assert set(test_real_detects).issubset({0, 1})

        class_auc = roc_auc_score(
            test_real_detects,
            class_distances
        )

        pauc = CustomKMeans.calculate_pauc(
            test_real_detects,
            class_distances
        )

        avg_auc_helps.append(class_auc)
        avg_pauc_helps.append(pauc)

        per_class_results[idx] = {
            "AUC": class_auc,
            "pAUC": pauc,
        }


    all_data_dict = {
        'model_name': model_name,
        'avg_auc_help': np.mean(avg_auc_helps),
        'avg_pauc_help': np.mean(avg_pauc_helps),

        **{
            f'class{label}_AUC':
            per_class_results[label]["AUC"]
            for label in per_class_results
        },

        **{
            f'class{label}_pAUC':
            per_class_results[label]["pAUC"]
            for label in per_class_results
        },
    }

    return (
        all_data_dict,
        per_class_results,
        np.mean(avg_auc_helps),
        np.mean(avg_pauc_helps))


class EarlyStopping:
    def __init__(self, mode='max', patience=10, monitor='score'):
        self.mode = mode
        self.patience = patience
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def _is_improvement(self, current):
        if self.best_score is None:
            return True
        if self.mode == 'min':
            return current < self.best_score
        else:
            return current > self.best_score

    def update(self, metrics: dict, model=False, save_path='best_model.pth'):
        current = metrics[self.monitor]

        if self._is_improvement(current):
            self.best_score = current
            self.counter = 0
            #if not os.path.exists(save_path):
                #torch.save(model.state_dict(), save_path)
            #print(f"New best model saved at {save_path} — {self.monitor}: {current:.5f}")
        else:
            self.counter += 1
            print(f"No improvement in {self.monitor} ({current:.5f}), counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered after {self.patience} rounds.")


def plot_valid_loss(train_loss_list, valid_loss_list):
    epochs = range(1, len(valid_loss_list) + 1)
    plt.plot(epochs, train_loss_list, 'r', label='Training loss')
    plt.plot(epochs, valid_loss_list, 'b', label='Validating loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    fig = plt.gcf()

    return fig


import os
def load_testmodels(model_save):
    test_models = []
    model_loss_pairs = []
    for save_model_name in os.listdir(model_save):
        if save_model_name.endswith("cnn.pth"):
            parts = save_model_name.split('_')
            loss_value = float(parts[1])  # Extract loss from the last part and convert to float
            model_loss_pairs.append((loss_value, save_model_name))  # Store the model and its loss value
    model_loss_pairs.sort(key=lambda x: x[0])  # Sort models by loss values in ascending order
    for loss, model_filename in model_loss_pairs[:20] :  #  [:choose_model_nums] Select the top 10 models with the best loss
        path_ = os.path.join(model_save, model_filename)
        test_models.append((model_filename, path_))
    return test_models


# kmean - kmean_featext
def kmean_featext(test_loader, test_model, gre_model, cnn_model):
    use_progress = False
    tarin_progress_bar = tqdm(test_loader, total=len(test_loader), desc=f'{test_model:<10}') if use_progress else test_loader
    test_newdata = []
    with torch.no_grad():
        use_gre = False
        if gre_model is not None: use_gre = True
        for (ex, labels, detects) in tarin_progress_bar:
            ex = ex.to(device, dtype=torch.float32).unsqueeze(1)
            if use_gre:
                _, _, _, _, ex = gre_model(ex)  # B_map, R, R_excited, R_feat
            labels, detects = labels.to(device, dtype=torch.long), detects.to(device, dtype=torch.long)
            _, embedis = cnn_model(ex)
            for (embedi, label, detect) in zip(embedis, labels, detects):
                test_newdata.append((embedi, label, detect))
    return test_newdata

