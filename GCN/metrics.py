from sklearn.metrics import log_loss
import torch
import numpy as np

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce(pred, gt):
    if torch.unique(gt).size(0) == 1:
        return np.nan
    cross_entropy = log_loss(gt.tolist(), pred.tolist())
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0