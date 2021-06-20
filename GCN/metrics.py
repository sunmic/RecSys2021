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
    
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    cross_entropy = log_loss(gt_np, pred_np)
    data_ctr = calculate_ctr(gt_np)
    strawman_cross_entropy = log_loss(gt_np, [data_ctr for _ in range(len(gt_np))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0