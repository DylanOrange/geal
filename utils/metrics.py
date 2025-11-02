import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def evaluating(pred, label):

    mae = torch.sum(torch.abs(pred-label), dim=(0,1))
    points_num = pred.shape[0] * pred.shape[1]

    return mae, points_num

def evaluating_2d(pred, label):

    sim = cal_sim(pred, label)
    mae = (np.abs(pred-label)).mean()
    
    return sim, mae

def KLD(map1, map2, eps = 1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    kld = np.sum(map2*np.log( map2/(map1+eps) + eps))
    return kld
    
def cal_SIM_3d(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def cal_kl(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld


def cal_sim(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)

    return np.sum(intersection)

def cal_nss(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred / 255.0
    gt = gt / 255.0
    std = np.std(pred)
    u = np.mean(pred)

    smap = (pred - u) / (std + 1e-12)
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
    fixation_map = image_binary(fixation_map, 0.1)

    nss = smap * fixation_map

    nss = np.sum(nss) / np.sum(fixation_map + 1e-12)

    return nss

def image_binary(image, threshold):
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if (image[xx][yy] > threshold):
                output[xx][yy] = 1
    return output

def cal_nss_batch(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # Normalize predictions and ground truth
    pred = pred / 255.0
    gt = gt / 255.0
    
    # Calculate mean and std along the spatial dimensions for each image in the batch
    std = np.std(pred, axis=(1, 2), keepdims=True)
    u = np.mean(pred, axis=(1, 2), keepdims=True)
    
    # Normalize saliency map
    smap = (pred - u) / (std + 1e-12)
    
    # Normalize and binarize the fixation map
    fixation_map = (gt - np.min(gt, axis=(1, 2), keepdims=True)) / (np.max(gt, axis=(1, 2), keepdims=True) - np.min(gt, axis=(1, 2), keepdims=True) + 1e-12)
    fixation_map = (fixation_map > 0.1).astype(np.float32)  # Vectorized thresholding
    
    # Calculate NSS
    nss = smap * fixation_map
    nss = np.sum(nss, axis=(1, 2)) / (np.sum(fixation_map, axis=(1, 2)) + 1e-12)
    nss = nss.mean()
    
    return nss

def calculate_batch_iou_auc(pred, target):
    """
    Compute IoU and AUC for each instance in a batch.
    - IoU is averaged across multiple thresholds.
    - AUC is computed with ROC-AUC.
    """
    num_samples = pred.shape[0]
    iou, auc = np.zeros(num_samples), np.zeros(num_samples)
    thresholds = np.linspace(0, 1, 20)
    target = (target >= 0.5).astype(int)

    for i in range(num_samples):
        t_true = target[i]
        p_score = pred[i]

        if np.sum(t_true) == 0:
            # Skip samples without positive labels
            iou[i] = np.nan
            auc[i] = np.nan
            continue

        # Compute AUC safely
        try:
            auc[i] = roc_auc_score(t_true, p_score)
        except ValueError:
            auc[i] = np.nan

        # Compute averaged IoU across thresholds
        temp_iou = []
        for thr in thresholds:
            p_mask = (p_score >= thr).astype(int)
            intersect = np.sum(p_mask & t_true)
            union = np.sum(p_mask | t_true)
            temp_iou.append(1.0 * intersect / union if union > 0 else 0.0)

        iou[i] = np.mean(temp_iou)

    return iou, auc


def calculate_batch_sim(pred, target):
    """Compute histogram intersection similarity."""
    sim = np.minimum(
        pred / (np.sum(pred, axis=1, keepdims=True) + 1e-12),
        target / (np.sum(target, axis=1, keepdims=True) + 1e-12)
    )
    return sim.sum(-1)


def calculate_batch_mae(pred, target):
    """Compute mean absolute error."""
    return np.mean(np.abs(pred - target), axis=1)


def calculate_batch_iou(results: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute IoU for each instance in a batch by averaging over multiple thresholds.
    """
    iou = np.zeros(results.shape[0])
    IOU_thres = np.linspace(0, 1, 20)
    targets = (targets >= 0.5).astype(int)

    for i in range(results.shape[0]):
        t_true = targets[i]
        p_score = results[i]
        if np.sum(t_true) == 0:
            iou[i] = np.nan
            continue

        vals = []
        for thre in IOU_thres:
            p_mask = (p_score >= thre).astype(int)
            intersect = np.sum(p_mask & t_true)
            union = np.sum(p_mask | t_true)
            vals.append(0.0 if union == 0 else (1.0 * intersect / union))
        iou[i] = float(np.mean(vals))
    return iou