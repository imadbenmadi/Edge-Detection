# ============================================
# LOSS + METRICS (FINAL, STABLE, CORRECT)
# ============================================

import numpy as np
import cv2
from sklearn.metrics import average_precision_score

# ------------------------------------------------
# 1. LOSS FUNCTION (Binary Cross-Entropy)
# ------------------------------------------------
class EdgeLoss(nn.Module):
    """
    Standard BCE loss for binary edge maps.
    Works best for 1-pixel hard GT edges.
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
    
    def forward(self, pred, label):
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        loss = -(label * torch.log(pred) + (1 - label) * torch.log(1 - pred))
        return loss.mean()

# ------------------------------------------------
# 2. NMS for predictions ONLY (allowed)
# ------------------------------------------------
def nms_edge(pred):
    """
    Non-Maximum Suppression surrogate using Canny.
    Does NOT modify GT. Only thins predicted edges.
    """
    pred_blur = cv2.GaussianBlur(pred, (3,3), 0)
    pred_uint8 = (np.clip(pred_blur, 0, 1) * 255).astype(np.uint8)
    
    # Use Canny as a stable thinning proxy
    canny = cv2.Canny(pred_uint8, 50, 150)
    return canny.astype(np.float32) / 255.0

# ------------------------------------------------
# 3. GT TOLERANCE (matching only, GT unchanged)
# ------------------------------------------------
def dilate_gt(gt, r=1):
    """
    Adds 1-pixel tolerance to GT *for matching only*.
    GT image itself remains unchanged.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    return cv2.dilate(gt, kernel)

# ------------------------------------------------
# 4. METRICS: ODS / OIS / AP
# ------------------------------------------------
def compute_ods_ois_ap(preds, labels, thresholds=30):
    """
    Computes:
    - ODS: best F1 score using one threshold globally
    - OIS: average of best F1 per image
    - AP: average precision from score curve
    """

    threshs = np.linspace(0.05, 0.95, thresholds)

    all_preds = []
    all_labels = []
    ois_f1_scores = []

    # --------------------------------------------
    # Per-image evaluation (for OIS)
    # --------------------------------------------
    for pred, label in zip(preds, labels):
        
        # 1-pixel tolerance for evaluation ONLY
        label_binary = (label > 0.5).astype(np.float32)
        label_tol = dilate_gt(label_binary, r=1).flatten()

        # Clean & stabilize prediction before thresholding
        pred_smooth = cv2.GaussianBlur(pred, (3,3), 0).flatten()

        all_preds.append(pred_smooth)
        all_labels.append(label_tol)

        # Best F1 for this image
        best_f1 = 0.0
        for t in threshs:
            pred_bin = (pred_smooth >= t).astype(np.float32)

            tp = np.sum(pred_bin * label_tol)
            fp = np.sum(pred_bin * (1 - label_tol))
            fn = np.sum((1 - pred_bin) * label_tol)

            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            best_f1 = max(best_f1, f1)

        ois_f1_scores.append(best_f1)

    # Flatten for ODS and AP
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # --------------------------------------------
    # Global threshold sweep (ODS)
    # --------------------------------------------
    best_ods = 0.0
    for t in threshs:
        pred_bin = (all_preds >= t).astype(np.float32)

        tp = np.sum(pred_bin * all_labels)
        fp = np.sum(pred_bin * (1 - all_labels))
        fn = np.sum((1 - pred_bin) * all_labels)

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        best_ods = max(best_ods, f1)

    # --------------------------------------------
    # OIS = mean of best F1 per image
    # --------------------------------------------
    ois = float(np.mean(ois_f1_scores))

    # --------------------------------------------
    # AP (area under precision-recall curve)
    # --------------------------------------------
    try:
        ap = float(average_precision_score(all_labels, all_preds))
    except:
        ap = 0.0

    return best_ods, ois, ap

print("Loss & Metrics (ODS/OIS/AP) loaded successfully.")
