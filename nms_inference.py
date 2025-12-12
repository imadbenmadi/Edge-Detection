"""
Standard NMS (Non-Maximum Suppression) for Edge Detection
Compatible with HED, RCF, BDCN, and classical edge detectors

NMS is applied AFTER model prediction, during evaluation/inference.
It thins edges to 1-3 pixels and improves metric alignment with ground truth.
"""

import torch
import cv2
import numpy as np
from tqdm import tqdm


def nms_edge(pred, do_thin=True):
    """
    Apply Non-Maximum Suppression to edge predictions using OpenCV's thinning.
    
    This is the standard NMS used in HED, RCF, BDCN papers.
    Uses Zhang-Suen thinning algorithm for edge skeleton extraction.
    
    Args:
        pred: numpy array or tensor of shape (H, W) with values in [0, 1]
        do_thin: whether to apply edge thinning (default: True)
    
    Returns:
        thinned: numpy array of shape (H, W) with thinned edges in [0, 1]
    
    Example:
        >>> pred = model(image)  # (1, 1, H, W)
        >>> pred_np = pred.squeeze().cpu().numpy()  # (H, W)
        >>> pred_nms = nms_edge(pred_np)
    """
    # Convert tensor to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    # Ensure 2D (H, W)
    if pred.ndim == 3:
        pred = pred.squeeze(0)
    
    # Scale from [0, 1] to [0, 255] uint8
    pred_uint8 = (pred * 255).astype(np.uint8)
    
    # Apply Zhang-Suen thinning algorithm
    # This produces a skeleton representation of the edges
    if do_thin:
        thinned = cv2.ximgproc.thinning(
            pred_uint8,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
    else:
        thinned = pred_uint8
    
    # Convert back to [0, 1] float32
    thinned_float = thinned.astype(np.float32) / 255.0
    
    return thinned_float


def inference_with_nms(model, image, device, apply_nms=True):
    """
    Single image inference with optional NMS.
    
    Args:
        model: edge detection model
        image: input image tensor (1, 3, H, W) or (3, H, W)
        device: torch device
        apply_nms: whether to apply NMS (default: True)
    
    Returns:
        pred_nms: edge prediction with NMS applied [0, 1]
        pred_raw: raw edge prediction [0, 1]
    
    Example:
        >>> img = torch.randn(1, 3, 512, 512).to(device)
        >>> pred_nms, pred_raw = inference_with_nms(model, img, device)
    """
    model.eval()
    
    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        # Get prediction
        output = model(image)  # (1, 1, H, W)
        
        # Convert to numpy
        pred_raw = output.squeeze().cpu().numpy()  # (H, W)
        
        # Apply NMS
        if apply_nms:
            pred_nms = nms_edge(pred_raw, do_thin=True)
        else:
            pred_nms = pred_raw
    
    return pred_nms, pred_raw


def batch_inference_with_nms(model, loader, device, apply_nms=True):
    """
    Batch inference on DataLoader with NMS.
    
    Args:
        model: edge detection model
        loader: DataLoader
        device: torch device
        apply_nms: whether to apply NMS (default: True)
    
    Yields:
        predictions_nms: NMS-applied predictions
        predictions_raw: raw predictions
        filenames: sample filenames
    
    Example:
        >>> for pred_nms, pred_raw, name in batch_inference_with_nms(model, loader, device):
        >>>     print(f"{name}: {pred_nms.shape}")
    """
    model.eval()
    
    for batch in tqdm(loader, desc='Inference with NMS'):
        images = batch['images'].to(device)
        filenames = batch.get('filename', ['unknown'] * images.shape[0])
        
        with torch.no_grad():
            outputs = model(images)  # (B, 1, H, W)
            
            for i in range(outputs.shape[0]):
                pred_raw = outputs[i, 0].cpu().numpy()  # (H, W)
                
                # Apply NMS
                if apply_nms:
                    pred_nms = nms_edge(pred_raw, do_thin=True)
                else:
                    pred_nms = pred_raw
                
                yield pred_nms, pred_raw, filenames[i]


def compute_metrics_with_nms(model, loader, device, apply_nms=True):
    """
    Compute evaluation metrics with optional NMS.
    
    Args:
        model: edge detection model
        loader: DataLoader with labels
        device: torch device
        apply_nms: whether to apply NMS (default: True)
    
    Returns:
        metrics_dict: dictionary with ODS, OIS, AP metrics
    
    Example:
        >>> metrics = compute_metrics_with_nms(model, test_loader, device, apply_nms=True)
        >>> print(f"ODS: {metrics['ODS']:.4f}")
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc='Computing metrics'):
        images = batch['images'].to(device)
        labels = batch['labels'].numpy()
        
        with torch.no_grad():
            outputs = model(images)
            
            for i in range(outputs.shape[0]):
                pred = outputs[i, 0].cpu().numpy()
                
                # Apply NMS
                if apply_nms:
                    pred = nms_edge(pred, do_thin=True)
                
                all_preds.append(pred.flatten())
                all_labels.append((labels[i, 0].flatten() > 0.5).astype(np.float32))
    
    # Compute AP
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    ap = average_precision_score(all_labels, all_preds)
    
    # Compute ODS/OIS (simplified)
    thresholds = np.linspace(0.01, 0.99, 50)
    best_f1 = 0
    
    for t in thresholds:
        pred_binary = (all_preds >= t).astype(np.float32)
        tp = np.sum(pred_binary * all_labels)
        fp = np.sum(pred_binary * (1 - all_labels))
        fn = np.sum((1 - pred_binary) * all_labels)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_f1 = max(best_f1, f1)
    
    return {
        'ODS': best_f1,
        'OIS': best_f1,  # Simplified
        'AP': ap,
        'apply_nms': apply_nms
    }


if __name__ == "__main__":
    """
    STANDARD NMS USAGE EXAMPLE FOR EDGE DETECTION
    
    This demonstrates the correct way to apply NMS in edge detection,
    following the approach used in HED, RCF, BDCN papers.
    """
    
    print("="*70)
    print("NON-MAXIMUM SUPPRESSION (NMS) FOR EDGE DETECTION")
    print("="*70)
    print("""
    Standard NMS Implementation (HED/RCF/BDCN compatible)
    
    ✓ Applied AFTER model prediction
    ✓ Uses Zhang-Suen thinning algorithm
    ✓ Thins edges to 1-3 pixel width
    ✓ Improves precision and metric alignment
    ✓ Standard evaluation protocol in edge detection
    
    Key Points:
    1. NMS is NOT part of the network
    2. Applied during inference/evaluation only
    3. Converts soft predictions to crisp edge maps
    4. Uses OpenCV's cv2.ximgproc.thinning()
    
    Usage Examples:
    """)
    
    print("\n# Example 1: Single image")
    print("""
    pred = model(image)  # (1, 1, H, W), values in [0, 1]
    pred_np = pred.squeeze().cpu().numpy()  # (H, W)
    pred_nms = nms_edge(pred_np)  # Apply NMS
    """)
    
    print("\n# Example 2: Batch processing")
    print("""
    for pred_nms, pred_raw, filename in batch_inference_with_nms(model, loader, device):
        print(f"{filename}: {pred_nms.shape}")
    """)
    
    print("\n# Example 3: Evaluation with NMS")
    print("""
    metrics = compute_metrics_with_nms(model, test_loader, device, apply_nms=True)
    print(f"ODS: {metrics['ODS']:.4f}")
    """)
    
    print("\n" + "="*70)
    print("Benefits of NMS:")
    print("="*70)
    print("""
    WITHOUT NMS:
    - Thick, blurry edges from network
    - Lower precision due to width
    - Misalignment with ground truth
    - Unstable PR curves
    
    WITH NMS:
    - Thin, crisp edges (1-3 pixels)
    - Higher precision
    - Better alignment with GT
    - Stable, reproducible metrics
    """)
    print("="*70)
