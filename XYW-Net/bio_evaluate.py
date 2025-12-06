"""
Bio-XYW-Net Evaluation Script
=============================

Evaluates models using standard edge detection metrics:
- ODS (Optimal Dataset Scale): Best F-score across all thresholds
- OIS (Optimal Image Scale): Per-image best threshold, then averaged
- AP (Average Precision): Area under precision-recall curve

Also computes:
- Model parameters
- FLOPs (Floating Point Operations)
- Inference speed
- Robustness to distortions

Usage:
    python bio_evaluate.py --model bio --checkpoint bio_model.pth --dataset BSDS500
    python bio_evaluate.py --compare --dataset BIPED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import json

from bio_model import BioXYWNet, BaselineXYWNet
from bio_frontend import BioFrontend


# ============================================================================
# FLOP COUNTER
# ============================================================================

def count_flops(model, input_size=(1, 3, 256, 256)):
    """Estimate FLOPs for model"""
    try:
        from thop import profile, clever_format
        
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        
        return flops, params
    except ImportError:
        print("Warning: thop not installed, skipping FLOP counting")
        return None, None


# ============================================================================
# EDGE DETECTION METRICS
# ============================================================================

def compute_f_score(precision, recall, beta=1.0):
    """Compute F-score"""
    if precision.sum() == 0 or recall.sum() == 0:
        return 0.0
    
    p = precision.mean()
    r = recall.mean()
    
    if p + r == 0:
        return 0.0
    
    f = (1 + beta**2) * (p * r) / ((beta**2 * p) + r + 1e-8)
    return f


def evaluate_single_image(pred_map, gt_map, thresholds=None):
    """Evaluate single image against ground truth
    
    Args:
        pred_map: (H, W) predicted edge map in [0, 1]
        gt_map: (H, W) binary ground truth {0, 1}
        thresholds: List of thresholds to evaluate
        
    Returns:
        dict with precision, recall, F-scores at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.01)
    
    pred_flat = pred_map.flatten()
    gt_flat = gt_map.flatten()
    
    results = {
        'precision': [],
        'recall': [],
        'f_score': []
    }
    
    for thresh in thresholds:
        pred_binary = (pred_flat > thresh).astype(np.float32)
        
        tp = np.sum(pred_binary * gt_flat)
        fp = np.sum(pred_binary * (1 - gt_flat))
        fn = np.sum((1 - pred_binary) * gt_flat)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f_score'].append(f_score)
    
    return {
        'precision': np.array(results['precision']),
        'recall': np.array(results['recall']),
        'f_score': np.array(results['f_score']),
        'best_f_score': np.max(results['f_score']),
        'best_threshold': thresholds[np.argmax(results['f_score'])]
    }


def compute_ods(all_results):
    """Compute ODS: Best F-score across all images at single threshold"""
    all_precisions = []
    all_recalls = []
    
    for result in all_results:
        all_precisions.extend(result['precision'].tolist())
        all_recalls.extend(result['recall'].tolist())
    
    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)
    
    # Find threshold that maximizes F-score
    f_scores = 2 * all_precisions * all_recalls / (all_precisions + all_recalls + 1e-8)
    best_idx = np.argmax(f_scores)
    
    ods = f_scores[best_idx]
    ods_precision = all_precisions[best_idx]
    ods_recall = all_recalls[best_idx]
    
    return {
        'ods': ods,
        'precision': ods_precision,
        'recall': ods_recall
    }


def compute_ois(all_results):
    """Compute OIS: Per-image F-score optimized, then averaged"""
    f_scores = [r['best_f_score'] for r in all_results]
    ois = np.mean(f_scores)
    
    return {
        'ois': ois,
        'f_scores': f_scores
    }


def compute_ap(all_results):
    """Compute AP: Area under Precision-Recall curve"""
    all_precisions = []
    all_recalls = []
    
    for result in all_results:
        all_precisions.extend(result['precision'].tolist())
        all_recalls.extend(result['recall'].tolist())
    
    # Sort by recall
    sorted_idx = np.argsort(all_recalls)
    all_recalls = np.array(all_recalls)[sorted_idx]
    all_precisions = np.array(all_precisions)[sorted_idx]
    
    # Compute AP as area under PR curve
    ap = np.trapz(all_precisions, all_recalls)
    
    return {'ap': ap}


# ============================================================================
# MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluator"""
    
    def __init__(self, model, device, model_name="Model"):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.model.eval()
        
        # Count parameters
        self.params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def evaluate_image(self, image_tensor, gt_tensor):
        """Evaluate on single image"""
        image_tensor = image_tensor.to(self.device)
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            output = self.model(image_tensor)
            if isinstance(output, tuple):
                output = output[0]
            output = torch.sigmoid(output)
        inference_time = time.time() - start_time
        
        # Resize to match GT if needed
        if output.shape[-2:] != gt_tensor.shape[-2:]:
            output = F.interpolate(
                output, size=gt_tensor.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        pred_np = output[0, 0].cpu().numpy()
        gt_np = gt_tensor[0, 0].numpy()
        
        results = evaluate_single_image(pred_np, gt_np)
        
        return results, inference_time, pred_np
    
    def evaluate_dataset(self, image_list, gt_list, max_images=None):
        """Evaluate on dataset"""
        print(f"\nEvaluating {self.model_name}...")
        
        all_results = []
        all_times = []
        
        num_images = min(len(image_list), max_images) if max_images else len(image_list)
        
        for i in tqdm(range(num_images), desc=f"Processing images"):
            # Load image
            img_path = image_list[i]
            if not Path(img_path).exists():
                print(f"Warning: Image not found {img_path}")
                continue
            
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            
            # Load GT
            gt_path = gt_list[i]
            if not Path(gt_path).exists():
                continue
            
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            gt = (gt > 0.5).astype(np.float32)
            gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
            
            # Resize to standard size for evaluation
            h, w = gt_tensor.shape[-2:]
            img_tensor = F.interpolate(img_tensor, size=(h, w), mode='bilinear', align_corners=False)
            
            # Evaluate
            results, inf_time, _ = self.evaluate_image(img_tensor, gt_tensor)
            all_results.append(results)
            all_times.append(inf_time)
        
        # Compute metrics
        ods_result = compute_ods(all_results)
        ois_result = compute_ois(all_results)
        ap_result = compute_ap(all_results)
        
        metrics = {
            'ods': ods_result['ods'],
            'ods_precision': ods_result['precision'],
            'ods_recall': ods_result['recall'],
            'ois': ois_result['ois'],
            'ap': ap_result['ap'],
            'avg_inference_time_ms': np.mean(all_times) * 1000,
            'fps': 1 / np.mean(all_times)
        }
        
        return metrics
    
    def print_summary(self, metrics=None):
        """Print model summary"""
        print(f"\n{'='*60}")
        print(f"MODEL: {self.model_name}")
        print(f"{'='*60}")
        print(f"Total parameters: {self.params:,}")
        print(f"Trainable parameters: {self.trainable_params:,}")
        
        if metrics:
            print(f"\nEvaluation Metrics:")
            print(f"  ODS: {metrics.get('ods', 0):.4f}")
            print(f"  OIS: {metrics.get('ois', 0):.4f}")
            print(f"  AP:  {metrics.get('ap', 0):.4f}")
            print(f"  Avg inference: {metrics.get('avg_inference_time_ms', 0):.2f}ms")
            print(f"  FPS: {metrics.get('fps', 0):.2f}")
        
        print(f"{'='*60}\n")


# ============================================================================
# ROBUSTNESS TESTS
# ============================================================================

def add_gaussian_noise(image, std=0.1):
    """Add Gaussian noise"""
    noise = np.random.normal(0, std, image.shape)
    noisy = np.clip(image + noise, 0, 1)
    return noisy


def adjust_illumination(image, gamma=1.5):
    """Adjust illumination (gamma correction)"""
    adjusted = np.power(image, 1.0 / gamma)
    return adjusted


def adjust_contrast(image, factor=1.5):
    """Adjust contrast"""
    adjusted = image * factor
    adjusted = np.clip(adjusted, 0, 1)
    return adjusted


def add_blur(image, kernel_size=5):
    """Add blur"""
    img_uint8 = (image * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
    return blurred.astype(np.float32) / 255.0


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_models(config):
    """Main evaluation function"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # For demonstration, create dummy datasets
    print(f"\nNote: This is a template evaluation script.")
    print(f"You need to provide actual image and GT paths for evaluation.")
    
    # Load models
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    
    # Bio-XYW-Net
    bio_model = BioXYWNet(use_learnable_bio=config.get('use_learnable_bio', False)).to(device)
    if config.get('bio_checkpoint'):
        checkpoint = torch.load(config['bio_checkpoint'], map_location=device)
        if 'model_state_dict' in checkpoint:
            bio_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            bio_model.load_state_dict(checkpoint)
    
    bio_evaluator = ModelEvaluator(bio_model, device, "Bio-XYW-Net")
    bio_evaluator.print_summary()
    
    # Baseline
    baseline_model = BaselineXYWNet().to(device)
    if config.get('baseline_checkpoint'):
        try:
            checkpoint = torch.load(config['baseline_checkpoint'], map_location=device)
            if 'model_state_dict' in checkpoint:
                baseline_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                baseline_model.load_state_dict(checkpoint)
        except:
            pass
    
    baseline_evaluator = ModelEvaluator(baseline_model, device, "Baseline XYW-Net")
    baseline_evaluator.print_summary()
    
    # Example evaluation (needs actual data)
    # bio_metrics = bio_evaluator.evaluate_dataset(image_list, gt_list)
    # baseline_metrics = baseline_evaluator.evaluate_dataset(image_list, gt_list)
    
    # For now, print model info
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"Bio-XYW-Net params: {bio_evaluator.params:,}")
    print(f"Baseline params: {baseline_evaluator.params:,}")
    print(f"Parameter reduction: {(1 - bio_evaluator.params / baseline_evaluator.params) * 100:.1f}%")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Bio-XYW-Net')
    parser.add_argument('--bio_checkpoint', type=str, default=None, help='Bio model checkpoint')
    parser.add_argument('--baseline_checkpoint', type=str, default=None, help='Baseline checkpoint')
    parser.add_argument('--dataset', type=str, default='BSDS500', help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--use_learnable_bio', action='store_true', help='Use learnable bio params')
    parser.add_argument('--max_images', type=int, default=100, help='Max images to evaluate')
    parser.add_argument('--output_json', type=str, default='eval_results.json', help='Output JSON')
    
    args = parser.parse_args()
    config = vars(args)
    
    evaluate_models(config)
