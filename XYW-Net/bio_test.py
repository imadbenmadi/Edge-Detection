"""
Bio-XYW-Net Inference & Testing Script
=====================================

Performs inference with Bio-XYW-Net and compares with baseline XYW-Net.
Generates edge detection results and visualizations.

Usage:
    python bio_test.py --image test.jpg --model bio --checkpoint bio_model.pth
    python bio_test.py --image test.jpg --model baseline --checkpoint xyw.pth --compare
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import time

from bio_model import BioXYWNet, BaselineXYWNet
from bio_frontend import BioFrontend


def load_image(image_path, size=None):
    """Load and preprocess image"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Resize if specified
    if size is not None:
        image = cv2.resize(image, (size, size))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_size


def inference(model, image_tensor, device):
    """Run inference"""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        start_time = time.time()
        output = model(image_tensor)
        inference_time = time.time() - start_time
        
        # Handle tuple output
        if isinstance(output, tuple):
            output = output[0]
        
        # Normalize output
        output = torch.sigmoid(output)
    
    return output.cpu(), inference_time


def preprocess_for_visualization(edge_map):
    """Convert edge map to uint8 for visualization"""
    edge_map = edge_map.squeeze().numpy()
    edge_map = (edge_map * 255).astype(np.uint8)
    
    # Optional: apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return edge_map


def save_comparison_image(original, bio_edges, baseline_edges, save_path):
    """Save 3-panel comparison image"""
    # Convert to uint8
    if isinstance(original, torch.Tensor):
        original = (original.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        original = (original * 255).astype(np.uint8) if original.max() <= 1 else original.astype(np.uint8)
    
    if isinstance(bio_edges, torch.Tensor):
        bio_edges = preprocess_for_visualization(bio_edges)
    else:
        bio_edges = (bio_edges * 255).astype(np.uint8) if bio_edges.max() <= 1 else bio_edges.astype(np.uint8)
    
    if isinstance(baseline_edges, torch.Tensor):
        baseline_edges = preprocess_for_visualization(baseline_edges)
    else:
        baseline_edges = (baseline_edges * 255).astype(np.uint8) if baseline_edges.max() <= 1 else baseline_edges.astype(np.uint8)
    
    # Resize edges to match original
    h, w = original.shape[:2]
    bio_edges = cv2.resize(bio_edges, (w, h))
    baseline_edges = cv2.resize(baseline_edges, (w, h))
    
    # Convert grayscale to RGB for display
    bio_edges_rgb = cv2.cvtColor(bio_edges, cv2.COLOR_GRAY2RGB)
    baseline_edges_rgb = cv2.cvtColor(baseline_edges, cv2.COLOR_GRAY2RGB)
    
    # Stack horizontally
    top_row = np.hstack([original, bio_edges_rgb])
    bottom_row = np.hstack([baseline_edges_rgb, np.zeros_like(baseline_edges_rgb)])
    
    # Stack vertically
    comparison = np.vstack([top_row, bottom_row])
    
    # Save
    cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"Saved comparison image to {save_path}")


def test_bio_frontend(image_tensor, device):
    """Test and visualize bio-frontend"""
    print("\nTesting Bio-Frontend processing...")
    
    bio_frontend = BioFrontend(sigma=2.0, kernel_size=5, return_intermediate=True).to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        bio_output, intermediates = bio_frontend(image_tensor)
    
    print(f"  Input shape: {image_tensor.shape}")
    print(f"  Photoreceptor output shape: {intermediates['photoreceptor'].shape}")
    print(f"  Surround response shape: {intermediates['surround'].shape}")
    print(f"  ON pathway shape: {intermediates['on'].shape}")
    print(f"  OFF pathway shape: {intermediates['off'].shape}")
    print(f"  Bio-processed output shape: {bio_output.shape}")
    
    # Visualize pathways
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Original RGB
    orig = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    axes[0, 0].imshow(orig)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Photoreceptor
    photor = intermediates['photoreceptor'][0, 0].cpu().numpy()
    axes[0, 1].imshow(photor, cmap='gray')
    axes[0, 1].set_title("Photoreceptor Adapted")
    axes[0, 1].axis('off')
    
    # Surround
    surround = intermediates['surround'][0, 0].cpu().numpy()
    axes[0, 2].imshow(surround, cmap='gray')
    axes[0, 2].set_title("Horizontal Cell Surround")
    axes[0, 2].axis('off')
    
    # ON pathway
    on = intermediates['on'][0, 0].cpu().numpy()
    axes[1, 0].imshow(on, cmap='hot')
    axes[1, 0].set_title("ON Bipolar")
    axes[1, 0].axis('off')
    
    # OFF pathway
    off = intermediates['off'][0, 0].cpu().numpy()
    axes[1, 1].imshow(off, cmap='cool')
    axes[1, 1].set_title("OFF Bipolar")
    axes[1, 1].axis('off')
    
    # Both combined
    combined = (on + off) / 2
    axes[1, 2].imshow(combined, cmap='gray')
    axes[1, 2].set_title("ON+OFF Combined")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('bio_frontend_visualization.png', dpi=150, bbox_inches='tight')
    print("  Saved bio-frontend visualization to bio_frontend_visualization.png")
    plt.close()


def compare_models(image_path, bio_checkpoint=None, baseline_checkpoint=None, output_dir=None, visualize_frontend=False):
    """Compare Bio-XYW-Net with Baseline"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image_tensor, original_size = load_image(image_path, size=512)
    print(f"  Original size: {original_size}")
    print(f"  Resized to: {image_tensor.shape}")
    
    # Optionally visualize bio-frontend
    if visualize_frontend:
        test_bio_frontend(image_tensor, device)
    
    # Create output directory
    if output_dir is None:
        output_dir = Path('./test_results')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Bio-XYW-Net
    print("\nLoading Bio-XYW-Net...")
    bio_model = BioXYWNet(use_learnable_bio=False).to(device)
    if bio_checkpoint and Path(bio_checkpoint).exists():
        checkpoint = torch.load(bio_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            bio_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            bio_model.load_state_dict(checkpoint)
        print(f"  Loaded checkpoint: {bio_checkpoint}")
    else:
        print("  No checkpoint provided, using random initialization for demo")
    
    bio_params = sum(p.numel() for p in bio_model.parameters())
    print(f"  Parameters: {bio_params:,}")
    
    # Run inference
    print("  Running inference...")
    bio_output, bio_time = inference(bio_model, image_tensor, device)
    print(f"  Inference time: {bio_time*1000:.2f}ms")
    print(f"  Output shape: {bio_output.shape}")
    print(f"  Output range: [{bio_output.min():.3f}, {bio_output.max():.3f}]")
    
    # Load Baseline XYW-Net
    print("\nLoading Baseline XYW-Net...")
    # For demo, we create a baseline model (in real case load from xyw.pth)
    from model import Net
    try:
        baseline_model = Net().to(device)
        if baseline_checkpoint and Path(baseline_checkpoint).exists():
            baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device))
            print(f"  Loaded checkpoint: {baseline_checkpoint}")
        else:
            print("  No checkpoint provided, using random initialization")
    except:
        print("  Could not load original XYW-Net, creating baseline for comparison")
        baseline_model = BaselineXYWNet().to(device)
    
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"  Parameters: {baseline_params:,}")
    
    # Run inference
    print("  Running inference...")
    baseline_output, baseline_time = inference(baseline_model, image_tensor, device)
    print(f"  Inference time: {baseline_time*1000:.2f}ms")
    print(f"  Output shape: {baseline_output.shape}")
    print(f"  Output range: [{baseline_output.min():.3f}, {baseline_output.max():.3f}]")
    
    # Resize outputs to match
    if bio_output.shape != baseline_output.shape:
        baseline_output = F.interpolate(
            baseline_output, size=bio_output.shape[-2:],
            mode='bilinear', align_corners=False
        )
    
    # Save individual results
    print("\nSaving results...")
    
    bio_edge = preprocess_for_visualization(bio_output)
    cv2.imwrite(str(output_dir / 'bio_edges.png'), bio_edge)
    print(f"  Saved Bio-XYW-Net edges to {output_dir / 'bio_edges.png'}")
    
    baseline_edge = preprocess_for_visualization(baseline_output)
    cv2.imwrite(str(output_dir / 'baseline_edges.png'), baseline_edge)
    print(f"  Saved Baseline edges to {output_dir / 'baseline_edges.png'}")
    
    # Save original for reference
    orig_np = image_tensor[0].permute(1, 2, 0).numpy()
    cv2.imwrite(str(output_dir / 'original.png'), cv2.cvtColor((orig_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    orig = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    bio_edges_np = bio_output[0, 0].numpy()
    baseline_edges_np = baseline_output[0, 0].numpy()
    
    axes[0].imshow(orig)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(bio_edges_np, cmap='gray')
    axes[1].set_title(f"Bio-XYW-Net\n({bio_params:,} params, {bio_time*1000:.1f}ms)", 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(baseline_edges_np, cmap='gray')
    axes[2].set_title(f"Baseline XYW-Net\n({baseline_params:,} params, {baseline_time*1000:.1f}ms)", 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    comparison_path = output_dir / 'comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"  Saved comparison figure to {comparison_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("COMPARISON STATISTICS")
    print("="*60)
    print(f"Bio-XYW-Net:")
    print(f"  Parameters: {bio_params:,}")
    print(f"  Inference time: {bio_time*1000:.2f}ms")
    print(f"  Output mean: {bio_output.mean():.4f}")
    print(f"  Output std: {bio_output.std():.4f}")
    
    print(f"\nBaseline XYW-Net:")
    print(f"  Parameters: {baseline_params:,}")
    print(f"  Inference time: {baseline_time*1000:.2f}ms")
    print(f"  Output mean: {baseline_output.mean():.4f}")
    print(f"  Output std: {baseline_output.std():.4f}")
    
    print(f"\nDifference map (Bio - Baseline):")
    diff = (bio_edges_np - baseline_edges_np)
    print(f"  Mean difference: {diff.mean():.4f}")
    print(f"  Std difference: {diff.std():.4f}")
    print(f"  Max difference: {diff.max():.4f}")
    print(f"  Min difference: {diff.min():.4f}")
    
    # Speedup
    speedup = baseline_time / bio_time
    param_reduction = (1 - bio_params / baseline_params) * 100
    print(f"\nSpeedup (Baseline / Bio): {speedup:.2f}x")
    if param_reduction > 0:
        print(f"Parameter reduction: {param_reduction:.1f}%")
    else:
        print(f"Parameter increase: {abs(param_reduction):.1f}%")
    
    print("="*60 + "\n")
    
    return {
        'bio': {
            'output': bio_output,
            'edges': bio_edge,
            'params': bio_params,
            'time': bio_time
        },
        'baseline': {
            'output': baseline_output,
            'edges': baseline_edge,
            'params': baseline_params,
            'time': baseline_time
        },
        'diff': diff
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Bio-XYW-Net edge detection')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--bio_checkpoint', type=str, default=None, help='Bio-XYW-Net checkpoint')
    parser.add_argument('--baseline_checkpoint', type=str, default='./xyw.pth', help='Baseline checkpoint')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory')
    parser.add_argument('--visualize_frontend', action='store_true', help='Visualize bio-frontend stages')
    
    args = parser.parse_args()
    
    results = compare_models(
        image_path=args.image,
        bio_checkpoint=args.bio_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        output_dir=args.output_dir,
        visualize_frontend=args.visualize_frontend
    )
