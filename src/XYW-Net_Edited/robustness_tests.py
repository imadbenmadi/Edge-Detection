"""
Bio-XYW-Net Robustness Testing
=============================

Tests model robustness to various image distortions:
- Illumination changes (gamma correction)
- Gaussian noise
- Contrast variations
- Blur
- JPEG compression

Usage:
    python robustness_tests.py --image test.jpg --checkpoint bio_model.pth
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import time

from bio_model import BioXYWNet, BaselineXYWNet


class RobustnessTest:
    """Robustness testing framework"""
    
    def __init__(self, model, device, model_name="Model"):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.model.eval()
    
    def preprocess_image(self, image_np):
        """Convert numpy image to tensor"""
        if image_np.max() > 1:
            image_np = image_np.astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        return image_tensor
    
    def run_inference(self, image_tensor):
        """Run inference"""
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            if isinstance(output, tuple):
                output = output[0]
            output = torch.sigmoid(output)
        
        return output.cpu()
    
    def apply_illumination_change(self, image, gamma=1.5):
        """Apply gamma correction for illumination change"""
        image = np.power(image, 1.0 / gamma)
        return np.clip(image, 0, 1)
    
    def apply_gaussian_noise(self, image, std=0.1):
        """Add Gaussian noise"""
        noise = np.random.normal(0, std, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)
    
    def apply_contrast_change(self, image, factor=1.5):
        """Adjust contrast"""
        mean = image.mean()
        adjusted = mean + factor * (image - mean)
        return np.clip(adjusted, 0, 1)
    
    def apply_blur(self, image, kernel_size=5):
        """Apply Gaussian blur"""
        img_uint8 = (image * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
        return blurred.astype(np.float32) / 255.0
    
    def apply_jpeg_compression(self, image, quality=30):
        """Apply JPEG compression"""
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Encode
        _, encimg = cv2.imencode('.jpg', img_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
        # Decode
        img_decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        
        return img_decoded.astype(np.float32) / 255.0
    
    def test_illumination_robustness(self, image_np, gammas=[0.5, 0.75, 1.0, 1.25, 1.5]):
        """Test robustness to illumination changes"""
        print(f"\n{self.model_name}: Testing illumination robustness...")
        
        results = {
            'gamma': [],
            'output': [],
            'variance': []
        }
        
        # Get baseline output
        baseline_tensor = self.preprocess_image(image_np)
        baseline_output = self.run_inference(baseline_tensor)
        baseline_np = baseline_output[0, 0].numpy()
        
        for gamma in gammas:
            modified = self.apply_illumination_change(image_np.copy(), gamma=gamma)
            modified_tensor = self.preprocess_image(modified)
            modified_output = self.run_inference(modified_tensor)
            modified_np = modified_output[0, 0].numpy()
            
            # Compute difference from baseline
            diff = np.abs(modified_np - baseline_np).mean()
            
            results['gamma'].append(gamma)
            results['output'].append(modified_np)
            results['variance'].append(diff)
        
        return results
    
    def test_noise_robustness(self, image_np, noise_stds=[0.01, 0.05, 0.1, 0.15, 0.2]):
        """Test robustness to Gaussian noise"""
        print(f"{self.model_name}: Testing noise robustness...")
        
        results = {
            'noise_std': [],
            'output': [],
            'variance': []
        }
        
        # Get baseline output
        baseline_tensor = self.preprocess_image(image_np)
        baseline_output = self.run_inference(baseline_tensor)
        baseline_np = baseline_output[0, 0].numpy()
        
        for std in noise_stds:
            modified = self.apply_gaussian_noise(image_np.copy(), std=std)
            modified_tensor = self.preprocess_image(modified)
            modified_output = self.run_inference(modified_tensor)
            modified_np = modified_output[0, 0].numpy()
            
            diff = np.abs(modified_np - baseline_np).mean()
            
            results['noise_std'].append(std)
            results['output'].append(modified_np)
            results['variance'].append(diff)
        
        return results
    
    def test_contrast_robustness(self, image_np, factors=[0.5, 0.75, 1.0, 1.25, 1.5]):
        """Test robustness to contrast changes"""
        print(f"{self.model_name}: Testing contrast robustness...")
        
        results = {
            'factor': [],
            'output': [],
            'variance': []
        }
        
        # Get baseline output
        baseline_tensor = self.preprocess_image(image_np)
        baseline_output = self.run_inference(baseline_tensor)
        baseline_np = baseline_output[0, 0].numpy()
        
        for factor in factors:
            modified = self.apply_contrast_change(image_np.copy(), factor=factor)
            modified_tensor = self.preprocess_image(modified)
            modified_output = self.run_inference(modified_tensor)
            modified_np = modified_output[0, 0].numpy()
            
            diff = np.abs(modified_np - baseline_np).mean()
            
            results['factor'].append(factor)
            results['output'].append(modified_np)
            results['variance'].append(diff)
        
        return results
    
    def test_blur_robustness(self, image_np, kernel_sizes=[3, 5, 7, 9, 11]):
        """Test robustness to blur"""
        print(f"{self.model_name}: Testing blur robustness...")
        
        results = {
            'kernel_size': [],
            'output': [],
            'variance': []
        }
        
        # Get baseline output
        baseline_tensor = self.preprocess_image(image_np)
        baseline_output = self.run_inference(baseline_tensor)
        baseline_np = baseline_output[0, 0].numpy()
        
        for ks in kernel_sizes:
            modified = self.apply_blur(image_np.copy(), kernel_size=ks)
            modified_tensor = self.preprocess_image(modified)
            modified_output = self.run_inference(modified_tensor)
            modified_np = modified_output[0, 0].numpy()
            
            diff = np.abs(modified_np - baseline_np).mean()
            
            results['kernel_size'].append(ks)
            results['output'].append(modified_np)
            results['variance'].append(diff)
        
        return results
    
    def test_jpeg_robustness(self, image_np, qualities=[10, 20, 30, 50, 80]):
        """Test robustness to JPEG compression"""
        print(f"{self.model_name}: Testing JPEG robustness...")
        
        results = {
            'quality': [],
            'output': [],
            'variance': []
        }
        
        # Get baseline output
        baseline_tensor = self.preprocess_image(image_np)
        baseline_output = self.run_inference(baseline_tensor)
        baseline_np = baseline_output[0, 0].numpy()
        
        for quality in qualities:
            modified = self.apply_jpeg_compression(image_np.copy(), quality=quality)
            modified_tensor = self.preprocess_image(modified)
            modified_output = self.run_inference(modified_tensor)
            modified_np = modified_output[0, 0].numpy()
            
            diff = np.abs(modified_np - baseline_np).mean()
            
            results['quality'].append(quality)
            results['output'].append(modified_np)
            results['variance'].append(diff)
        
        return results


def plot_robustness_comparison(bio_results, baseline_results, distortion_name, parameter_name, save_path):
    """Plot robustness comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Extract parameters and variances
    if distortion_name == 'illumination':
        params = bio_results['gamma']
        param_label = 'Gamma'
    elif distortion_name == 'noise':
        params = bio_results['noise_std']
        param_label = 'Noise Std'
    elif distortion_name == 'contrast':
        params = bio_results['factor']
        param_label = 'Contrast Factor'
    elif distortion_name == 'blur':
        params = bio_results['kernel_size']
        param_label = 'Kernel Size'
    elif distortion_name == 'jpeg':
        params = bio_results['quality']
        param_label = 'JPEG Quality'
    
    bio_var = bio_results['variance']
    baseline_var = baseline_results['variance']
    
    # Plot 1: Robustness curves
    axes[0].plot(params, bio_var, 'b-o', label='Bio-XYW-Net', linewidth=2, markersize=6)
    axes[0].plot(params, baseline_var, 'r-s', label='Baseline XYW-Net', linewidth=2, markersize=6)
    axes[0].set_xlabel(param_label, fontsize=12)
    axes[0].set_ylabel('Output Difference (L1)', fontsize=12)
    axes[0].set_title(f'{distortion_name.capitalize()} Robustness', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Edge outputs at extreme
    bio_extreme = bio_results['output'][-1]
    baseline_extreme = baseline_results['output'][-1]
    
    im1 = axes[1].imshow(bio_extreme, cmap='gray')
    axes[1].set_title(f'Bio-XYW-Net ({distortion_name} {params[-1]})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def run_robustness_tests(image_path, bio_checkpoint=None, baseline_checkpoint=None, output_dir=None):
    """Run all robustness tests"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if output_dir is None:
        output_dir = Path('./robustness_results')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = cv2.resize(image, (512, 512))  # Standardize size
    print(f"Image shape: {image.shape}")
    
    # Load models
    print("\nLoading models...")
    
    bio_model = BioXYWNet(use_learnable_bio=False).to(device)
    if bio_checkpoint and Path(bio_checkpoint).exists():
        checkpoint = torch.load(bio_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            bio_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            bio_model.load_state_dict(checkpoint)
    
    baseline_model = BaselineXYWNet().to(device)
    if baseline_checkpoint and Path(baseline_checkpoint).exists():
        checkpoint = torch.load(baseline_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            baseline_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            baseline_model.load_state_dict(checkpoint)
    
    bio_tester = RobustnessTest(bio_model, device, "Bio-XYW-Net")
    baseline_tester = RobustnessTest(baseline_model, device, "Baseline XYW-Net")
    
    print("\n" + "="*60)
    print("ROBUSTNESS TESTING")
    print("="*60)
    
    # Test 1: Illumination
    print("\n[1/5] Illumination Changes")
    bio_illum = bio_tester.test_illumination_robustness(image)
    baseline_illum = baseline_tester.test_illumination_robustness(image)
    plot_robustness_comparison(
        bio_illum, baseline_illum, 'illumination', 'gamma',
        output_dir / 'illumination_robustness.png'
    )
    
    # Test 2: Noise
    print("\n[2/5] Gaussian Noise")
    bio_noise = bio_tester.test_noise_robustness(image)
    baseline_noise = baseline_tester.test_noise_robustness(image)
    plot_robustness_comparison(
        bio_noise, baseline_noise, 'noise', 'std',
        output_dir / 'noise_robustness.png'
    )
    
    # Test 3: Contrast
    print("\n[3/5] Contrast Changes")
    bio_contrast = bio_tester.test_contrast_robustness(image)
    baseline_contrast = baseline_tester.test_contrast_robustness(image)
    plot_robustness_comparison(
        bio_contrast, baseline_contrast, 'contrast', 'factor',
        output_dir / 'contrast_robustness.png'
    )
    
    # Test 4: Blur
    print("\n[4/5] Gaussian Blur")
    bio_blur = bio_tester.test_blur_robustness(image)
    baseline_blur = baseline_tester.test_blur_robustness(image)
    plot_robustness_comparison(
        bio_blur, baseline_blur, 'blur', 'kernel_size',
        output_dir / 'blur_robustness.png'
    )
    
    # Test 5: JPEG
    print("\n[5/5] JPEG Compression")
    bio_jpeg = bio_tester.test_jpeg_robustness(image)
    baseline_jpeg = baseline_tester.test_jpeg_robustness(image)
    plot_robustness_comparison(
        bio_jpeg, baseline_jpeg, 'jpeg', 'quality',
        output_dir / 'jpeg_robustness.png'
    )
    
    print("\n" + "="*60)
    print("ROBUSTNESS TESTING COMPLETE")
    print("="*60)
    print(f"Results saved to {output_dir}")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"\nIllumination - Mean variance difference: {np.mean(np.array(bio_illum['variance']) - np.array(baseline_illum['variance'])):.6f}")
    print(f"Noise - Mean variance difference: {np.mean(np.array(bio_noise['variance']) - np.array(baseline_noise['variance'])):.6f}")
    print(f"Contrast - Mean variance difference: {np.mean(np.array(bio_contrast['variance']) - np.array(baseline_contrast['variance'])):.6f}")
    print(f"Blur - Mean variance difference: {np.mean(np.array(bio_blur['variance']) - np.array(baseline_blur['variance'])):.6f}")
    print(f"JPEG - Mean variance difference: {np.mean(np.array(bio_jpeg['variance']) - np.array(baseline_jpeg['variance'])):.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robustness testing for Bio-XYW-Net')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--bio_checkpoint', type=str, default=None, help='Bio model checkpoint')
    parser.add_argument('--baseline_checkpoint', type=str, default=None, help='Baseline checkpoint')
    parser.add_argument('--output_dir', type=str, default='./robustness_results', help='Output directory')
    
    args = parser.parse_args()
    
    run_robustness_tests(
        image_path=args.image,
        bio_checkpoint=args.bio_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        output_dir=args.output_dir
    )
