"""
Tang et al. - nCRF for Contour Detection
Learning Nonclassical Receptive Field Modulation for Contour Detection
IEEE Transactions on Image Processing, 2019

Bio-Inspiration: Nonclassical receptive field (nCRF) modulation mechanisms from V1
Key Features:
- Center-surround normalization
- Contextual modulation
- Multi-scale feature integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class nCRFContourModule(nn.Module):
    """
    Nonclassical Receptive Field module for contour detection.
    
    Implements center-surround modulation inspired by contextual 
    modulation in primary visual cortex (V1).
    """
    def __init__(self, channels):
        super().__init__()
        # Center receptive field - smaller kernel
        self.center = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Surround receptive field - larger kernel with depthwise convolution
        self.surround = nn.Conv2d(
            channels, channels, 
            kernel_size=7, padding=3, 
            groups=channels  # Depthwise convolution
        )
        
        # Modulation layer to combine center and surround
        self.modulate = nn.Conv2d(channels * 2, channels, kernel_size=1)
    
    def forward(self, x):
        # Extract center and surround features
        center = F.relu(self.center(x))
        surround = F.relu(self.surround(x))
        
        # Normalize features (key bio-inspired mechanism)
        norm_center = center / (center.norm(dim=1, keepdim=True) + 1e-8)
        norm_surround = surround / (surround.norm(dim=1, keepdim=True) + 1e-8)
        
        # Contextual modulation by combining normalized features
        modulated = self.modulate(torch.cat([norm_center, norm_surround], dim=1))
        
        return F.relu(modulated)


class TangNet(nn.Module):
    """
    Tang et al. nCRF Network for Contour Detection.
    
    Architecture:
    - Three-stage encoder with nCRF modules
    - Multi-scale feature extraction
    - Single contour map output
    """
    def __init__(self):
        super().__init__()
        
        # Stage 1: Initial feature extraction (64 channels)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.ncrf1 = nCRFContourModule(64)
        
        # Stage 2: Mid-level features (128 channels)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.ncrf2 = nCRFContourModule(128)
        
        # Stage 3: High-level features (256 channels)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.ncrf3 = nCRFContourModule(256)
        
        # Final contour prediction
        self.contour = nn.Conv2d(256, 1, kernel_size=1)
    
    def forward(self, x):
        h, w = x.shape[2:]
        
        # Stage 1: Low-level features with nCRF
        x = F.relu(self.conv1(x))
        x = self.ncrf1(x)
        
        # Stage 2: Mid-level features with pooling and nCRF
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = self.ncrf2(x)
        
        # Stage 3: High-level features with pooling and nCRF
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = self.ncrf3(x)
        
        # Generate contour map and upsample to original size
        contour = self.contour(x)
        contour = F.interpolate(contour, size=(h, w), mode='bilinear', align_corners=False)
        
        return torch.sigmoid(contour)


def create_model(pretrained=False):
    """
    Create Tang nCRF model.
    
    Args:
        pretrained: If True, loads pretrained weights (not implemented yet)
    
    Returns:
        TangNet model
    """
    model = TangNet()
    
    if pretrained:
        # TODO: Load pretrained weights when available
        print("Warning: Pretrained weights not yet available")
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Tang nCRF Model")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
