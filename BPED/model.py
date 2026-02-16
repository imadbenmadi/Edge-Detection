"""
BPED - Bio-inspired Pyramid Edge Detection

A deep learning model combining:
1. Bio-inspired visual processing (V1-like orientation selectivity)
2. Pyramid multi-scale architecture
3. Progressive feature refinement

Architecture inspired by visual cortex hierarchical processing with 
multi-scale pyramid feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrientedConv(nn.Module):
    """V1-like orientation-selective convolution"""
    def __init__(self, in_ch, out_ch, n_orientations=4):
        super().__init__()
        self.n_orientations = n_orientations
        # Each orientation gets its own conv
        self.oriented_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch // n_orientations, 3, padding=1)
            for _ in range(n_orientations)
        ])
        self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        # Apply each oriented filter
        responses = [conv(x) for conv in self.oriented_convs]
        # Concatenate across channel dimension
        out = torch.cat(responses, dim=1)
        return F.relu(self.bn(out))


class PyramidBlock(nn.Module):
    """Multi-scale pyramid block"""
    def __init__(self, channels):
        super().__init__()
        # Different scales
        self.scale1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.scale2 = nn.Conv2d(channels, channels, 5, padding=2)
        self.scale3 = nn.Conv2d(channels, channels, 7, padding=3)
        
        # Fusion
        self.fusion = nn.Conv2d(channels * 3, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        s1 = F.relu(self.scale1(x))
        s2 = F.relu(self.scale2(x))
        s3 = F.relu(self.scale3(x))
        
        combined = torch.cat([s1, s2, s3], dim=1)
        out = self.fusion(combined)
        return F.relu(self.bn(out)) + x  # Residual


class EdgeEnhancement(nn.Module):
    """Bio-inspired edge enhancement (sharpening)"""
    def __init__(self, channels):
        super().__init__()
        self.center = nn.Conv2d(channels, channels, 3, padding=1)
        self.surround = nn.Conv2d(channels, channels, 5, padding=2)
        self.modulate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        center = self.center(x)
        surround = self.surround(x)
        # Center-surround antagonism
        enhanced = center - 0.3 * surround
        # Attention modulation
        attention = self.modulate(x)
        return enhanced * attention


class BPEDEncoder(nn.Module):
    """Bio-inspired Pyramid Encoder"""
    def __init__(self):
        super().__init__()
        # Stage 1: Low-level features with orientation selectivity
        self.stage1 = nn.Sequential(
            OrientedConv(3, 64, n_orientations=4),
            OrientedConv(64, 64, n_orientations=4),
            PyramidBlock(64)
        )
        
        # Stage 2: Mid-level features
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = nn.Sequential(
            OrientedConv(64, 128, n_orientations=4),
            PyramidBlock(128),
            EdgeEnhancement(128)
        )
        
        # Stage 3: High-level features
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PyramidBlock(256),
            EdgeEnhancement(256)
        )
        
        # Stage 4: Deep features
        self.pool3 = nn.MaxPool2d(2)
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PyramidBlock(512)
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        s1 = self.stage1(x)
        
        s2 = self.pool1(s1)
        s2 = self.stage2(s2)
        
        s3 = self.pool2(s2)
        s3 = self.stage3(s3)
        
        s4 = self.pool3(s3)
        s4 = self.stage4(s4)
        
        return s1, s2, s3, s4


class BPEDDecoder(nn.Module):
    """Progressive pyramid decoder with skip connections"""
    def __init__(self):
        super().__init__()
        # Decoder stages
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PyramidBlock(256)
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PyramidBlock(128)
        )
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Edge predictions at multiple scales (deep supervision)
        self.edge4 = nn.Conv2d(512, 1, 1)
        self.edge3 = nn.Conv2d(256, 1, 1)
        self.edge2 = nn.Conv2d(128, 1, 1)
        self.edge1 = nn.Conv2d(64, 1, 1)
        
        # Final fusion
        self.final = nn.Conv2d(4, 1, 1)
        
    def forward(self, s1, s2, s3, s4):
        h, w = s1.shape[2:]
        
        # Stage 4 edge prediction
        edge4 = torch.sigmoid(self.edge4(s4))
        edge4_up = F.interpolate(edge4, (h, w), mode='bilinear', align_corners=False)
        
        # Decode 4->3
        d3 = self.up1(s4)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec1(d3)
        edge3 = torch.sigmoid(self.edge3(d3))
        edge3_up = F.interpolate(edge3, (h, w), mode='bilinear', align_corners=False)
        
        # Decode 3->2
        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)
        edge2 = torch.sigmoid(self.edge2(d2))
        edge2_up = F.interpolate(edge2, (h, w), mode='bilinear', align_corners=False)
        
        # Decode 2->1
        d1 = self.up3(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec3(d1)
        edge1 = torch.sigmoid(self.edge1(d1))
        
        # Fuse all scales
        all_edges = torch.cat([edge1, edge2_up, edge3_up, edge4_up], dim=1)
        final_edge = torch.sigmoid(self.final(all_edges))
        
        if self.training:
            # Return all predictions for deep supervision
            return final_edge, edge1, edge2, edge3, edge4
        else:
            return final_edge


class BPED(nn.Module):
    """
    BPED: Bio-inspired Pyramid Edge Detection
    
    Architecture:
        Input → Encoder (4 stages with pyramid blocks) → 
        Decoder (progressive refinement) → Edge Map
        
    Features:
        - Orientation-selective convolutions (V1-like)
        - Multi-scale pyramid processing
        - Edge enhancement (center-surround)
        - Deep supervision at multiple scales
    """
    def __init__(self):
        super().__init__()
        self.encoder = BPEDEncoder()
        self.decoder = BPEDDecoder()
        
    def forward(self, x):
        # Encode
        s1, s2, s3, s4 = self.encoder(x)
        
        # Decode
        output = self.decoder(s1, s2, s3, s4)
        
        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    model = BPED()
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 320, 320)
    
    # Eval mode
    model.eval()
    with torch.no_grad():
        out = model(x)
        print(f"\nEval mode:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
    
    # Training mode
    model.train()
    outputs = model(x)
    print(f"\nTraining mode (deep supervision):")
    for i, o in enumerate(outputs):
        print(f"  Output {i}: {o.shape}")
