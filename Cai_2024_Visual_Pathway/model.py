"""
Cai et al. (2024) - Image Contour Detection Based on Visual Pathway Information Transfer

Key Features:
1. Double receptive fields (weighted combination)
2. Double stream information fusion (Magno/Parvo pathways)
3. Adaptive response adjustment
4. Full LGN → V1 → V2 → V4 hierarchy

Paper: Neural Processing Letters, 2024
DOI: 10.1007/s11063-024-11486-3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleReceptiveField(nn.Module):
    """Weighted combination of double receptive fields (LGN-like)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Center receptive field (small, detailed)
        self.rf_center = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # Surround receptive field (large, contextual)
        self.rf_surround = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive weighting
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        center = self.rf_center(x)
        surround = self.rf_surround(x)
        
        # Adaptive weights
        weights = self.weight_gen(x)  # [B, 2, 1, 1]
        w_center = weights[:, 0:1, :, :]
        w_surround = weights[:, 1:2, :, :]
        
        # Weighted combination
        out = w_center * center + w_surround * surround
        return out


class DoubleStreamModule(nn.Module):
    """Double stream information fusion (Magno/Parvo pathways)"""
    def __init__(self, channels):
        super().__init__()
        # Magno stream: fast, motion-sensitive, low-pass
        self.magno_stream = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Parvo stream: detailed, color-sensitive, high-pass
        self.parvo_stream = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Stream fusion with attention
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        magno = self.magno_stream(x)
        parvo = self.parvo_stream(x)
        fused = torch.cat([magno, parvo], dim=1)
        return self.fusion(fused)


class AdaptiveResponseModule(nn.Module):
    """Adaptive adjustment of response based on image statistics"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Adaptive modulation network
        self.modulation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        feat = F.relu(self.bn1(self.conv1(x)))
        # Adaptive scaling based on global context
        scale = self.modulation(x)
        return feat * scale


class V1Block(nn.Module):
    """V1: Primary visual cortex with orientation selectivity"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.orientations = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch // 4, 3, padding=1) for _ in range(4)
        ])
        self.fusion = nn.Conv2d(out_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.adaptive = AdaptiveResponseModule(out_ch)
        
    def forward(self, x):
        # Multi-orientation responses
        responses = [orient(x) for orient in self.orientations]
        combined = torch.cat(responses, dim=1)
        out = F.relu(self.bn(self.fusion(combined)))
        return self.adaptive(out)


class V2Block(nn.Module):
    """V2: Secondary visual cortex with curvature detection"""
    def __init__(self, channels):
        super().__init__()
        self.double_stream = DoubleStreamModule(channels)
        self.adaptive = AdaptiveResponseModule(channels)
        
    def forward(self, x):
        x = self.double_stream(x)
        return self.adaptive(x)


class V4Block(nn.Module):
    """V4: Higher-level shape and contour integration"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.adaptive = AdaptiveResponseModule(out_ch)
        
    def forward(self, x):
        x = self.conv(x)
        return self.adaptive(x)


class VisualPathwayNet(nn.Module):
    """
    Cai et al. 2024: Visual Pathway Information Transfer Network
    
    Architecture: Input → LGN → V1 → V2 → V4 → Output
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # LGN: Lateral Geniculate Nucleus with double receptive fields
        self.lgn = DoubleReceptiveField(in_channels, 32)
        
        # V1: Primary visual cortex (4 stages)
        self.v1_1 = V1Block(32, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.v1_2 = V1Block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # V2: Secondary visual cortex (2 stages)
        self.v2_1 = V2Block(128)
        self.pool3 = nn.MaxPool2d(2)
        self.v2_2 = V2Block(128)
        
        # V4: Higher-level processing
        self.v4 = V4Block(128, 256)
        
        # Decoder with multi-scale fusion
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final edge prediction
        self.edge_out = nn.Conv2d(32, 1, 1)
        
        # Side outputs for deep supervision
        self.side1 = nn.Conv2d(64, 1, 1)
        self.side2 = nn.Conv2d(128, 1, 1)
        self.side3 = nn.Conv2d(128, 1, 1)
        self.side4 = nn.Conv2d(256, 1, 1)
        
    def forward(self, x):
        # LGN processing
        lgn_out = self.lgn(x)
        
        # V1 pathway
        v1_1_out = self.v1_1(lgn_out)
        v1_1_pool = self.pool1(v1_1_out)
        
        v1_2_out = self.v1_2(v1_1_pool)
        v1_2_pool = self.pool2(v1_2_out)
        
        # V2 pathway
        v2_1_out = self.v2_1(v1_2_pool)
        v2_1_pool = self.pool3(v2_1_out)
        
        v2_2_out = self.v2_2(v2_1_pool)
        
        # V4 processing
        v4_out = self.v4(v2_2_out)
        
        # Decoder with skip connections
        d1 = self.up1(v4_out)
        d1 = torch.cat([d1, v2_1_out], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, v1_2_out], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, v1_1_out], dim=1)
        d3 = self.dec3(d3)
        
        # Main output
        edge_map = torch.sigmoid(self.edge_out(d3))
        
        # Side outputs for deep supervision
        if self.training:
            side1 = torch.sigmoid(self.side1(v1_1_out))
            side2 = torch.sigmoid(F.interpolate(self.side2(v1_2_out), scale_factor=2, mode='bilinear', align_corners=False))
            side3 = torch.sigmoid(F.interpolate(self.side3(v2_1_out), scale_factor=4, mode='bilinear', align_corners=False))
            side4 = torch.sigmoid(F.interpolate(self.side4(v4_out), scale_factor=8, mode='bilinear', align_corners=False))
            return [edge_map, side1, side2, side3, side4]
        else:
            return edge_map


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    model = VisualPathwayNet(in_channels=3)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 320, 320)
    model.eval()
    with torch.no_grad():
        out = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
    
    # Training mode
    model.train()
    outputs = model(x)
    print(f"\nTraining mode - Side outputs:")
    for i, side in enumerate(outputs):
        print(f"  Output {i}: {side.shape}")
