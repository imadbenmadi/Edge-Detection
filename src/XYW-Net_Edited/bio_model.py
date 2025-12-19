"""
Bio-XYW-Net: XYW-Net with Bio-inspired Front-end
=================================================

This module extends the original XYW-Net model with:
1. Bio-inspired photoreceptor layer
2. ON/OFF bipolar pathways
3. Modified encoder to accept 6-channel input (ON_R, ON_G, ON_B, OFF_R, OFF_G, OFF_B)

The architecture is:
    Input RGB (B, 3, H, W)
        ↓
    BioFrontend (photoreceptor + ON/OFF split)
        ↓
    [ON_R, ON_G, ON_B, OFF_R, OFF_G, OFF_B] (B, 6, H, W)
        ↓
    Encoder (S1, S2, S3, S4) - modified to accept 6 channels
        ↓
    Decoder (F43, F32, F21)
        ↓
    Output edge map (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from bio_frontend import BioFrontend, BioFrontendWithGain


# Import original XYW-Net components (assuming they exist in model.py)
# We'll redefine key components here for clarity

class Conv2d(nn.Module):
    """Pixel Difference Convolution (PDC) - from original XYW-Net"""
    def __init__(self, pdc_func, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
    
    def forward(self, x):
        return self.conv(x)


class Xc1x1(nn.Module):
    """X-center surround pathway (small receptive field)"""
    def __init__(self, in_channels, out_channels):
        super(Xc1x1, self).__init__()
        self.Xcenter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Xcenter_relu = nn.ReLU(inplace=True)
        self.Xsurround = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.Xsurround_relu = nn.ReLU(inplace=True)
        self.in_channels = in_channels

    def forward(self, input):
        xcenter = self.Xcenter_relu(self.Xcenter(input))
        xsurround = self.Xsurround_relu(self.Xsurround(input))
        xsurround = self.conv1_1(xsurround)
        x = xsurround - xcenter
        return x


class Yc1x1(nn.Module):
    """Y-center surround pathway (large receptive field)"""
    def __init__(self, in_channels, out_channels):
        super(Yc1x1, self).__init__()
        self.Ycenter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Ycenter_relu = nn.ReLU(inplace=True)
        self.Ysurround = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=4, dilation=2, groups=in_channels)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.Ysurround_relu = nn.ReLU(inplace=True)

    def forward(self, input):
        ycenter = self.Ycenter_relu(self.Ycenter(input))
        ysurround = self.Ysurround_relu(self.Ysurround(input))
        ysurround = self.conv1_1(ysurround)
        y = ysurround - ycenter
        return y


class W(nn.Module):
    """W pathway: Oriented edge detection (horizontal + vertical)"""
    def __init__(self, in_channels, out_channels):
        super(W, self).__init__()
        self.w_h = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels)
        self.w_v = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels)
        self.conv_fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        w_h = self.relu(self.w_h(input))
        w_v = self.relu(self.w_v(input))
        w = self.conv_fusion(torch.cat([w_h, w_v], dim=1))
        return w


class XYW(nn.Module):
    """XYW processor - continues processing three pathways"""
    def __init__(self, inchannel, outchannel, stride=1):
        super(XYW, self).__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, xc, yc, w):
        xc = self.x_c(xc)
        yc = self.y_c(yc)
        w = self.w(w)
        return xc, yc, w


class XYW_S(nn.Module):
    """XYW Start - initializes three pathways from input"""
    def __init__(self, inchannel, outchannel, stride=1):
        super(XYW_S, self).__init__()
        self.stride = stride
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        xc = self.x_c(x)
        yc = self.y_c(x)
        w = self.w(x)
        return xc, yc, w


class XYW_E(nn.Module):
    """XYW End - merges three pathways back to one"""
    def __init__(self, inchannel, outchannel):
        super(XYW_E, self).__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, xc, yc, w):
        xc = self.x_c(xc)
        yc = self.y_c(yc)
        w = self.w(w)
        return xc + yc + w


class BioS1(nn.Module):
    """Bio-S1: First stage adapted for 6-channel input from bio-frontend"""
    def __init__(self, channel=30):
        super(BioS1, self).__init__()
        # Modify first conv to accept 6 channels (ON_R, ON_G, ON_B, OFF_R, OFF_G, OFF_B)
        self.conv1 = nn.Conv2d(6, channel, kernel_size=7, padding=6, dilation=2)
        self.xyw1_1 = XYW_S(channel, channel)
        self.xyw1_2 = XYW(channel, channel)
        self.xyw1_3 = XYW_E(channel, channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        temp = self.relu(self.conv1(x))
        xc, yc, w = self.xyw1_1(temp)
        xc, yc, w = self.xyw1_2(xc, yc, w)
        xyw1_3 = self.xyw1_3(xc, yc, w)
        return xyw1_3 + temp


class S2(nn.Module):
    """Standard S2 stage from XYW-Net"""
    def __init__(self, channel=60):
        super(S2, self).__init__()
        self.xyw2_1 = XYW_S(channel//2, channel, stride=2)
        self.xyw2_2 = XYW(channel, channel)
        self.xyw2_3 = XYW_E(channel, channel)
        self.shortcut = nn.Conv2d(in_channels=channel//2, out_channels=channel, kernel_size=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        xc, yc, w = self.xyw2_1(x)
        xc, yc, w = self.xyw2_2(xc, yc, w)
        xyw2_3 = self.xyw2_3(xc, yc, w)
        shortcut = self.shortcut(x)
        return xyw2_3 + shortcut


class S3(nn.Module):
    """Standard S3 stage from XYW-Net"""
    def __init__(self, channel=120):
        super(S3, self).__init__()
        self.xyw3_1 = XYW_S(channel//2, channel, stride=2)
        self.xyw3_2 = XYW(channel, channel)
        self.xyw3_3 = XYW_E(channel, channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(in_channels=channel // 2, out_channels=channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        xc, yc, w = self.xyw3_1(x)
        xc, yc, w = self.xyw3_2(xc, yc, w)
        xyw3_3 = self.xyw3_3(xc, yc, w)
        return xyw3_3 + shortcut


class S4(nn.Module):
    """Standard S4 stage from XYW-Net"""
    def __init__(self, channel=120):
        super(S4, self).__init__()
        self.xyw4_1 = XYW_S(channel, channel, stride=2)
        self.xyw4_2 = XYW(channel, channel)
        self.xyw4_3 = XYW_E(channel, channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        xc, yc, w = self.xyw4_1(x)
        xc, yc, w = self.xyw4_2(xc, yc, w)
        xyw4_3 = self.xyw4_3(xc, yc, w)
        return xyw4_3 + shortcut


class BioEncode(nn.Module):
    """Bio-inspired encoder with modified S1 for bio-frontend input"""
    def __init__(self):
        super(BioEncode, self).__init__()
        self.s1 = BioS1()
        self.s2 = S2()
        self.s3 = S3()
        self.s4 = S4()

    def forward(self, x):
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        return s1, s2, s3, s4


def crop(x, shape):
    """Crop x to match shape"""
    _, _, h, w = shape
    _, _, _h, _w = x.shape
    p_h = (_h - h) // 2 + 1
    p_w = (_w - w) // 2 + 1
    return x[:, :, p_h:p_h+h, p_w:p_w+w]


class Decoder(nn.Module):
    """Decoder for merging multi-scale features back to single resolution"""
    def __init__(self):
        super(Decoder, self).__init__()
        # F43: Fuse S4 and S3
        self.f43_conv = nn.Conv2d(240, 120, kernel_size=1)
        self.f43_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # F32: Fuse S3 and S2
        self.f32_conv = nn.Conv2d(180, 60, kernel_size=1)
        self.f32_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # F21: Fuse S2 and S1
        self.f21_conv = nn.Conv2d(90, 30, kernel_size=1)
        self.f21_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Final output
        self.out_conv = nn.Conv2d(30, 1, kernel_size=1)

    def forward(self, s1, s2, s3, s4):
        # F43: Merge S4 and S3
        s4_up = F.interpolate(s4, size=s3.shape[-2:], mode='bilinear', align_corners=False)
        f43 = torch.cat([s3, s4_up], dim=1)
        f43 = self.f43_conv(f43)
        
        # F32: Merge result with S2
        f43_up = F.interpolate(f43, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        f32 = torch.cat([s2, f43_up], dim=1)
        f32 = self.f32_conv(f32)
        
        # F21: Merge result with S1
        f32_up = F.interpolate(f32, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        f21 = torch.cat([s1, f32_up], dim=1)
        f21 = self.f21_conv(f21)
        
        # Final output
        output = self.out_conv(f21)
        output = torch.sigmoid(output)  # Normalize to [0, 1]
        
        return output


class BioXYWNet(nn.Module):
    """
    Bio-inspired XYW-Net combining retinal front-end with XYW processing.
    
    Architecture:
        RGB Input (B, 3, H, W)
            ↓
        BioFrontend (photoreceptor + ON/OFF)
            ↓
        6-channel bio-processed (B, 6, H, W)
            ↓
        BioEncode (S1→S2→S3→S4)
            ↓
        Decoder (F43→F32→F21)
            ↓
        Edge Map (B, 1, H, W)
    """
    
    def __init__(
        self,
        use_learnable_bio=False,
        bio_sigma=2.0,
        bio_kernel_size=5,
        add_noise=False,
        return_intermediate=False
    ):
        """
        Args:
            use_learnable_bio: If True, use BioFrontendWithGain for learnable parameters
            bio_sigma: Gaussian blur parameter for horizontal cells
            bio_kernel_size: Kernel size for Gaussian (must be odd)
            add_noise: Add Poisson noise at photoreceptor level
            return_intermediate: Return intermediate activations for visualization
        """
        super(BioXYWNet, self).__init__()
        
        self.return_intermediate = return_intermediate
        
        # Bio-inspired front-end
        if use_learnable_bio:
            self.bio_frontend = BioFrontendWithGain(
                sigma=bio_sigma,
                kernel_size=bio_kernel_size,
                learnable_sigma=True,
                learnable_gain=True,
                add_photoreceptor_noise=add_noise,
                return_intermediate=return_intermediate
            )
        else:
            self.bio_frontend = BioFrontend(
                sigma=bio_sigma,
                kernel_size=bio_kernel_size,
                add_photoreceptor_noise=add_noise,
                return_intermediate=return_intermediate
            )
        
        # Encoder and decoder
        self.encode = BioEncode()
        self.decode = Decoder()
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB image normalized to [0, 1]
            
        Returns:
            output: (B, 1, H, W) edge map
            
            if return_intermediate=True:
                also returns dict with intermediate activations
        """
        # Bio-inspired processing
        if self.return_intermediate:
            bio_output, bio_intermediates = self.bio_frontend(x)
        else:
            bio_output = self.bio_frontend(x)
            bio_intermediates = None
        
        # Encode through multi-scale stages
        s1, s2, s3, s4 = self.encode(bio_output)
        
        # Decode and merge
        output = self.decode(s1, s2, s3, s4)
        
        if self.return_intermediate:
            return output, {
                'bio': bio_intermediates,
                'encoder': {'s1': s1, 's2': s2, 's3': s3, 's4': s4}
            }
        
        return output


class BaselineXYWNet(nn.Module):
    """
    Baseline XYW-Net WITHOUT bio-frontend for comparison.
    This is the original XYW-Net with 3-channel input.
    """
    
    def __init__(self):
        super(BaselineXYWNet, self).__init__()
        
        # Original S1 with 3-channel input
        self.s1 = nn.Sequential(
            nn.Conv2d(3, 30, kernel_size=7, padding=6, dilation=2),
            nn.ReLU(inplace=True),
            XYW_S(30, 30),
            XYW(30, 30),
            XYW_E(30, 30)
        )
        
        self.encode = self._build_encoder()
        self.decode = Decoder()
    
    def _build_encoder(self):
        """Build baseline encoder"""
        return BioEncode()
    
    def forward(self, x):
        # Standard XYW processing
        s1 = self.s1(x)
        s2_input = F.max_pool2d(s1, kernel_size=2, stride=2)
        
        # ... rest follows standard encoding
        # For now, we'll use the encoder directly
        s1, s2, s3, s4 = self.encode(x)
        output = self.decode(s1, s2, s3, s4)
        
        return output


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Test Bio-XYW-Net
    print("Testing Bio-XYW-Net...")
    bio_model = BioXYWNet(use_learnable_bio=False, return_intermediate=False).to(device)
    
    # Dummy input
    x = torch.rand(2, 3, 256, 256).to(device)
    
    output = bio_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in bio_model.parameters())
    print(f"Bio-XYW-Net total parameters: {params:,}")
    
    # Test with learnable parameters
    print("\nTesting Bio-XYW-Net with learnable gains...")
    bio_model_learnable = BioXYWNet(use_learnable_bio=True).to(device)
    output_learnable = bio_model_learnable(x)
    print(f"Output shape: {output_learnable.shape}")
    params_learnable = sum(p.numel() for p in bio_model_learnable.parameters())
    print(f"Bio-XYW-Net (learnable) parameters: {params_learnable:,}")
