"""
Bio-inspired Front-end for Edge Detection
==========================================

This module implements a biological retina-inspired front-end that processes
visual input through stages that mimic:

1. Photoreceptor layer: Log nonlinearity, light adaptation, optional noise
2. Horizontal cells: Lateral inhibition via Gaussian blur
3. ON/OFF bipolar pathways: Center-surround receptive fields

Mathematical Formulation:
------------------------

Stage 1: Photoreceptor Adaptation
    L_adapted = log(1 + I / (1 + sum_surround(I)))
    where I is input intensity (0-1 normalized)

Stage 2: Photoreceptor Nonlinearity (Weber's Law)
    L_out = L_adapted / (1 + L_adapted)  # Dividing normalization

Stage 3: Horizontal Cell Surround (Gaussian-weighted average)
    H(x,y) = G_sigma(x,y) * L_out
    G_sigma(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2 + y^2)/(2*sigma^2))

Stage 4: ON/OFF Bipolar Splitting
    ON_response = ReLU(L_out - H)        # On-center OFF-surround
    OFF_response = ReLU(H - L_out)       # Off-center ON-surround

Stage 5: Normalization
    ON_norm = ON / (1 + ON + OFF)
    OFF_norm = OFF / (1 + ON + OFF)

References:
-----------
- Masland, R. H. (2012). The Neuronal Organization of the Retina
- Gollisch, T., & Meister, M. (2010). Eye smarter than scientists believed
- Maheswaranathan, N., et al. (2019). Deep learning models of the retinal response
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhotoreceptorLayer(nn.Module):
    """
    Simulates photoreceptor adaptation using:
    - Logarithmic nonlinearity (Weber's law)
    - Light adaptation (divisive normalization)
    - Optional quantum noise
    
    Input:  (B, C, H, W) normalized to [0, 1]
    Output: (B, C, H, W) adapted intensity
    """
    
    def __init__(self, epsilon=1e-4, add_noise=False, noise_std=0.01):
        super(PhotoreceptorLayer, self).__init__()
        self.epsilon = epsilon
        self.add_noise = add_noise
        self.noise_std = noise_std
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) in range [0, 1]
            
        Returns:
            adapted: (B, C, H, W) photoreceptor output
            
        Mathematical:
            adapted = log(1 + x / (1 + x))
                    = log(1 + x) - log(2 + x)
        """
        # Ensure input is in valid range
        x = torch.clamp(x, 0, 1)
        
        # Apply logarithmic nonlinearity with normalization
        # This implements Weber's law: response is proportional to log(I/I_background)
        adapted = torch.log(1.0 + x + self.epsilon) - torch.log(2.0 + x + self.epsilon)
        
        # Normalize output to [0, 1]
        adapted = (adapted - adapted.min()) / (adapted.max() - adapted.min() + self.epsilon)
        
        # Add optional quantum noise (Poisson-like)
        if self.add_noise and self.training:
            noise = torch.randn_like(adapted) * self.noise_std
            adapted = adapted + noise
            adapted = torch.clamp(adapted, 0, 1)
        
        return adapted


class HorizontalCellLayer(nn.Module):
    """
    Simulates horizontal cells that provide lateral inhibition.
    
    Horizontal cells respond to the average luminance in their receptive field
    and feed back inhibition to photoreceptors.
    
    Implementation: Gaussian-weighted blur with learnable sigma
    
    Input:  (B, C, H, W) photoreceptor adapted output
    Output: (B, C, H, W) horizontal cell surround response
    """
    
    def __init__(self, sigma=2.0, kernel_size=5):
        super(HorizontalCellLayer, self).__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('kernel', kernel)
        
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create 2D Gaussian kernel"""
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) photoreceptor output
            
        Returns:
            surround: (B, C, H, W) horizontal cell response
        """
        B, C, H, W = x.shape
        
        # Apply Gaussian blur to each channel
        # Expand kernel to match number of channels
        kernel = self.kernel.expand(C, 1, -1, -1).to(x.device)
        
        surround = F.conv2d(
            x.reshape(B * C, 1, H, W),
            kernel,
            padding=self.kernel_size // 2
        ).reshape(B, C, H, W)
        
        return surround


class BipolarCellLayer(nn.Module):
    """
    ON/OFF Bipolar cell pathways implementing center-surround receptive fields.
    
    ON pathway (ON-center, OFF-surround):
        ON = ReLU(center - surround)
        
    OFF pathway (OFF-center, ON-surround):
        OFF = ReLU(surround - center)
    
    Mathematical formulation (using photoreceptor L and horizontal H):
        ON = max(0, L - H)          # ON-bipolar
        OFF = max(0, H - L)         # OFF-bipolar
    
    Then normalize to prevent saturation:
        ON_norm = ON / (1 + ON + OFF + eps)
        OFF_norm = OFF / (1 + ON + OFF + eps)
    """
    
    def __init__(self, epsilon=1e-4):
        super(BipolarCellLayer, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, photoreceptor, surround):
        """
        Args:
            photoreceptor: (B, C, H, W) center response
            surround: (B, C, H, W) horizontal cell (surround) response
            
        Returns:
            on_pathway: (B, C, H, W) ON-bipolar response
            off_pathway: (B, C, H, W) OFF-bipolar response
        """
        # ON pathway: center > surround (increases in luminance)
        on_response = torch.clamp(photoreceptor - surround, min=0.0)
        
        # OFF pathway: surround > center (decreases in luminance)
        off_response = torch.clamp(surround - photoreceptor, min=0.0)
        
        # Divisive normalization (Weber's law continuation)
        # This prevents saturation and implements automatic gain control
        total_response = on_response + off_response + self.epsilon
        
        on_normalized = on_response / total_response
        off_normalized = off_response / total_response
        
        return on_normalized, off_normalized


class BioFrontend(nn.Module):
    """
    Complete bio-inspired front-end combining all retinal stages.
    
    Pipeline:
        Input (RGB) 
        → Photoreceptor adaptation 
        → Horizontal cell surround 
        → ON/OFF bipolar split 
        → Concatenated output [ON, OFF] → (B, 6, H, W)
    
    Each RGB channel is processed independently through the retina,
    then ON and OFF pathways are concatenated for network input.
    """
    
    def __init__(
        self,
        sigma=2.0,
        kernel_size=5,
        add_photoreceptor_noise=False,
        noise_std=0.01,
        epsilon=1e-4,
        return_intermediate=False
    ):
        """
        Args:
            sigma: Gaussian blur sigma for horizontal cells
            kernel_size: Gaussian kernel size (should be odd)
            add_photoreceptor_noise: Whether to add Poisson noise
            noise_std: Standard deviation of noise
            epsilon: Small constant for numerical stability
            return_intermediate: If True, return intermediate activations
        """
        super(BioFrontend, self).__init__()
        
        self.return_intermediate = return_intermediate
        
        self.photoreceptor = PhotoreceptorLayer(
            epsilon=epsilon,
            add_noise=add_photoreceptor_noise,
            noise_std=noise_std
        )
        self.horizontal = HorizontalCellLayer(sigma=sigma, kernel_size=kernel_size)
        self.bipolar = BipolarCellLayer(epsilon=epsilon)
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB image normalized to [0, 1]
            
        Returns:
            if return_intermediate=False:
                output: (B, 6, H, W) concatenated [ON_R, ON_G, ON_B, OFF_R, OFF_G, OFF_B]
            
            if return_intermediate=True:
                output, intermediates: dict with 'photoreceptor', 'surround', 'on', 'off'
        """
        # Ensure input is in valid range
        x = torch.clamp(x, 0, 1)
        
        # Stage 1: Photoreceptor adaptation
        photoreceptor_response = self.photoreceptor(x)
        
        # Stage 2: Horizontal cell surround
        surround_response = self.horizontal(photoreceptor_response)
        
        # Stage 3: Bipolar pathways
        on_pathway, off_pathway = self.bipolar(photoreceptor_response, surround_response)
        
        # Concatenate ON and OFF channels: [ON_R, ON_G, ON_B, OFF_R, OFF_G, OFF_B]
        output = torch.cat([on_pathway, off_pathway], dim=1)
        
        if self.return_intermediate:
            intermediates = {
                'photoreceptor': photoreceptor_response,
                'surround': surround_response,
                'on': on_pathway,
                'off': off_pathway
            }
            return output, intermediates
        
        return output


class BioFrontendWithGain(nn.Module):
    """
    Extended bio-frontend with learnable gain control.
    
    This allows the network to learn optimal parameters for
    photoreceptor sensitivity and horizontal cell extent.
    """
    
    def __init__(
        self,
        sigma=2.0,
        kernel_size=5,
        learnable_sigma=True,
        learnable_gain=True,
        add_photoreceptor_noise=False,
        noise_std=0.01,
        epsilon=1e-4,
        return_intermediate=False
    ):
        super(BioFrontendWithGain, self).__init__()
        
        self.return_intermediate = return_intermediate
        
        # Learnable parameters
        if learnable_gain:
            self.photoreceptor_gain = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('photoreceptor_gain', torch.tensor(1.0))
        
        if learnable_sigma:
            self.sigma_param = nn.Parameter(torch.tensor(sigma))
        else:
            self.register_buffer('sigma_param', torch.tensor(sigma))
        
        self.photoreceptor = PhotoreceptorLayer(
            epsilon=epsilon,
            add_noise=add_photoreceptor_noise,
            noise_std=noise_std
        )
        self.horizontal = HorizontalCellLayer(sigma=sigma, kernel_size=kernel_size)
        self.bipolar = BipolarCellLayer(epsilon=epsilon)
        
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        
    def forward(self, x):
        """Forward with learnable parameters"""
        x = torch.clamp(x, 0, 1)
        
        # Apply photoreceptor with learnable gain
        photoreceptor_response = self.photoreceptor(x) * torch.clamp(
            self.photoreceptor_gain, min=0.1, max=10.0
        )
        photoreceptor_response = torch.clamp(photoreceptor_response, 0, 1)
        
        # Update horizontal cell sigma if learnable
        sigma = torch.clamp(self.sigma_param, min=0.5, max=10.0)
        self.horizontal.sigma = sigma.item()
        
        # Recreate Gaussian kernel with new sigma
        kernel = self.horizontal._create_gaussian_kernel(self.kernel_size, sigma.item())
        self.horizontal.kernel = kernel.to(x.device)
        
        # Forward through rest of network
        surround_response = self.horizontal(photoreceptor_response)
        on_pathway, off_pathway = self.bipolar(photoreceptor_response, surround_response)
        
        output = torch.cat([on_pathway, off_pathway], dim=1)
        
        if self.return_intermediate:
            intermediates = {
                'photoreceptor': photoreceptor_response,
                'surround': surround_response,
                'on': on_pathway,
                'off': off_pathway
            }
            return output, intermediates
        
        return output


if __name__ == "__main__":
    # Test the bio-frontend
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create dummy input (batch of 4 RGB images)
    x = torch.rand(4, 3, 256, 256).to(device)
    
    # Test basic frontend
    print("Testing BioFrontend...")
    frontend = BioFrontend(sigma=2.0, kernel_size=5, return_intermediate=True).to(device)
    output, intermediates = frontend(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Photoreceptor shape: {intermediates['photoreceptor'].shape}")
    print(f"Surround shape: {intermediates['surround'].shape}")
    print(f"ON shape: {intermediates['on'].shape}")
    print(f"OFF shape: {intermediates['off'].shape}")
    
    # Test with gain control
    print("\nTesting BioFrontendWithGain...")
    frontend_gain = BioFrontendWithGain(learnable_sigma=True, learnable_gain=True).to(device)
    output_gain = frontend_gain(x)
    print(f"Output with gain shape: {output_gain.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in frontend.parameters())
    print(f"\nBioFrontend parameters: {params}")
    
    params_gain = sum(p.numel() for p in frontend_gain.parameters())
    print(f"BioFrontendWithGain parameters: {params_gain}")
