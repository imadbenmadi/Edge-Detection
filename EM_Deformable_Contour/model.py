"""
Bio-inspired Contour Extraction via EM-driven Deformable Model

This implementation combines:
1. Active Contour Models (Snakes) for deformable contour extraction
2. Expectation-Maximization (EM) for adaptive parameter learning
3. Bio-inspired energy functionals based on visual perception

Key Components:
- Internal Energy: Elasticity + Curvature (bio-inspired smoothness)
- External Energy: Edge-based attraction (V1-like gradient responses)
- EM Algorithm: Adaptive parameter optimization
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import interp1d
from typing import Tuple, Optional


class BioInspiredEnergyField:
    """Bio-inspired energy computation mimicking V1 edge responses"""
    
    def __init__(self, sigma_gabor=2.0, n_orientations=8):
        self.sigma = sigma_gabor
        self.n_orientations = n_orientations
        
    def compute_v1_responses(self, image: np.ndarray) -> np.ndarray:
        """
        Compute V1-like orientation-selective responses using Gabor filters
        
        Args:
            image: Grayscale image [H, W]
            
        Returns:
            energy_map: Combined orientation energy [H, W]
        """
        h, w = image.shape
        responses = []
        
        # Multi-orientation Gabor filters (simulating V1 simple cells)
        for theta in np.linspace(0, np.pi, self.n_orientations, endpoint=False):
            kernel = cv2.getGaborKernel(
                (21, 21), 
                self.sigma, 
                theta, 
                10.0,  # wavelength
                0.5,   # aspect ratio
                0, 
                ktype=cv2.CV_32F
            )
            response = np.abs(cv2.filter2D(image, cv2.CV_32F, kernel))
            responses.append(response)
        
        # Max pooling across orientations (V1 complex cell-like)
        energy_map = np.max(responses, axis=0)
        
        # Normalize
        energy_map = cv2.normalize(energy_map, None, 0, 1, cv2.NORM_MINMAX)
        
        return energy_map
    
    def compute_gradient_field(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient field for contour attraction
        
        Returns:
            fx, fy: Gradient components
        """
        # V1-like edge response
        v1_energy = self.compute_v1_responses(image)
        
        # Smooth to create force field
        smoothed = cv2.GaussianBlur(v1_energy, (0, 0), sigmaX=5.0)
        
        # Compute gradient (attractive force toward edges)
        gy, gx = np.gradient(-smoothed)  # Negative for attraction
        
        return gx, gy


class EMDeformableContour:
    """
    EM-driven Active Contour Model with bio-inspired energy terms
    """
    
    def __init__(
        self,
        alpha: float = 0.01,    # Elasticity (continuity)
        beta: float = 0.1,      # Curvature (smoothness)
        gamma: float = 0.1,     # External energy weight
        n_points: int = 100,
        n_orientations: int = 8
    ):
        """
        Args:
            alpha: Weight for internal elasticity energy
            beta: Weight for internal curvature energy  
            gamma: Weight for external edge energy
            n_points: Number of snake points
            n_orientations: Number of Gabor orientations
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_points = n_points
        
        self.energy_field = BioInspiredEnergyField(n_orientations=n_orientations)
        
        # EM parameters
        self.param_history = []
        
    def initialize_contour(
        self, 
        image_shape: Tuple[int, int], 
        center: Optional[Tuple[float, float]] = None,
        radius: Optional[float] = None
    ) -> np.ndarray:
        """
        Initialize circular contour
        
        Returns:
            contour: [n_points, 2] array of (x, y) coordinates
        """
        h, w = image_shape
        
        if center is None:
            center = (w // 2, h // 2)
        if radius is None:
            radius = min(h, w) // 4
            
        angles = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        
        return np.column_stack([x, y])
    
    def compute_internal_energy(self, contour: np.ndarray) -> float:
        """
        Internal energy: elasticity + curvature
        Bio-inspired: smooth contours like perceived object boundaries
        """
        n = len(contour)
        
        # Elasticity: penalize stretching (continuity)
        contour_roll = np.roll(contour, -1, axis=0)
        distances = np.linalg.norm(contour - contour_roll, axis=1)
        avg_distance = np.mean(distances)
        elasticity = np.sum((distances - avg_distance) ** 2)
        
        # Curvature: penalize bending (smoothness)
        contour_prev = np.roll(contour, 1, axis=0)
        contour_next = np.roll(contour, -1, axis=0)
        curvature = np.sum(np.linalg.norm(
            contour_prev - 2 * contour + contour_next, axis=1
        ) ** 2)
        
        return self.alpha * elasticity + self.beta * curvature
    
    def compute_external_energy(
        self, 
        contour: np.ndarray, 
        gradient_x: np.ndarray, 
        gradient_y: np.ndarray
    ) -> float:
        """
        External energy: edge attraction
        Bio-inspired: V1-like orientation responses
        """
        h, w = gradient_x.shape
        energy = 0.0
        
        for x, y in contour:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                # Gradient magnitude at contour point
                grad_mag = np.sqrt(gradient_x[yi, xi]**2 + gradient_y[yi, xi]**2)
                energy += grad_mag
                
        return -self.gamma * energy  # Negative to attract
    
    def evolve_contour(
        self,
        contour: np.ndarray,
        gradient_x: np.ndarray,
        gradient_y: np.ndarray,
        dt: float = 0.1
    ) -> np.ndarray:
        """
        Evolve contour by one step using gradient descent
        """
        h, w = gradient_x.shape
        new_contour = contour.copy()
        
        for i in range(len(contour)):
            x, y = contour[i]
            
            # Internal forces (smoothing)
            prev_point = contour[(i - 1) % len(contour)]
            next_point = contour[(i + 1) % len(contour)]
            
            # Elasticity force
            elastic_force = (prev_point + next_point - 2 * contour[i]) * self.alpha
            
            # Curvature force
            pp = contour[(i - 2) % len(contour)]
            nn = contour[(i + 2) % len(contour)]
            curvature_force = (pp + nn - 2 * contour[i]) * self.beta
            
            # External force (gradient field)
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                external_force = np.array([
                    gradient_x[yi, xi],
                    gradient_y[yi, xi]
                ]) * self.gamma
            else:
                external_force = np.array([0.0, 0.0])
            
            # Update position
            total_force = elastic_force + curvature_force + external_force
            new_contour[i] = contour[i] + dt * total_force
            
        return new_contour
    
    def em_step(
        self, 
        contour: np.ndarray, 
        image: np.ndarray,
        gradient_x: np.ndarray,
        gradient_y: np.ndarray
    ):
        """
        EM step: adaptively update parameters based on contour-image fit
        
        E-step: Estimate current model fit
        M-step: Update parameters to maximize fit
        """
        # E-step: Compute energy with current parameters
        internal_energy = self.compute_internal_energy(contour)
        external_energy = self.compute_external_energy(contour, gradient_x, gradient_y)
        total_energy = internal_energy + external_energy
        
        # M-step: Update parameters based on energy decomposition
        # Adaptive weighting: increase edge weight if internal energy dominates
        if internal_energy > abs(external_energy):
            self.gamma = min(1.0, self.gamma * 1.05)
        else:
            self.gamma = max(0.01, self.gamma * 0.95)
            
        self.param_history.append({
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'internal_energy': internal_energy,
            'external_energy': external_energy,
            'total_energy': total_energy
        })
    
    def fit(
        self,
        image: np.ndarray,
        initial_contour: Optional[np.ndarray] = None,
        n_iterations: int = 100,
        em_interval: int = 10,
        convergence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, dict]:
        """
        Fit deformable contour to image
        
        Args:
            image: Grayscale image [H, W]
            initial_contour: Initial contour points [n_points, 2]
            n_iterations: Maximum iterations
            em_interval: Apply EM step every N iterations
            convergence_threshold: Stop if movement < threshold
            
        Returns:
            final_contour: Final contour [n_points, 2]
            info: Dictionary with convergence info
        """
        # Initialize
        if initial_contour is None:
            contour = self.initialize_contour(image.shape)
        else:
            contour = initial_contour.copy()
        
        # Precompute gradient field
        gradient_x, gradient_y = self.energy_field.compute_gradient_field(image)
        
        # Evolution
        for iteration in range(n_iterations):
            prev_contour = contour.copy()
            
            # Evolve contour
            contour = self.evolve_contour(contour, gradient_x, gradient_y)
            
            # EM parameter adaptation
            if iteration % em_interval == 0:
                self.em_step(contour, image, gradient_x, gradient_y)
            
            # Check convergence
            movement = np.mean(np.linalg.norm(contour - prev_contour, axis=1))
            if movement < convergence_threshold:
                print(f"Converged at iteration {iteration}, movement={movement:.4f}")
                break
        
        info = {
            'iterations': iteration + 1,
            'final_movement': movement,
            'param_history': self.param_history
        }
        
        return contour, info
    
    def extract_contour_mask(
        self, 
        contour: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert contour to binary mask
        
        Returns:
            mask: Binary mask [H, W]
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        contour_int = contour.astype(np.int32)
        cv2.fillPoly(mask, [contour_int], 1)
        return mask


def demo():
    """Demo: Extract contour from synthetic image"""
    # Create synthetic test image
    img = np.zeros((256, 256), dtype=np.float32)
    cv2.circle(img, (128, 128), 60, 1.0, -1)
    img = cv2.GaussianBlur(img, (11, 11), 2.0)
    img = (img * 255).astype(np.uint8)
    
    # Initialize model
    model = EMDeformableContour(
        alpha=0.01,
        beta=0.1,
        gamma=0.5,
        n_points=50
    )
    
    # Fit contour
    contour, info = model.fit(img, n_iterations=100, em_interval=10)
    
    print(f"\nFitting complete:")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Final movement: {info['final_movement']:.4f}")
    print(f"  Final gamma: {model.gamma:.4f}")
    
    return contour, img, model


if __name__ == '__main__':
    contour, img, model = demo()
    print("\n✅ EM-driven deformable contour demo complete")
