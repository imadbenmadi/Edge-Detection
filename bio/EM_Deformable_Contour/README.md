# Bio-inspired Contour Extraction via EM-driven Deformable Model

Implementation of **active contour models (snakes)** with **Expectation-Maximization** (EM) for adaptive parameter learning and **bio-inspired energy functionals**.

## 📄 Overview

This method combines classical deformable models with bio-inspired energy terms and adaptive parameter optimization:

1. **Active Contours (Snakes)**: Parametric curves that evolve to fit object boundaries
2. **EM Algorithm**: Adaptively optimizes energy weights during evolution
3. **Bio-inspired Energy**: V1-like Gabor filter responses for edge detection
4. **Internal Energy**: Elasticity and curvature for smooth, natural contours

## 🏗️ Model Architecture

```
Input Image (Grayscale)
    ↓
Bio-inspired Energy Field
    ├─ V1-like Gabor Responses (8 orientations)
    ├─ Max Pooling (complex cell-like)
    └─ Gradient Field (attractive forces)
    ↓
Active Contour Evolution
    ├─ Internal Energy
    │   ├─ Elasticity (α): Continuity constraint
    │   └─ Curvature (β): Smoothness constraint
    ├─ External Energy (γ): Edge attraction
    └─ Gradient Descent Update
    ↓
EM Parameter Adaptation
    ├─ E-step: Estimate model fit
    └─ M-step: Update γ (edge weight)
    ↓
Final Contour
```

## 🔬 Energy Functional

The contour evolution minimizes:

```
E_total = E_internal + E_external
```

### Internal Energy (Regularization)

**Elasticity** (continuity):

```
E_elastic = α Σ ||v_i - v_{i-1}||²
```

**Curvature** (smoothness):

```
E_curvature = β Σ ||v_{i-1} - 2v_i + v_{i+1}||²
```

### External Energy (Data Term)

**Edge attraction**:

```
E_external = -γ Σ ||∇I(v_i)||
```

Where:

- `v_i`: Contour point i
- `∇I`: Image gradient (from V1-like Gabor responses)
- `α, β, γ`: Energy weights

## 🧠 Bio-inspired Components

### V1-like Energy Field

- **Gabor Filters**: 8 orientations mimicking V1 simple cells
- **Max Pooling**: Complex cell-like orientation invariance
- **Gradient Field**: Attractive forces toward edges

### Smooth Contours

- **Elasticity**: Mimics cortical grouping (continuity)
- **Curvature**: Mimics smooth object boundaries
- **EM Adaptation**: Adjusts to image statistics

## 🚀 Usage

### Running Evaluation

```bash
jupyter notebook evaluate.ipynb
```

Or use the model directly:

```python
from model import EMDeformableContour

# Initialize model
model = EMDeformableContour(
    alpha=0.01,     # Elasticity weight
    beta=0.1,       # Curvature weight
    gamma=0.3,      # Edge weight
    n_points=100,   # Number of contour points
    n_orientations=8  # Gabor orientations
)

# Initialize contour
initial_contour = model.initialize_contour(
    image.shape,
    center=(w//2, h//2),
    radius=50
)

# Fit contour with EM adaptation
final_contour, info = model.fit(
    image,
    initial_contour=initial_contour,
    n_iterations=100,
    em_interval=10
)
```

### Parameters

| Parameter        | Description             | Default | Range        |
| ---------------- | ----------------------- | ------- | ------------ |
| `alpha`          | Elasticity (continuity) | 0.01    | [0.001, 0.1] |
| `beta`           | Curvature (smoothness)  | 0.1     | [0.01, 1.0]  |
| `gamma`          | Edge weight (adaptive)  | 0.3     | [0.01, 1.0]  |
| `n_points`       | Contour points          | 100     | [30, 200]    |
| `n_orientations` | Gabor orientations      | 8       | [4, 12]      |

## 📊 EM Algorithm

The EM algorithm adaptively optimizes the edge weight γ:

**E-step** (Expectation):

- Compute E_internal and E_external with current parameters

**M-step** (Maximization):

- If E_internal > |E_external|: Increase γ (more edge attraction)
- Otherwise: Decrease γ (more regularization)

This allows the model to adapt to different image characteristics automatically.

## 📁 Directory Structure

```
EM_Deformable_Contour/
├── model.py              # Core implementation
├── evaluate.ipynb        # Evaluation notebook
├── README.md             # This file
└── outputs/              # Results
    ├── em_deformable_results.json
    ├── visualization.png
    └── em_adaptation.png
```

## 🎯 Performance

Evaluated on **HED_Small** test set (20 images):

| Metric  | Score                          |
| ------- | ------------------------------ |
| **ODS** | To be determined after running |
| **OIS** | To be determined after running |
| **AP**  | To be determined after running |

_Note: Run `evaluate.ipynb` to generate results_

## 🔧 Requirements

```bash
pip install opencv-python numpy scipy matplotlib tqdm scikit-learn
```

## 💡 Key Advantages

1. **Interpretable**: Clear energy terms with physical meaning
2. **Adaptive**: EM algorithm adjusts to image statistics
3. **Bio-inspired**: Mimics V1 edge detection and contour perception
4. **No Training**: Classical method, no training data required
5. **Flexible**: Works with various initialization strategies

## 🔗 Related Methods

### Active Contour Variants

- **GVF Snake**: Gradient Vector Flow for better capture range
- **Level Sets**: Implicit representation for topology changes
- **Chan-Vese**: Region-based active contours

### Bio-inspired Edge Detection

- **Cai et al. 2024**: Visual pathway network (see `../Cai_2024_Visual_Pathway/`)
- **Traditional bio models**: See `../bio/` for 12 classical studies

## 📚 References

**Active Contours**:

- Kass et al. (1988): "Snakes: Active contour models"
- Xu & Prince (1998): "Snakes, shapes, and gradient vector flow"

**EM Algorithm**:

- Dempster et al. (1977): "Maximum likelihood from incomplete data via the EM algorithm"

**Bio-inspired Vision**:

- Hubel & Wiesel (1962): "Receptive fields in cat visual cortex"
- Marr & Hildreth (1980): "Theory of edge detection"

## 🧪 Testing

Run the built-in demo:

```python
from model import demo

contour, img, model = demo()
# Creates synthetic circular object and fits contour
```

## ⚙️ Implementation Details

### Contour Representation

- **Parametric**: (x(s), y(s)) where s ∈ [0, 1]
- **Discrete**: N sample points uniformly spaced
- **Closed**: v_N = v_0 (cyclic boundary)

### Numerical Evolution

- **Gradient Descent**: v^{t+1} = v^t + Δt · F(v^t)
- **Time Step**: Δt = 0.1 (stable for typical images)
- **Convergence**: Stop when ||Δv|| < ε

### Multi-contour Detection

- Initialize multiple contours using Canny edges
- Fit each contour independently
- Merge results into final edge map

---

**Implementation**: Classical computer vision + bio-inspired energy + adaptive EM  
**Complexity**: O(N · K · M) where N = points, K = iterations, M = image pixels
