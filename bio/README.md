# Traditional Bio-Inspired Edge Detection Models

This folder contains implementations of **12 traditional bio-inspired edge detection models** based on neuroscience principles (Table 1, 2015-2024).

## 📚 Models Overview

All models implement classical computer vision techniques inspired by the human visual system, without deep learning.

| #   | Study         | Year | Bio Features                   | Key Contribution                      |
| --- | ------------- | ---- | ------------------------------ | ------------------------------------- |
| 01  | Kim et al.    | 2015 | LGN + V1                       | DoG filters + 8 Gabor orientations    |
| 02  | Patel et al.  | 2018 | LGN + V1                       | Multi-scale DoG processing            |
| 03  | Li et al.     | 2018 | LGN + V1                       | Adaptive intensity-dependent DoG      |
| 04  | Lee et al.    | 2019 | LGN + V1 + V2/V4 + Multi-level | First complete visual hierarchy       |
| 05  | Zhang et al.  | 2019 | LGN + V1                       | Optimized 12-orientation Gabor        |
| 06  | Park et al.   | 2020 | LGN + V1 + Multi-level         | Two-level hierarchical fusion         |
| 07  | Chen et al.   | 2020 | LGN + V1 + V2/V4 + Multi-level | Complete LGN/V1/V2/V4 hierarchy       |
| 08  | Nguyen et al. | 2020 | LGN + V1 + V2/V4 + Multi-level | 16 orientations + feedback modulation |
| 09  | Wang et al.   | 2020 | LGN + V1 + Multi-level         | Hierarchical pooling across scales    |
| 10  | Zhao et al.   | 2022 | LGN + V1 + V2/V4 + Multi-level | Adaptive hierarchy (2022 SOTA)        |
| 11  | Wu et al.     | 2022 | LGN + V1 + Multi-level         | Efficient 2-scale processing          |
| 12  | Smith et al.  | 2024 | LGN + V1 + V2/V4 + Multi-level | Latest SOTA with dense orientations   |

## 🧠 Bio-Inspired Components

### LGN (Lateral Geniculate Nucleus)

- **Mechanism**: Difference of Gaussians (DoG)
- **Function**: Center-surround receptive fields
- **Implementation**: `cv2.GaussianBlur` with dual scales

### V1 (Primary Visual Cortex)

- **Mechanism**: Gabor filters
- **Function**: Orientation-selective simple cells
- **Implementation**: `cv2.getGaborKernel` with multiple orientations

### V2 (Secondary Visual Cortex)

- **Mechanism**: Sobel operators
- **Function**: Curvature and complex contour detection
- **Implementation**: `cv2.Sobel` for derivatives

### V4 (Visual Area 4)

- **Mechanism**: Canny edge detection
- **Function**: High-level shape integration
- **Implementation**: `cv2.Canny` with morphological ops

### Multi-level Processing

- **Mechanism**: Hierarchical fusion
- **Function**: Coarse-to-fine integration
- **Implementation**: Weighted combination of scale levels

## 📊 Evaluation

**Dataset**: HED_Small (20 test images)  
**Metrics**:

- **ODS**: Optimal Dataset Scale (fixed threshold)
- **OIS**: Optimal Image Scale (per-image threshold)
- **AP**: Average Precision

Run [00_COMPARISON_all_bio_models.ipynb](00_COMPARISON_all_bio_models.ipynb) to aggregate all results.

## 🚀 Usage

Each notebook is self-contained:

```python
# Example: Run Kim et al. 2015
jupyter notebook 01_Kim_et_al_2015.ipynb
```

Results are saved to `outputs/<Study_Year>/`:

- `<study>_metrics.json`: ODS, OIS, AP scores

## 📈 Evolution Timeline

- **2015-2018**: Basic LGN+V1 models (DoG + Gabor)
- **2019**: First complete visual hierarchy (Lee et al.)
- **2020**: Multi-level processing becomes standard
- **2022**: Adaptive/efficient variants
- **2024**: Dense orientations + advanced fusion (Smith et al.)

## 🔬 Technical Details

**Dependencies**:

- OpenCV: Classical CV operations
- NumPy: Numerical processing
- scikit-learn: AP metric computation

**No PyTorch Required**: Pure classical computer vision approach

## 📄 Related Work

For **bio-inspired deep learning** models (XYW-Net, LVP-Net, etc.), see [bio DL/](../bio%20DL/) folder.

---

**Reference**: Table 1 - Traditional Bio-Inspired Edge Detection Studies (2015-2024)
