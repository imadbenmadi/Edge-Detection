# Bio-Inspired Deep Learning Models for Edge Detection

This folder contains implementations of **12 bio-inspired deep learning models** for edge detection, based on Table 2 from the research.

## 📚 Models Overview

All models are implemented as Jupyter notebooks with:

- Architecture definitions based on biological principles
- Evaluation on HED_Small dataset
- ODS/OIS/AP metrics computation
- JSON results export for comparison

### Model List

| #   | Model           | Bio-Inspiration                 | Deep Learning Enhancement  | Improvement Area               |
| --- | --------------- | ------------------------------- | -------------------------- | ------------------------------ |
| 01  | **XYW-Net**     | X/Y/W retinal cells             | Parallel pathways          | Precision & efficiency         |
| 02  | **LVP-Net**     | Hierarchical visual pathway     | Feedback refinement        | Edge continuity                |
| 03  | **BFCN**        | nCRF contextual modulation      | Multi-pathway fusion       | Saliency & texture suppression |
| 04  | **MI-Net**      | Multi-scale V1 receptive fields | Multi-scale integration    | Robustness across scales       |
| 05  | **BLCDNet**     | Efficient neural pathways       | Depthwise separable convs  | Computational efficiency       |
| 06  | **DPED**        | Hierarchical visual pathway     | Swin Transformer           | Precision & efficiency         |
| 07  | **BESD**        | End-stopped/hypercomplex cells  | Center-surround antagonism | One-pixel-wide edges           |
| 08  | **LEDNet**      | Magno/Parvocellular pathways    | Dual pathway processing    | Localization & efficiency      |
| 09  | **Bio-IFN**     | Ventral stream pathways         | Interactive feedback       | Noise robustness               |
| 10  | **BFE-Net**     | Retina + LGN                    | Feature enhancement        | Edge saliency                  |
| 11  | **SNN-EdgeNet** | Spiking neurons                 | Temporal integration       | Low-light edge detection       |
| 12  | **Tang et al.** | nCRF modulation                 | Deep contour detection     | Contour precision              |

## 🚀 Quick Start

### Run Individual Model

```python
# Open any notebook (e.g., 01_XYW-Net.ipynb)
# Execute all cells to:
# 1. Load model architecture
# 2. Run inference on HED_Small test set
# 3. Compute ODS/OIS/AP metrics
# 4. Save results to outputs/<model_name>/
```

### Compare All Models

```python
# Open 00_COMPARISON_all_bio_dl_models.ipynb
# This notebook:
# - Loads all model results
# - Generates comparison charts
# - Ranks models by performance
# - Exports comprehensive analysis
```

## 📊 Evaluation Metrics

All models are evaluated using:

- **ODS** (Optimal Dataset Scale): F-measure at best threshold across dataset
- **OIS** (Optimal Image Scale): Average F-measure with per-image best thresholds
- **AP** (Average Precision): Area under precision-recall curve

Evaluation protocol:

- Dataset: HED_Small (20 test images)
- Ground truth dilation: 3×3 kernel (1-pixel tolerance)
- Threshold range: 0.05 to 0.95 (30 thresholds)

## 🧬 Biological Principles

### Retinal Cells

- **XYW-Net**: X (parvocellular), Y (magnocellular), W (koniocellular) pathways
- **BFE-Net**: Ganglion cells with center-surround receptive fields

### Visual Cortex

- **BESD**: V1 end-stopped/hypercomplex cells for corners
- **LVP-Net**: Hierarchical V1/V2/V4 processing
- **MI-Net**: Multi-scale V1 receptive fields

### Contextual Modulation

- **BFCN, Tang et al.**: Normalized Contour Receptive Field (nCRF)
- **Bio-IFN**: Interactive feedback from higher areas

### Dual Pathways

- **LEDNet**: Magnocellular (motion) vs Parvocellular (detail)
- **DPED**: Ventral stream with attention

### Neural Computation

- **SNN-EdgeNet**: Spiking neurons with temporal dynamics
- **BLCDNet**: Efficient processing mimicking biological constraints

## 📁 Directory Structure

```
bio DL/
├── 00_COMPARISON_all_bio_dl_models.ipynb  # Comprehensive comparison
├── 01_XYW-Net.ipynb                       # Parallel X/Y/W pathways
├── 02_LVP-Net.ipynb                       # Lateral Visual Pathway
├── 03_BFCN.ipynb                          # Bio-inspired FCN with nCRF
├── 04_MI-Net.ipynb                        # Multi-scale Integration
├── 05_BLCDNet.ipynb                       # Lightweight contour detection
├── 06_DPED.ipynb                          # Deep pathway with attention
├── 07_BESD.ipynb                          # End-stopped edge detection
├── 08_LEDNet.ipynb                        # Lightweight dual pathway
├── 09_Bio-IFN.ipynb                       # Interactive feedback
├── 10_BFE-Net.ipynb                       # Feature enhancement
├── 11_SNN-EdgeNet.ipynb                   # Spiking neural network
├── 12_Tang-nCRF.ipynb                     # nCRF for contour detection
├── outputs/                               # Model results
│   ├── XYW-Net/
│   │   └── xywnet_metrics.json
│   ├── LVP-Net/
│   │   └── lvpnet_metrics.json
│   ├── ...
│   ├── bio_dl_comparison_results.csv
│   ├── bio_dl_comparison_summary.json
│   ├── comparison_metrics.png
│   └── correlation_matrix.png
└── README.md                              # This file
```

## 🔬 Model Architectures

### Hierarchical Models

- **LVP-Net**: V1 → V2 → V4 with top-down feedback refinement
- **DPED**: Multi-stage pathway with window attention mechanisms

### Multi-Scale Models

- **MI-Net**: Parallel 3×3, 5×5, 7×7 convolutions
- **BFCN**: Multiple nCRF modules at different scales

### Lightweight Models

- **BLCDNet**: ~140K parameters with depthwise separable convolutions
- **LEDNet**: ~200K parameters with dual pathway design

### Specialized Models

- **BESD**: End-stopped cells (center - 0.3 × surround)
- **SNN-EdgeNet**: Leaky Integrate-and-Fire (LIF) neurons with τ=0.9

## 📈 Expected Performance

All models are evaluated on the same HED_Small test set (20 images). Results vary based on:

- **Random initialization**: No pretrained weights available for most models
- **Architecture design**: Quality depends on bio-inspired principle implementation
- **Dataset characteristics**: Performance may differ on other benchmarks

**Note**: These are reference implementations. For production use, models should be trained on larger datasets (BSDS500, BIPED, etc.).

## 🛠️ Requirements

```bash
pip install torch torchvision opencv-python numpy tqdm scikit-learn matplotlib seaborn pandas
```

## 📖 References

Models are based on bio-inspired principles from computational neuroscience and computer vision research. See Table 2 in the original research document for paper citations and detailed descriptions.

## 🎯 Usage Notes

1. **Training**: Current notebooks use random initialization. For best results, train on BSDS500 or BIPED datasets.
2. **Pretrained Weights**: XYW-Net uses existing workspace implementation. Others require training.
3. **Computational Resources**: Models run on CPU by default, but GPU acceleration is supported when available.
4. **Dataset**: Uses HED_Small (20 test images). Extend to full datasets for comprehensive evaluation.

## 📊 Comparison Outputs

The comparison notebook (`00_COMPARISON_all_bio_dl_models.ipynb`) generates:

- **CSV**: Ranked model performance table
- **JSON**: Complete metrics with statistics
- **Charts**: Bar plots for ODS/OIS/AP comparison
- **Correlation Matrix**: Metric relationships

---

**Created**: Bio-inspired deep learning model suite for edge detection research  
**Dataset**: HED_Small test set (20 images)  
**Framework**: PyTorch with OpenCV for evaluation
