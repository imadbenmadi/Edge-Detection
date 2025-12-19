# XYW-Net Ablation Study - Complete Scope Document

## 📋 Overview

Created comprehensive ablation study notebook (`ablation_study_v1.ipynb`) that systematically tests **35+ model variants** by removing or modifying each component individually and in combination.

---

## ✅ What's Included

### 1. **Decoder Variants** (2 experiments)

-   `rcf_baseline` - Standard RCF decoder (baseline)
-   `elc_enabled` - Edge Localization Convolution decoder

### 2. **Encoder Stage Ablations** (4 experiments)

-   `no_s1` - Remove stage 1 (30 channels)
-   `no_s2` - Remove stage 2 (60 channels)
-   `no_s3` - Remove stage 3 (120 channels)
-   `no_s4` - Remove stage 4 (120 channels)
-   **Question answered**: Which encoder layers are most critical?

### 3. **XYW Pathway Ablations** (9 experiments)

-   `no_X` - Remove center-surround pathway
-   `no_Y` - Remove dilated (large RF) pathway
-   `no_W` - Remove directional pathway
-   `only_X` - X pathway alone
-   `only_Y` - Y pathway alone
-   `only_W` - W pathway alone
-   `XY_pair` - X + Y (no W)
-   `XW_pair` - X + W (no Y)
-   `YW_pair` - Y + W (no X)
-   **Question answered**: What does each pathway contribute? Are they complementary?

### 4. **Convolution Type** (2 experiments)

-   `pdc_cv` - Use standard conv instead of PDC (Pixel Difference Conv)
-   `elc_pdc_cv` - ELC with standard conv
-   **Question answered**: Is the PDC structural prior necessary?

### 5. **Normalization Strategies** (3 experiments)

-   `norm_batch` - BatchNorm instead of InstanceNorm
-   `norm_group` - GroupNorm instead of InstanceNorm
-   `norm_none` - No normalization in decoder
-   **Question answered**: Which normalization is best for edge detection?

### 6. **Gating & Shortcuts** (4 experiments)

-   `no_adap_gate` - Disable adaptive gating in refine blocks
-   `no_shortcuts` - Disable residual shortcuts
-   `shortcut_alpha_0.5` - Weaken shortcuts (scale 0.5x)
-   `shortcut_alpha_2.0` - Strengthen shortcuts (scale 2.0x)
-   **Question answered**: How important are gating and residual connections?

### 7. **Deconvolution Learnability** (2 experiments)

-   `learnable_deconv` - Make bilinear deconv weights trainable (RCF)
-   `elc_learnable_deconv` - Learnable deconv for ELC
-   **Question answered**: Can the upsampling layer learn better structure?

### 8. **Pooling Strategy** (1 experiment)

-   `stride_pooling` - Replace max pooling with stride-2 convolution
-   **Question answered**: Can stride conv preserve signal better than max pool?

### 9. **Loss Function Sweeps** (6 experiments)

-   `loss_dice_0.05` - Dice coefficient 0.05
-   `loss_dice_0.1` - Dice coefficient 0.1
-   `loss_dice_0.2` - Dice coefficient 0.2
-   `loss_pos_weight_2` - CE positive class weight 2x
-   `loss_pos_weight_4` - CE positive class weight 4x
-   `loss_dice_dice_pos2` - Combined: Dice 0.1 + pos weight 2x
-   **Question answered**: Does loss weighting help with edge emphasis?

### 10. **Evaluation Controls** (2 experiments)

-   `no_thinning` - Skip edge thinning/NMS
-   `tolerance_r2` - GT tolerance radius 2 (vs 1)
-   **Question answered**: How much do post-processing and evaluation metrics matter?

### 11. **Combined Interactions** (1 experiment)

-   `rcf_batch_learn_stride` - BatchNorm + learnable deconv + stride pooling
-   **Question answered**: Do these improvements stack?

---

## 📊 Output & Analysis

The notebook produces:

### **Artifacts:**

1. **CSV Results** - All metrics for every variant

    - Columns: experiment, decoder, stages, pathways, norm_type, ODS, OIS, AP, train_loss, time

2. **Comparison Plots**

    - Bar charts: Top 20 variants ranked by ODS/OIS/AP
    - Color-coded by performance

3. **Component Impact Analysis**

    - Decoder impact (RCF vs ELC)
    - Encoder stage importance
    - XYW pathway contributions
    - Architecture component effects
    - Loss tuning impact
    - Evaluation sensitivity

4. **Rankings & Summaries**

    - Top 10 best variants
    - Bottom 10 worst variants (biggest drops)
    - Component contribution report

5. **Exportable Data**
    - `ablation_results_*.csv` - All results
    - `ablation_*.png` - Plots
    - `ablation_detailed_*.json` - Full data
    - `experiments_*.json` - Experiment registry (reproducibility)

---

## 🔧 How to Use

### **Step 1: Setup**

```python
# In notebook Cell 1
EPOCHS_PER_VARIANT = 5  # Quick test; set to 10–20 for final
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
```

### **Step 2: Integrate Actual Training**

Replace the placeholder `train_variant_placeholder()` in Cell 6 with real training:

-   Copy `train_epoch()` from `xywnet_v2.2_gbt.ipynb`
-   Copy `evaluate()` for validation metrics
-   Wire model factory to support ablation configs

### **Step 3: Run Study**

```python
# Cell 7 runs all 35 variants
# Monitor progress bar
# Results saved automatically to ablation_results/
```

### **Step 4: Analyze**

-   Cell 8: Summary tables (top/bottom)
-   Cell 9: Visual comparison plots
-   Cell 10: Component impact analysis
-   Cell 11–12: Recommendations

---

## 📈 Key Metrics Tracked

Per variant:

-   **ODS** - Optimal Dataset Scale (best F1 globally)
-   **OIS** - Optimal Image Scale (avg best F1 per image)
-   **AP** - Average Precision
-   **train_loss** - Final training loss
-   **time_sec** - Training time per variant
-   **best_epoch** - Which epoch gave best ODS

---

## 💡 What You'll Learn

After running this ablation study, you'll answer:

1. **Is RCF or ELC better?** Which decoder achieves higher ODS/OIS/AP?
2. **Which encoder stages matter most?** s1 vs s2 vs s3 vs s4?
3. **What do X/Y/W do?** Are all three pathways necessary? Are pairs sufficient?
4. **Is PDC critical?** How much does pixel-difference convolution help?
5. **What normalization works best?** InstanceNorm vs Batch vs Group?
6. **Do gating & shortcuts help?** By how much?
7. **Can deconv learn?** Frozen vs learnable bilinear upsampling?
8. **Stride vs max pooling?** Better signal preservation?
9. **Best loss function?** Dice, positive weighting, or combination?
10. **What's the optimal configuration?** Which variant wins?

---

## 🚀 Next Steps

1. **Run quick ablation** (5 epochs per variant) → ~30–60 min on GPU

    - Identify top 5 variants

2. **Validate winners** (20 epochs) → ~1–2 hrs each

    - Train top 3–5 variants to full convergence
    - Compare on test set

3. **Publish findings**

    - Document which components are critical vs redundant
    - Propose simplified/optimized XYW-Net

4. **Optional: Extended ablations**
    - Test kernel sizes (e.g., 5×5 vs 7×7 in s1)
    - Test channel widths (30→50, 60→100, etc.)
    - Test data augmentation impact

---

## 📁 File Structure

```
ablation_study_v1.ipynb          (Main notebook - run this)
ablation_results/
  ├── ablation_results_*.csv     (All results)
  ├── ablation_top20_ODS_*.png   (Comparison plots)
  ├── ablation_top20_OIS_*.png
  ├── ablation_top20_AP_*.png
  ├── ablation_detailed_*.json   (Full data export)
  ├── experiments_*.json         (Experiment config)
  └── model_weights/
      ├── rcf_baseline/          (Per-variant checkpoints)
      ├── elc_enabled/
      ├── no_s1/
      └── ...
```

---

## ⚙️ Configuration Options

You can easily extend the ablation study by adding to `EXPERIMENTS` list:

```python
# Add custom experiment
{
    'name': 'my_experiment',
    'decoder': 'rcf',
    'disable_stages': ['s1'],
    'norm_type': 'batch',
    'dice_coeff': 0.1,
    'description': 'Custom combo: no s1 + batch norm + dice'
}
```

Supported flags:

-   `decoder` - 'rcf' or 'elc'
-   `disable_stages` - list of ['s1', 's2', 's3', 's4']
-   `disable_pathways` - list of ['X', 'Y', 'W']
-   `pdc_type` - '2sd' or 'cv'
-   `norm_type` - 'instance', 'batch', 'group'
-   `disable_adap_gate` - True/False
-   `disable_shortcuts` - True/False
-   `shortcut_alpha` - float (default 1.0)
-   `learnable_deconv` - True/False
-   `pool_type` - 'maxpool' or 'stride_conv'
-   `dice_coeff` - float (default 0.0)
-   `ce_pos_weight` - float (default 1.0)
-   `thinning` - True/False
-   `tolerance_radius` - int (default 1)

---

## ✨ Summary

**Complete ablation study framework with:**

-   ✅ 35+ pre-configured experiments
-   ✅ Automatic result tracking & comparison
-   ✅ Visualization tools
-   ✅ Component impact analysis
-   ✅ Reproducible configuration export
-   ✅ Easy extensibility for custom variants

**Run time estimate:**

-   Quick test (5 epochs): 30–60 min
-   Full study (20 epochs): 2–4 hrs on GPU
