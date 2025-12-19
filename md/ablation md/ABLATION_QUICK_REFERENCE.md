# Ablation Study: Quick Reference Guide

## 📊 Experiment Taxonomy

```
XYW-Net Ablation Study (36 experiments)
│
├── 🏗️ ARCHITECTURE (15 experiments)
│   ├── Decoder
│   │   ├── RCF (baseline)
│   │   └── ELC
│   ├── Encoder Stages
│   │   ├── No s1 (30ch)
│   │   ├── No s2 (60ch)
│   │   ├── No s3 (120ch)
│   │   └── No s4 (120ch)
│   ├── XYW Pathways
│   │   ├── Remove X (center-surround)
│   │   ├── Remove Y (dilated)
│   │   ├── Remove W (directional)
│   │   ├── Only X
│   │   ├── Only Y
│   │   ├── Only W
│   │   ├── X+Y (no W)
│   │   ├── X+W (no Y)
│   │   └── Y+W (no X)
│
├── 🔧 COMPONENTS (11 experiments)
│   ├── Convolution
│   │   ├── PDC (2sd)
│   │   └── Standard (cv)
│   ├── Normalization
│   │   ├── InstanceNorm
│   │   ├── BatchNorm
│   │   ├── GroupNorm
│   │   └── None
│   ├── Gating & Shortcuts
│   │   ├── Adaptive gating ON/OFF
│   │   ├── Shortcuts ON/OFF
│   │   ├── Shortcut α=0.5 (weak)
│   │   └── Shortcut α=2.0 (strong)
│   └── Deconv
│       ├── Frozen bilinear
│       └── Learnable bilinear
│
├── 💾 TRAINING (8 experiments)
│   ├── Loss Functions
│   │   ├── Dice 0.05
│   │   ├── Dice 0.1
│   │   ├── Dice 0.2
│   │   ├── CE pos_weight 2x
│   │   ├── CE pos_weight 4x
│   │   └── Combined (Dice 0.1 + pos_weight 2x)
│   └── Pooling
│       ├── Max pooling
│       └── Stride convolution
│
└── 📈 EVALUATION (2 experiments)
    └── Controls
        ├── No thinning
        └── Tolerance radius r=2
```

---

## 🔬 What Each Experiment Measures

### **Decoder Impact**

| Exp          | Question           | Expected  | Ablation Tests |
| ------------ | ------------------ | --------- | -------------- |
| RCF baseline | Standard RCF good? | Baseline  | Yes            |
| ELC enabled  | ELC better/worse?  | ±1-3% ODS | Yes            |

### **Encoder Stage Importance**

| Exp   | Question  | Impact     | Hint                    |
| ----- | --------- | ---------- | ----------------------- |
| no_s1 | Critical? | Very high  | s1 = low-level features |
| no_s2 | Critical? | Medium     | s2 = mid-level features |
| no_s3 | Critical? | Low-medium | s3 = high-level context |
| no_s4 | Critical? | Low        | s4 = most abstract      |

### **XYW Pathway Roles**

| Exp      | Question             | What Measures          | Hypothesis              |
| -------- | -------------------- | ---------------------- | ----------------------- |
| no_X     | X important?         | Center-surround impact | Early pixel-level edges |
| no_Y     | Y important?         | Dilated RF impact      | Large structure edges   |
| no_W     | W important?         | Directional impact     | Oriented edges (minor?) |
| only_X   | X alone sufficient?  | Low-level only         | Probably not            |
| only_Y   | Y alone sufficient?  | Dilated only           | Probably not            |
| only_W   | W alone sufficient?  | Directional only       | Definitely not          |
| X+Y pair | Two pathways enough? | No W                   | Test complementarity    |
| X+W pair | Two pathways enough? | No Y                   | Test complementarity    |
| Y+W pair | Two pathways enough? | No X                   | Test complementarity    |

### **Component Tuning**

| Component     | Variants                   | What We Learn                 |
| ------------- | -------------------------- | ----------------------------- |
| Convolution   | PDC vs cv                  | Structural bias effectiveness |
| Normalization | Instance/Batch/Group/None  | Which stabilizes training?    |
| Gating        | ON vs OFF                  | Adaptive refinement value     |
| Shortcuts     | ON vs OFF, α∈{0.5,1.0,2.0} | Residual strength sweet spot  |
| Deconv        | Frozen vs Learnable        | Can upsampling improve?       |

### **Loss Tuning**

| Loss          | Variants            | What We Learn            |
| ------------- | ------------------- | ------------------------ |
| Dice          | 0.0, 0.05, 0.1, 0.2 | Best supervision balance |
| CE pos weight | 1.0, 2.0, 4.0       | Edge emphasis benefit    |
| Combined      | dice + pos_weight   | Synergy?                 |

---

## 📈 Expected Ranking Pattern

**Hypothesis (typical findings):**

```
Rank 1-5:  RCF baseline, ELC variants, learnable deconv → ~52-54% ODS
Rank 6-15: Loss tuning, normalization variants → ~50-52% ODS
Rank 16-25: Shortened pathways (X+Y, Y+W) → ~48-50% ODS
Rank 26-32: Single pathways (only X/Y/W) → ~40-45% ODS
Rank 33-36: Stage removal (no s1, s2, s3, s4) → ~35-48% ODS
```

**Biggest ODS drops expected:**

1. Remove s1 → -0.04 to -0.06
2. Remove s3 → -0.02 to -0.03
3. Remove Y → -0.01 to -0.02
4. Remove X → -0.01 to -0.015
5. Only W (remove X+Y) → -0.03 to -0.05

---

## 🎯 Critical Experiments (Run First)

If you need to prioritize, run these 10 first:

```
TIER 1 (Baselines):
1. rcf_baseline
2. elc_enabled

TIER 2 (Encoder):
3. no_s1
4. no_s2

TIER 3 (Pathways):
5. no_X
6. no_Y
7. no_W

TIER 4 (Components):
8. norm_batch
9. learnable_deconv

TIER 5 (Loss):
10. loss_dice_0.1
```

**Time:** 10 variants × 5 epochs ≈ 50–80 min

---

## 📊 Output Interpretation

### **CSV Columns Explained**

| Column     | Meaning                  | Range    | Good Value       |
| ---------- | ------------------------ | -------- | ---------------- |
| ODS        | Optimal Dataset Scale F1 | 0.0–1.0  | >0.50            |
| OIS        | Optimal Image Scale F1   | 0.0–1.0  | >0.50            |
| AP         | Average Precision        | 0.0–1.0  | >0.40            |
| train_loss | Final training loss      | 0.0–1.0+ | <0.30            |
| best_epoch | Which epoch best?        | 1–5      | Earlier = better |
| time_sec   | Training time            | 0–600    | Faster = better  |

### **Ranking Metrics (Primary)**

Use **ODS** for ablation ranking:

-   ODS accounts for both precision and recall globally
-   Most aligned with paper results
-   Single metric for comparison

### **Ranking Metrics (Secondary)**

-   **OIS**: Per-image best F1 (smoother, less sensitive to global thresholds)
-   **AP**: Area under precision-recall curve (robust to imbalance)

---

## 🚀 Analysis Workflow

```
Step 1: Run all 36 experiments
        ↓
Step 2: Save results → CSV + plots
        ↓
Step 3: Rank by ODS
        ↓
Step 4: Identify winners
        - Top 3-5 variants
        - Component patterns
        ↓
Step 5: Analyze contributions
        - Which components matter most?
        - Which are redundant?
        ↓
Step 6: Train top winners for 20 epochs
        - Validate findings
        - Get final scores
        ↓
Step 7: Draw conclusions
        - Best architecture?
        - Simplification opportunities?
```

---

## 💾 Configuration Template

**To add a custom experiment:**

```python
EXPERIMENTS.append({
    'name': 'my_experiment_name',
    'decoder': 'rcf',  # or 'elc'
    'disable_stages': ['s1'],  # subset of ['s1','s2','s3','s4']
    'disable_pathways': [],  # subset of ['X','Y','W']
    'pdc_type': '2sd',  # or 'cv'
    'norm_type': 'instance',  # or 'batch','group','none'
    'disable_adap_gate': False,
    'disable_shortcuts': False,
    'shortcut_alpha': 1.0,  # or 0.5, 2.0
    'learnable_deconv': False,  # or True
    'pool_type': 'maxpool',  # or 'stride_conv'
    'dice_coeff': 0.0,  # or 0.05, 0.1, 0.2
    'ce_pos_weight': 1.0,  # or 2.0, 4.0
    'thinning': True,
    'tolerance_radius': 1,  # or 2
    'description': 'Clear description of what this tests'
})
```

---

## 🎓 Key Questions to Answer

After ablation study finishes:

| Question                 | Answer From | How                                |
| ------------------------ | ----------- | ---------------------------------- |
| Is RCF or ELC better?    | Cells 8-9   | Compare top 2 variants             |
| Most important stage?    | Cell 10     | Look at no_s1 vs no_s4 drop        |
| Can we remove X pathway? | Cell 10     | See if no_X performance acceptable |
| Best normalization?      | Cell 10     | Compare norm\_\* variants          |
| Does gating help?        | Cell 10     | Compare no_adap_gate drop          |
| What's optimal config?   | Cells 8-9   | Top variant summary                |

---

## 📁 Files Reference

| File                            | Use For                        |
| ------------------------------- | ------------------------------ |
| `ablation_study_v1.ipynb`       | Run experiments                |
| `ABLATION_STUDY_README.md`      | Understand scope               |
| `ABLATION_INTEGRATION_GUIDE.md` | Wire training code             |
| `ABLATION_COMPLETE_SUMMARY.md`  | Big picture overview           |
| `ablation_results/`             | All outputs (CSV, plots, JSON) |

---

## ⏱️ Time Estimates

| Setup              | Variants | Epochs | GPU Time | Total     |
| ------------------ | -------- | ------ | -------- | --------- |
| Integration        | —        | —      | —        | 15-30 min |
| Quick test         | 1        | 2      | 5 min    | 5 min     |
| Tier 1 (Baselines) | 2        | 5      | 30 min   | 30 min    |
| Top 10             | 10       | 5      | 1.5 hrs  | 2 hrs     |
| Full study         | 36       | 5      | 5 hrs    | 6 hrs     |
| Winner validation  | 3        | 20     | 6 hrs    | 7 hrs     |

---

## ✨ Good Luck!

Run the ablation study → Get answers → Build better XYW-Net! 🚀
