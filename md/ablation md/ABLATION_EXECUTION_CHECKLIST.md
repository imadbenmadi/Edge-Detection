# 🎯 ABLATION STUDY - EXECUTION CHECKLIST

Print this or reference while running the study.

---

## ✅ PRE-STUDY SETUP (15-30 min)

-   [ ] Read `ABLATION_QUICK_REFERENCE.md` (5 min)
-   [ ] Read `ABLATION_INTEGRATION_GUIDE.md` (10 min)
-   [ ] Copy training functions:
    -   [ ] `train_epoch()` from xywnet_v2.2_gbt.ipynb
    -   [ ] `evaluate()` from xywnet_v2.2_gbt.ipynb
    -   [ ] `EdgeLoss` class
    -   [ ] `nms_edge()`, `dilate_gt()`, `compute_ods_ois_ap()`
-   [ ] Implement model factory:
    -   [ ] `build_ablated_xywnet()` function
    -   [ ] Support `disable_stages` flag
    -   [ ] Support `disable_pathways` flag
    -   [ ] Support `norm_type` flag
    -   [ ] Support other config keys
-   [ ] Update ablation notebook:
    -   [ ] Replace `train_variant_placeholder()` with `train_variant()`
    -   [ ] Update Cell 7 training loop
    -   [ ] Set `EPOCHS_PER_VARIANT` (recommend 5 for quick test)
    -   [ ] Verify all imports work

---

## 🧪 TEST RUN (5 min)

-   [ ] Run Cell 1 (imports & device)
    -   Verify: "Using device: cuda" (or cpu)
-   [ ] Run Cell 2 (dataset)
    -   Verify: "✓ Train: X samples", "✓ Val: Y samples", "✓ Test: Z samples"
-   [ ] Run Cell 3 (experiments registry)
    -   Verify: "Defined 36 experiments"
-   [ ] Run Cell 4 (tracker)
    -   Verify: "✓ Tracker initialized"
-   [ ] Run Cell 5 (loss & metrics)
    -   Verify: "✓ Loss & metrics functions loaded"
-   [ ] Run Cell 6 (training harness)
    -   Verify: No syntax errors
-   [ ] **Test with 1 variant:**
    -   Run: `train_variant('test', {'decoder': 'rcf'}, epochs=1)`
    -   Verify: Returns (ods, ois, ap, loss, epoch, time)
    -   Verify: No GPU memory errors
    -   Verify: Results reasonable (ODS > 0.2, loss < 1.0)

**If test passes → Green light for full study!**

---

## 🚀 RUN FULL ABLATION STUDY

### Phase 1: Tier 1 (5 min setup)

-   [ ] Set `EPOCHS_PER_VARIANT = 5`
-   [ ] Set `BATCH_SIZE = 4` (adjust if OOM)
-   [ ] Start fresh Jupyter kernel (clear memory)
-   [ ] Re-run cells 1-6 (initialization)
-   [ ] **Run Cell 7 (main training loop)**
    -   ⏱️ Time: ~2-4 hrs for 36 variants × 5 epochs
    -   Monitor: Progress bar should move steadily
    -   If hung: Check GPU usage, restart kernel, try fewer workers

### Phase 2: Monitoring (during study)

-   [ ] Check GPU memory:
    -   `nvidia-smi` in terminal
    -   If >90%: Reduce BATCH_SIZE or NUM_WORKERS
-   [ ] Monitor loss trends:
    -   First few variants should show decreasing loss per epoch
    -   If loss increases: LR too high, optimizer issue
-   [ ] Spot-check ODS values:
    -   Should be 0.3–0.55 range
    -   If 0.0 or 1.0: Evaluation bug
    -   If <0.2: Model not learning

### Phase 3: Results (after Cell 7)

-   [ ] Run Cell 8 (save CSV)
    -   Verify: "✓ Saved results CSV"
    -   Verify: CSV file in `ablation_results/`
-   [ ] Run Cell 9 (plot comparisons)
    -   Verify: "✓ Saved plot:" (3 plots for ODS/OIS/AP)
    -   Verify: Images look reasonable (bars sorted by metric)
-   [ ] Run Cell 10 (component analysis)
    -   Verify: Ranked results printed
    -   Verify: Best variant listed
    -   Verify: Component impact table
-   [ ] Run Cell 11 (export)
    -   Verify: JSON files exported
-   [ ] Run Cell 12 (recommendations)
    -   Verify: Summary printed with findings

---

## 📊 ANALYSIS & INTERPRETATION (30 min)

### Data Quality Checks

-   [ ] Open `ablation_results_*.csv` in Excel/Pandas
-   [ ] Verify all 36 rows present
-   [ ] Verify metrics in expected range:
    -   [ ] ODS: 0.2–0.6
    -   [ ] OIS: 0.2–0.6
    -   [ ] AP: 0.1–0.5
    -   [ ] train_loss: 0.1–1.0
    -   [ ] time_sec: 60–600
-   [ ] Sort by ODS, verify ranking makes sense
-   [ ] Check for outliers (ODS=0 or OIS=1.0)

### Component Analysis

-   [ ] Verify findings match hypothesis:
    -   [ ] s1 removal hurts most?
    -   [ ] ELC comparable/better than RCF?
    -   [ ] Single pathways significantly worse?
    -   [ ] Normalization variations show difference?
-   [ ] Identify top 3 variants:
    -   [ ] Record names and ODS scores
    -   [ ] Note config differences

### Ranking Interpretation

-   [ ] Top variant info:
    -   Name: ********\_\_\_********
    -   ODS: **\_** OIS: **\_** AP: **\_**
    -   Key differences from baseline: ********\_\_\_********
-   [ ] Biggest ablation hit:
    -   Component removed: ********\_\_\_********
    -   ODS drop: **\_** (should be 0.02–0.06)
-   [ ] Most redundant component:
    -   Component removed: ********\_\_\_********
    -   ODS drop: **\_** (should be <0.005)

---

## 🏆 VALIDATE WINNERS (1-2 hrs per variant)

After identifying top 3 variants:

### For each top variant:

-   [ ] Create new training config:
    -   Set `EPOCHS_PER_VARIANT = 20`
    -   Copy variant config exactly
-   [ ] Train dedicated model:
    -   Run full 20 epochs
    -   Save to `models/winner_variant_name/`
-   [ ] Evaluate on test set:
    -   Run evaluation
    -   Record final ODS/OIS/AP
    -   Compare to ablation result (should be higher)
-   [ ] Save final checkpoint:
    -   `models/winner_variant_name_final.pth`

---

## 📝 DOCUMENT FINDINGS

Create summary document:

```markdown
# Ablation Study Results

## Baseline

-   rcf_baseline: ODS=\_**\_ OIS=\_\_** AP=\_\_\_\_

## Winner

-   [name]: ODS=\_**\_ OIS=\_\_** AP=\_\_\_\_
-   Improvement: +\_\_\_\_ ODS
-   Config: [list key differences]

## Key Findings

1. Component X most important (ODS drop \_\_\_\_)
2. Component Y somewhat important (ODS drop \_\_\_\_)
3. Component Z negligible (ODS drop \_\_\_\_)
4. Best decoder: RCF vs ELC?
5. Best loss: Dice vs pos_weight?
6. Best norm: Instance vs Batch vs Group?

## Recommendations

-   Keep: [components that matter]
-   Remove: [redundant components]
-   Modify: [better hyperparameters]

## Files

-   Ablation results: ablation*results*\*.csv
-   Plots: ablation\_\*.png
-   Winner model: models/winner\_\*.pth
```

---

## 🐛 TROUBLESHOOTING

| Issue                      | Cause                 | Fix                                           |
| -------------------------- | --------------------- | --------------------------------------------- |
| **OOM (out of memory)**    | Batch size too large  | Reduce BATCH_SIZE to 2 or 1                   |
| **All ODS = 0.0**          | Model not learning    | Check loss function, learning rate            |
| **All ODS = same value**   | Evaluation bug        | Verify nms_edge() and compute_ods_ois_ap()    |
| **Train loop hangs**       | Stuck on data loading | Reduce NUM_WORKERS to 0                       |
| **CSV has <36 rows**       | Variants crashed      | Check error messages in Cell 7                |
| **Plots don't show**       | Matplotlib issue      | Reinstall: `pip install matplotlib --upgrade` |
| **Time estimates way off** | GPU busy/slow         | Close other processes, check nvidia-smi       |

---

## 📞 HELP RESOURCES

| Question               | See File                      | Section      |
| ---------------------- | ----------------------------- | ------------ |
| What experiments?      | ABLATION_STUDY_README.md      | Section 1-11 |
| How to integrate?      | ABLATION_INTEGRATION_GUIDE.md | Steps 1-4    |
| What's tested?         | ABLATION_QUICK_REFERENCE.md   | Taxonomy     |
| What's optimal config? | ablation_study_v1.ipynb       | Cell 12      |
| Model architecture?    | xywnet_v2.2_gbt.ipynb         | Cells 4-5    |

---

## 🎉 SUCCESS CRITERIA

Study is complete when:

-   ✅ All 36 variants trained
-   ✅ CSV with results exported
-   ✅ Plots generated (ODS/OIS/AP)
-   ✅ Top 3 variants identified
-   ✅ Component analysis complete
-   ✅ Winner models validated (20 epochs)
-   ✅ Findings documented

---

## 📈 NEXT PHASE: Paper/Report

After ablation study:

1. Write abstract with key findings
2. Create table: Variant | ODS | OIS | AP | Config
3. Create figure: "Component Contribution to ODS"
4. Publish findings:
    - Email to advisor/team
    - Add to thesis/report
    - Prepare presentation

---

## 🚀 Ready? GO!

```
Cell 1: Imports → ✓
Cell 2: Dataset → ✓
Cell 3: Experiments → ✓
Cell 4: Tracker → ✓
Cell 5: Loss → ✓
Cell 6: Training → ✓
Cell 7: RUN STUDY → ▶️ GO!
Cell 8-12: Analyze → ✓
Validate winners → ✓
Document → ✓
DONE! 🎉
```

---

Good luck! 🚀
