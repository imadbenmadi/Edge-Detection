Thanks—here’s the quick parity check and what to tweak.

**Loss**
- **Type**: Probability-based weighted cross-entropy; positives weighted by label, negatives via `(1−p)` log.
- **Dice (optional)**: `DICE_COEFF` defaults `0.0`. Set `0.1` if you want the paper-style dice blend; otherwise leave off for baseline parity.
- **ELC/RCF**: Both decoders output probabilities (`sigmoid`), so the same loss is valid for either path.

**Training**
- **Optimizer/Schedule**: `Adam` + `StepLR(10, gamma=0.5)`; matches the original style and is stable with RCF. For ELC, consider `LR=5e-5` for the first 5–10 epochs (warmup) if training is spiky.
- **Data**: ImageNet normalization is active; loaders use `pin_memory` and adaptive `num_workers`.

**Evaluation**
- **Metrics**: ODS/OIS/AP via a global threshold sweep and per-image best. This mirrors the original pipeline intent.
- **Thinning**: Prediction thinning via `cv2.ximgproc.thinning` (fallback to Canny) + 1-pixel GT tolerance for matching; this is common in BSDS/HED-style evals and consistent with the paper’s edge evaluation approach.
- **Smoothing**: A light `GaussianBlur` before sweeping thresholds improves stability; if you want stricter 1:1 parity, you can remove that blur—results typically change minimally.

**Exact Parity Tips**
- **Baseline**: Keep `DECODE_MODE='rcf'`, `DICE_COEFF=0.0`, and ensure ImageNet normalization; that matches the released code’s behavior.
- **Strict eval match**: Ensure `opencv-contrib-python` is installed for thinning; or disable thinning if the original comparison you target didn’t thin.
- **ELC path**: If you switch to `DECODE_MODE='elc'`, leave the loss/eval unchanged since outputs are probabilities.

If you want, I can flip `DECODE_MODE` to `elc`, run the sanity-check cell, and kick off a 20-epoch training to compare ODS/OIS/AP against the RCF baseline.