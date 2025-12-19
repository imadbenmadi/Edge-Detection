

**Paper vs Baseline**
- **What authors used**: The released baseline sticks to the RCF-style decoder with gated `adap_conv` and a final `sigmoid` probability map. That’s what we’ve matched in your current notebook for parity.
- **ELC block**: Mentioned as an edge localization variant in the paper, typically producing logits before `sigmoid`. It targets crisper, thin edges but is often omitted in the final training runs due to stability/complexity trade-offs.

**Recommendation**
- **First baseline, then ablate**: Keep training on the RCF decoder to establish consistent ODS/OIS/AP, then enable ELC as a controlled ablation to see if it helps your data.
- **Metrics and loss unchanged**: Continue to evaluate with thinning and 1-pixel GT tolerance; ensure ELC outputs pass through `sigmoid` so the current `EdgeLoss` (CE + optional dice) remains valid.

**Enable ELC (Option)**
- **Toggle**: Add a simple `DECODE_MODE = 'rcf'|'elc'` and route `XYWNet` to `decode_elc` when set to `'elc'`.
- **Outputs**: If `ELCBlock` returns logits, apply `torch.sigmoid` before computing loss/metrics.
- **Training tips**: Use a slightly lower `LR` (e.g., `5e-5`), keep `weight_decay=1e-4`, and consider a short warmup (5–10 epochs) to stabilize.

**Where in the Notebook**
- **Model definition**: The XYW-Net encoder/RCF decoder is in Cell 10 of xywnet_v2.2_gbt.ipynb. We can wire a decode-mode switch there.
- **Loss/metrics**: The CE + optional dice and thinning pipeline are in Cell 13; no changes are required if we `sigmoid` the ELC output.
- **Training**: The training loop now runs 20 epochs (Cell 16) and saves checkpoints; it will work with either decoder once toggled.
