# Integration Guide: Wiring Ablation Study to Main Training

## Overview

The ablation study notebook includes a **placeholder training function**. To run actual ablation studies, you need to integrate:

1. The actual model factory (supporting all ablation configs)
2. The actual training loop
3. The actual evaluation functions

---

## Step 1: Replace Placeholder Training Function

### Current (Placeholder) - Cell 6:

```python
def train_variant_placeholder(model_name, config, epochs=EPOCHS_PER_VARIANT):
    # Mock results...
    ods, ois, ap = 0.45, 0.50, 0.42
    return ods, ois, ap, train_loss, best_epoch, elapsed
```

### Replace With (Actual) - Integrate from xywnet_v2.2_gbt.ipynb:

```python
def train_variant(model_name, config, epochs=EPOCHS_PER_VARIANT):
    """
    Train a single ablation variant using actual training loop.
    """
    print(f"\n{'='*70}")
    print(f"Variant: {model_name}")
    print(f"Config: {config}")
    print(f"Epochs: {epochs}")
    print(f"{'='*70}\n")

    variant_start = time.time()

    # Step 1: Build ablated model
    model = build_ablated_xywnet(config).to(DEVICE)

    # Step 2: Setup training
    criterion = EdgeLoss(
        dice_coeff=config.get('dice_coeff', 0.0),
        ce_pos_weight=config.get('ce_pos_weight', 1.0)
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {'train_loss': [], 'val_ods': [], 'val_ois': [], 'val_ap': []}
    best_ods = 0
    best_epoch = 0

    # Step 3: Training loop (copy from xywnet_v2.2_gbt.ipynb)
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        history['train_loss'].append(train_loss)

        # Evaluate on validation
        val_ods, val_ois, val_ap = evaluate(
            model,
            val_loader,
            DEVICE,
            apply_thinning=config.get('thinning', True),
            tolerance_radius=config.get('tolerance_radius', 1)
        )
        history['val_ods'].append(val_ods)
        history['val_ois'].append(val_ois)
        history['val_ap'].append(val_ap)

        scheduler.step()

        print(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | ODS: {val_ods:.4f} | OIS: {val_ois:.4f} | AP: {val_ap:.4f}")

        # Track best
        if val_ods > best_ods:
            best_ods = val_ods
            best_epoch = epoch

            # Save checkpoint
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}_epoch{epoch}.pth")

    elapsed = time.time() - variant_start
    final_loss = history['train_loss'][-1] if history['train_loss'] else 0.0

    print(f"✓ Best ODS: {best_ods:.4f} at epoch {best_epoch}")
    print(f"✓ Final loss: {final_loss:.4f}")
    print(f"✓ Time: {elapsed/60:.1f} min")

    return best_ods, val_ois, val_ap, final_loss, best_epoch, elapsed
```

---

## Step 2: Implement Model Factory

Create a function that builds models with all ablation configs:

```python
def build_ablated_xywnet(config):
    """
    Build XYW-Net with ablation flags applied.

    Config keys:
    - decoder: 'rcf' or 'elc'
    - disable_stages: ['s1', 's2', 's3', 's4'] (any subset)
    - disable_pathways: ['X', 'Y', 'W'] (any subset)
    - pdc_type: '2sd' or 'cv'
    - norm_type: 'instance', 'batch', 'group'
    - disable_adap_gate: bool
    - disable_shortcuts: bool
    - shortcut_alpha: float
    - learnable_deconv: bool
    - pool_type: 'maxpool' or 'stride_conv'
    """

    # Start with base model
    model = XYWNetAblated(config)

    return model
```

You'll need to modify the `XYWNet` class (from xywnet_v2.2_gbt.ipynb) to support these flags:

### Encoder Modification:

```python
class encode_ablatable(nn.Module):
    def __init__(self, disable_stages=None, disable_pathways=None, pdc_type='2sd', pool_type='maxpool'):
        super().__init__()
        self.disable_stages = disable_stages or []
        self.disable_pathways = disable_pathways or []
        self.pdc_type = pdc_type
        self.pool_type = pool_type

        # Build stages
        self.s1_ = s1_ablatable(disable_pathways, pdc_type) if 's1' not in disable_stages else None
        self.s2_ = s2_ablatable(disable_pathways, pdc_type, pool_type) if 's2' not in disable_stages else None
        self.s3_ = s3_ablatable(disable_pathways, pdc_type, pool_type) if 's3' not in disable_stages else None
        self.s4_ = s4_ablatable(disable_pathways, pdc_type, pool_type) if 's4' not in disable_stages else None

    def forward(self, x):
        s1o = self.s1_(x) if self.s1_ else x
        s2o = self.s2_(s1o) if self.s2_ else s1o
        s3o = self.s3_(s2o) if self.s3_ else s2o
        s4o = self.s4_(s3o) if self.s4_ else s3o

        return s1o, s2o, s3o, s4o
```

### Decoder Modification:

```python
class decode_rcf_ablatable(nn.Module):
    def __init__(self, norm_type='instance', disable_adap_gate=False,
                 disable_shortcuts=False, shortcut_alpha=1.0,
                 learnable_deconv=False, disable_instance_norm=False):
        super().__init__()

        # Build refine blocks with options
        self.f43 = Refine_block2_1_ablatable(
            in_channel=(120, 120), out_channel=60, factor=2,
            norm_type=norm_type, disable_adap_gate=disable_adap_gate,
            require_grad=learnable_deconv
        )
        # ... similar for f32, f21

        self.f = nn.Conv2d(24, 1, kernel_size=1, padding=0)
        self.shortcut_alpha = shortcut_alpha if not disable_shortcuts else 0.0

    def forward(self, x):
        s3 = self.f43(x[2], x[3])
        s2 = self.f32(x[1], s3)
        s1 = self.f21(x[0], s2)
        out = self.f(s1)
        return out.sigmoid()
```

### Loss Function with Config:

```python
class EdgeLoss(nn.Module):
    def __init__(self, dice_coeff=0.0, ce_pos_weight=1.0):
        super().__init__()
        self.dice_coeff = dice_coeff
        self.ce_pos_weight = ce_pos_weight
        self.eps = 1e-6

    def _cross_entropy_with_weight(self, pred, labels):
        p = pred.view(-1).clamp(self.eps, 1.0 - self.eps)
        y = labels.view(-1)
        pos = y > 0
        neg = y == 0

        loss = 0.0
        if pos.numel() > 0:
            p_pos = p[pos]
            loss = loss + (-p_pos.log() * self.ce_pos_weight).mean()
        if neg.numel() > 0:
            p_neg = p[neg]
            loss = loss + (-(1.0 - p_neg).log()).mean()

        return loss

    def _dice_loss(self, pred, labels):
        p = pred.view(-1)
        y = labels.view(-1)
        num = (p * y).sum() * 2 + self.eps
        den = p.sum() + y.sum() + self.eps
        return (num / den).pow(-1)

    def forward(self, pred, labels):
        B = pred.shape[0]
        total_ce = 0.0
        total_dice = 0.0
        for i in range(B):
            total_ce = total_ce + self._cross_entropy_with_weight(pred[i, 0], labels[i, 0])
            total_dice = total_dice + self._dice_loss(pred[i, 0], labels[i, 0])

        ce = total_ce / max(B, 1)
        dc = total_dice / max(B, 1)
        return ce + self.dice_coeff * dc
```

---

## Step 3: Copy Evaluation Functions

From xywnet_v2.2_gbt.ipynb, copy:

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, apply_thinning=True, tolerance_radius=1):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc='Evaluating'):
        images = batch['images'].to(device)
        labels = batch['labels']

        outputs = model(images)

        for i in range(outputs.shape[0]):
            pred_np = outputs[i, 0].cpu().numpy()
            pred_np = nms_edge(pred_np, apply_thinning=apply_thinning)
            pred_np = cv2.GaussianBlur(pred_np, (3,3), 0)
            all_preds.append(pred_np)

            label_np = labels[i, 0].numpy().astype(np.float32)
            all_labels.append(label_np)

    ods, ois, ap = compute_ods_ois_ap(
        all_preds,
        all_labels,
        apply_thinning=apply_thinning,
        tolerance_radius=tolerance_radius
    )
    return ods, ois, ap
```

---

## Step 4: Update Cell 7 (Main Loop)

In the ablation notebook, change:

```python
# OLD (placeholder)
ods, ois, ap, train_loss, best_epoch, elapsed = train_variant_placeholder(
    exp_name, config, epochs=EPOCHS_PER_VARIANT
)

# NEW (actual)
ods, ois, ap, train_loss, best_epoch, elapsed = train_variant(
    exp_name, config, epochs=EPOCHS_PER_VARIANT
)
```

---

## Checklist

Before running ablation study:

-   [ ] Copy `train_epoch()` from main notebook
-   [ ] Copy `evaluate()` from main notebook
-   [ ] Copy loss/metric functions
-   [ ] Implement `build_ablated_xywnet()` factory
-   [ ] Modify encoder to support stage disabling
-   [ ] Modify decoder to support all flags
-   [ ] Create `EdgeLoss` with dice_coeff and ce_pos_weight
-   [ ] Update `nms_edge()` to accept `apply_thinning` flag
-   [ ] Update `compute_ods_ois_ap()` to accept `tolerance_radius` flag
-   [ ] Update Cell 6: replace placeholder with `train_variant()`
-   [ ] Update Cell 7: call `train_variant()` instead of placeholder
-   [ ] Test with 1–2 variants first (validate pipeline works)
-   [ ] Set `EPOCHS_PER_VARIANT` to desired length (5 for quick, 20 for final)
-   [ ] Run full ablation study

---

## Testing

Before full ablation study, test one variant:

```python
# Quick test
config = {'decoder': 'rcf'}
ods, ois, ap, loss, epoch, time = train_variant('test_run', config, epochs=2)
print(f"Test passed: ODS={ods:.4f}, OIS={ois:.4f}, AP={ap:.4f}")
```

If that works, you can safely run the full study.

---

## Performance Tips

-   Use `EPOCHS_PER_VARIANT = 5` for quick exploration
-   Use `BATCH_SIZE = 2` if GPU memory is limited
-   Use `NUM_WORKERS = 0` on Windows (avoid multiprocessing issues)
-   Consider running variants in parallel on multiple GPUs (advanced)
-   Save checkpoints for top-5 variants separately

---

## Questions?

Refer to:

-   Main training notebook: `xywnet_v2.2_gbt.ipynb` (cells 6–9 have all training code)
-   Model definition: cells 4–5 in main notebook
-   Loss/eval: cells 6–7 in main notebook
