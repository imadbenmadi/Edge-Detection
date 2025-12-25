# Cell 6: Train XYW-Net (RESUME FROM 3epoch -> 20)

# -------------------------------
# Hyperparameters
# -------------------------------
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# -------------------------------
# Initialize
# -------------------------------
model = XYWNet().to(device)
criterion = EdgeLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# -------------------------------
# Training history
# -------------------------------
history = {'train_loss': [], 'val_ods': [], 'val_ois': [], 'val_ap': []}
best_ods = 0.0
best_epoch = 0

# -------------------------------
# Crash-safe per-epoch logging + dual-backup (recommended for unstable/remote runs)
# -------------------------------
import csv
import time
import shutil

EPOCH_METRICS_CSV = ARTIFACTS_DIR / "epoch_metrics.csv"

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))

def _atomic_torch_save(obj, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))

def _append_epoch_row(row: dict) -> None:
    is_new = not EPOCH_METRICS_CSV.exists()
    with open(EPOCH_METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

# -------------------------------
# Backup target #1: local/attached path (external drive, network share, synced folder)
# -------------------------------
# Configure via the "Backup + S3 configuration" cell (no env vars).
BACKUP_DIR = globals().get('BACKUP_DIR', None)
BACKUP_BASE = Path(BACKUP_DIR) if BACKUP_DIR else None
BACKUP_RUN_DIR = None
if BACKUP_BASE is not None:
    # Keep runs separated to avoid name collisions
    BACKUP_RUN_DIR = BACKUP_BASE / Path(RUN_DIR).name
    (BACKUP_RUN_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (BACKUP_RUN_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
    print(f"✓ Local backup enabled -> {BACKUP_RUN_DIR}")
else:
    print("(local backup disabled) Set BACKUP_DIR in the Backup cell to enable")

def _backup_to_local(src: Path, rel_dest: str) -> None:
    if BACKUP_RUN_DIR is None:
        return
    try:
        src = Path(src)
        if not src.exists():
            return
        dest = BACKUP_RUN_DIR / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(str(src), str(tmp))
        os.replace(str(tmp), str(dest))
    except Exception as e:
        print(f"! Local backup warning: {e}")

# -------------------------------
# Backup target #2: S3 / S3-compatible object storage
# -------------------------------
# Configure via notebook variables in the Backup cell (no env vars):
#   S3_BUCKET        (required to enable)
#   S3_PREFIX        (optional)
#   S3_ENDPOINT_URL  (optional for MinIO/OSS/COS; AWS can omit)
#   S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY / S3_REGION (optional)
S3_BUCKET = globals().get('S3_BUCKET', None) or None
S3_PREFIX = str(globals().get('S3_PREFIX', 'xyw-net')).strip().strip('/')
S3_ENDPOINT_URL = globals().get('S3_ENDPOINT_URL', None) or None
S3_REGION = globals().get('S3_REGION', None) or None
S3_ACCESS_KEY_ID = globals().get('S3_ACCESS_KEY_ID', None) or None
S3_SECRET_ACCESS_KEY = globals().get('S3_SECRET_ACCESS_KEY', None) or None
S3_ENABLED = False
S3_CLIENT = globals().get('S3_CLIENT', None)
if S3_BUCKET and S3_CLIENT is None:
    try:
        import boto3
        client_kwargs = {}
        if S3_ENDPOINT_URL:
            client_kwargs['endpoint_url'] = S3_ENDPOINT_URL
        if S3_REGION:
            client_kwargs['region_name'] = S3_REGION
        if S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY:
            client_kwargs['aws_access_key_id'] = S3_ACCESS_KEY_ID
            client_kwargs['aws_secret_access_key'] = S3_SECRET_ACCESS_KEY
        S3_CLIENT = boto3.client('s3', **client_kwargs)
    except Exception as e:
        S3_CLIENT = None
        print(f"! S3 disabled (boto3/config issue): {e}")

if S3_BUCKET and S3_CLIENT is not None:
    S3_ENABLED = True
    print(f"✓ S3 backup enabled -> s3://{S3_BUCKET}/{S3_PREFIX}/{Path(RUN_DIR).name}/...")
else:
    print("(S3 backup disabled) Set S3_BUCKET in the Backup cell to enable")

def _s3_key(rel_dest: str) -> str:
    rel_dest = rel_dest.replace('\\', '/').lstrip('/')
    return f"{S3_PREFIX}/{Path(RUN_DIR).name}/{rel_dest}"

def _backup_to_s3(src: Path, rel_dest: str) -> None:
    if not S3_ENABLED or S3_CLIENT is None:
        return
    try:
        src = Path(src)
        if not src.exists():
            return
        key = _s3_key(rel_dest)
        S3_CLIENT.upload_file(str(src), S3_BUCKET, key)
    except Exception as e:
        print(f"! S3 backup warning: {e}")

# -------------------------------
# Resume from checkpoint if available
# -------------------------------
start_epoch = 3
resume_msg = ""
try:
    legacy_ckpt = Path('models') / '3epoch.pth'
    run_ckpt = CHECKPOINT_DIR / '3epoch.pth'
    full_ckpt_run = CHECKPOINT_DIR / 'xywnet_full_checkpoint.pth'
    full_ckpt_legacy = Path('models') / 'xywnet_full_checkpoint.pth'

    if run_ckpt.exists():
        print(f"Loading pretrained weights from {run_ckpt} ...")
        state = torch.load(str(run_ckpt), map_location=device)
        model.load_state_dict(state)
        start_epoch = 4
        resume_msg = f"Resuming training from epoch {start_epoch} using run checkpoint 3epoch.pth"
    elif legacy_ckpt.exists():
        print(f"Loading pretrained weights from {legacy_ckpt} ...")
        state = torch.load(str(legacy_ckpt), map_location=device)
        model.load_state_dict(state)
        start_epoch = 4
        resume_msg = f"Resuming training from epoch {start_epoch} using legacy models/3epoch.pth"
    elif full_ckpt_run.exists() or full_ckpt_legacy.exists():
        ckpt_path = full_ckpt_run if full_ckpt_run.exists() else full_ckpt_legacy
        print(f"Loading full checkpoint {ckpt_path} ...")
        full = torch.load(str(ckpt_path), map_location=device)
        if 'model_state' in full:
            model.load_state_dict(full['model_state'])
        if 'optimizer_state' in full:
            try:
                optimizer.load_state_dict(full['optimizer_state'])
            except Exception:
                pass
        if 'scheduler_state' in full:
            try:
                scheduler.load_state_dict(full['scheduler_state'])
            except Exception:
                pass
        best_ods = float(full.get('best_ods', 0.0))
        best_epoch = int(full.get('best_epoch', 0))
        start_epoch = int(full.get('epoch', 0)) + 1
        if isinstance(full.get('history', None), dict):
            history = full['history']
        resume_msg = f"Resuming training from epoch {start_epoch} using full checkpoint"
    else:
        resume_msg = "No resume checkpoint found; starting at epoch 3"
except Exception as e:
    resume_msg = f"Resume load failed: {e}; starting at epoch 3"

print(resume_msg)
print(f"Training XYW-Net from epoch {start_epoch} to {NUM_EPOCHS}...")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"Checkpoints: {CHECKPOINT_DIR}")
print(f"Artifacts:   {ARTIFACTS_DIR}")
print("="*60)

def save_full_checkpoint(path: Path, epoch: int, best_ods_val: float):
    _atomic_torch_save({
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_ods": float(best_ods_val),
        "best_epoch": int(best_epoch),
        "history": history,
    }, path)

# -------------------------------
# TRAINING LOOP (resume-style indexing)
# -------------------------------
t0 = time.time()
for epoch in range(start_epoch, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    history['train_loss'].append(float(train_loss))
    
    # Evaluate on validation
    val_ods, val_ois, val_ap = evaluate(model, val_loader, device)
    history['val_ods'].append(float(val_ods))
    history['val_ois'].append(float(val_ois))
    history['val_ap'].append(float(val_ap))
    
    scheduler.step()
    
    print(f"Loss: {train_loss:.4f} | ODS: {val_ods:.4f} | OIS: {val_ois:.4f} | AP: {val_ap:.4f}")
    
    # Save per-epoch full checkpoint
    epoch_ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch:03d}_full.pth"
    save_full_checkpoint(epoch_ckpt_path, epoch, best_ods)
    
    is_best = bool(val_ods > best_ods)
    # Save best model
    if is_best:
        best_ods = float(val_ods)
        best_epoch = int(epoch)
        
        best_weights_path = CHECKPOINT_DIR / "best_xyw_net.pth"
        best_full_path = CHECKPOINT_DIR / "xywnet_full_checkpoint.pth"
        _atomic_torch_save(model.state_dict(), CHECKPOINT_DIR / f"{epoch}epoch.pth")
        _atomic_torch_save(model.state_dict(), best_weights_path)
        save_full_checkpoint(best_full_path, epoch, best_ods)
        
        # Export deployment .pt model
        real_sample = train_dataset[0]
        example_input = real_sample['images'].unsqueeze(0).to(device)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(str(ARTIFACTS_DIR / "xywnet_model.pt"))
        
        print(f"  -> Saved best model (ODS: {best_ods:.4f})")
        print(f"  -> Saved best weights: {best_weights_path}")
        print(f"  -> Saved traced model: {ARTIFACTS_DIR / 'xywnet_model.pt'}")
    
# Persist epoch metrics + history snapshot (crash-safe)
    history_path = ARTIFACTS_DIR / "history.json"
    summary_path = ARTIFACTS_DIR / "run_summary.json"
    _append_epoch_row({
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_ods": float(val_ods),
        "val_ois": float(val_ois),
        "val_ap": float(val_ap),
        "best_ods": float(best_ods),
        "best_epoch": int(best_epoch),
        "is_best": int(is_best),
        "elapsed_sec": float(time.time() - t0),
    })
    _atomic_write_text(history_path, json.dumps(history, indent=2))
    _atomic_write_text(summary_path, json.dumps({"best_ods": best_ods, "best_epoch": best_epoch}, indent=2))

    # Dual-backup after each epoch (local folder + S3).
    _backup_to_local(epoch_ckpt_path, f"checkpoints/{epoch_ckpt_path.name}")
    _backup_to_local(EPOCH_METRICS_CSV, "artifacts/epoch_metrics.csv")
    _backup_to_local(history_path, "artifacts/history.json")
    _backup_to_local(summary_path, "artifacts/run_summary.json")
    _backup_to_s3(epoch_ckpt_path, f"checkpoints/{epoch_ckpt_path.name}")
    _backup_to_s3(EPOCH_METRICS_CSV, "artifacts/epoch_metrics.csv")
    _backup_to_s3(history_path, "artifacts/history.json")
    _backup_to_s3(summary_path, "artifacts/run_summary.json")
    if is_best:
        _backup_to_local(CHECKPOINT_DIR / "best_xyw_net.pth", "checkpoints/best_xyw_net.pth")
        _backup_to_local(CHECKPOINT_DIR / "xywnet_full_checkpoint.pth", "checkpoints/xywnet_full_checkpoint.pth")
        _backup_to_local(ARTIFACTS_DIR / "xywnet_model.pt", "artifacts/xywnet_model.pt")
        _backup_to_s3(CHECKPOINT_DIR / "best_xyw_net.pth", "checkpoints/best_xyw_net.pth")
        _backup_to_s3(CHECKPOINT_DIR / "xywnet_full_checkpoint.pth", "checkpoints/xywnet_full_checkpoint.pth")
        _backup_to_s3(ARTIFACTS_DIR / "xywnet_model.pt", "artifacts/xywnet_model.pt")

print("\n" + "="*60)
print(f"Training complete. Best ODS: {best_ods:.4f} (epoch {best_epoch})")
print(f"All files saved to: {RUN_DIR}")