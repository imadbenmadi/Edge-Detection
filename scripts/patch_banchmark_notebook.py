import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"e:\Edge Detection\Comparison\banchmark.ipynb")


def get_cell_text(cell) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def set_cell_text(cell, text: str) -> None:
    cell["source"] = text.splitlines(True)


def patch_config(text: str) -> str:
    # Move to HED_Medium + root-level models folder.
    if "PROJECT_ROOT" not in text:
        text = text.replace(
            "from pathlib import Path\n\n",
            "from pathlib import Path\n\n# Notebook is in ./Comparison, so project root is one level up\nPROJECT_ROOT = Path('..')\n\n",
            1,
        )

    text = text.replace(
        "DATASET_ROOT = Path('../datasets/HED_Small')\n",
        "DATASET_ROOT = PROJECT_ROOT / 'datasets' / 'HED_Medium'\n",
    )

    # Keep CHECKPOINTS_DIR name used later, but point it to root models folder.
    if "MODELS_DIR" not in text:
        text = text.replace(
            "# Download/cache folders\nCHECKPOINTS_DIR = Path('./checkpoints')\nREPOS_DIR = Path('./_repos')",
            "# Download/cache folders\nMODELS_DIR = PROJECT_ROOT / 'models'\nCHECKPOINTS_DIR = MODELS_DIR\nREPOS_DIR = MODELS_DIR / '_repos'",
            1,
        )

    return text


def patch_hed_cell(text: str) -> str:
    # HED expects BGR mean subtraction; predictor feeds RGB 0..1.
    if "RGB -> BGR" not in text:
        text = text.replace(
            "        tenInput = tenInput01 * 255.0\n",
            "        tenInput = tenInput01 * 255.0\n        tenInput = tenInput[:, [2, 1, 0], :, :]  # RGB -> BGR\n",
            1,
        )
    return text


def patch_rcf_cell(text: str) -> str:
    # RCF expects BGR mean subtraction; predictor feeds RGB 0..1.
    if "mean_bgr = torch.tensor([104.00698793" in text:
        return text
    text = text.replace(
        "        img_H, img_W = x01.shape[2], x01.shape[3]\n        x = x01\n",
        "        img_H, img_W = x01.shape[2], x01.shape[3]\n"
        "        x = x01 * 255.0\n"
        "        x = x[:, [2, 1, 0], :, :]  # RGB -> BGR\n"
        "        mean_bgr = torch.tensor([104.00698793, 116.66876762, 122.67891434], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)\n"
        "        x = x - mean_bgr\n",
        1,
    )
    return text


def patch_predictors_cell(text: str) -> str:
    # Fix DexiNed preprocessing + postprocessing: BGR+mean, then sigmoid on final map.
    if "elif model_name == 'DexiNed':" not in text:
        return text

    dex_old = (
        "    elif model_name == 'DexiNed':\n"
        "        outs = model(ten)\n"
        "        prob = outs[-1][0, 0].float().detach().cpu().numpy()\n"
        "        prob = normalize01(prob)\n"
    )

    dex_new = (
        "    elif model_name == 'DexiNed':\n"
        "        # DexiNed expects BGR order + mean subtraction in 0..255 space.\n"
        "        ten_bgr = ten * 255.0\n"
        "        ten_bgr = ten_bgr[:, [2, 1, 0], :, :]  # RGB -> BGR\n"
        "        mean_bgr = torch.tensor([103.939, 116.779, 123.68], device=ten_bgr.device, dtype=ten_bgr.dtype).view(1, 3, 1, 1)\n"
        "        ten_bgr = ten_bgr - mean_bgr\n"
        "        outs = model(ten_bgr)\n"
        "        prob = torch.sigmoid(outs[-1])[0, 0].float().detach().cpu().numpy()\n"
    )

    if dex_old in text:
        return text.replace(dex_old, dex_new, 1)

    # If the notebook has drifted slightly, try a looser replacement.
    return text


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    changed = 0
    for cell in nb.get("cells", []):
        text = get_cell_text(cell)
        new_text = None

        if "# ===== Config =====" in text:
            new_text = patch_config(text)
        elif "# ===== HED (PyTorch, pretrained) =====" in text:
            new_text = patch_hed_cell(text)
        elif "# ===== RCF (PyTorch, pretrained) =====" in text:
            new_text = patch_rcf_cell(text)
        elif "# ===== Predictors (HED / RCF / DexiNed / XDoG) =====" in text:
            new_text = patch_predictors_cell(text)

        if new_text is not None and new_text != text:
            set_cell_text(cell, new_text)
            changed += 1

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched cells:", changed)


if __name__ == "__main__":
    main()