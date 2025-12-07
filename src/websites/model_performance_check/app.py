import os
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# ------------------------------
# Config
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIRS = [
    Path("e:/Edge Detection/models"),
    Path("e:/Edge Detection/src/XYW-Net_original")
]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(BASE_DIR / "static"), template_folder=str(BASE_DIR / "templates"))
app.secret_key = "xywnet-model-check-secret"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB uploads

# ------------------------------
# >>> FIX ADDED HERE <<<
# LETTERBOX RESIZE (must match Kaggle training)
# ------------------------------
def letterbox_resize(img, target_size=(512, 512)):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Use reflect padding to avoid artificial hard edges at borders
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REFLECT_101
    )

    # Valid content mask (1 inside original content, 0 on padding)
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = 1

    return padded, mask

# ------------------------------
# Minimal XYWNet (Your original unchanged)
# ------------------------------
class Xc1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Xcenter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Xcenter_relu = nn.ReLU(inplace=True)
        self.Xsurround = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.Xsurround_relu = nn.ReLU(inplace=True)
    def forward(self, input):
        xcenter = self.Xcenter_relu(self.Xcenter(input))
        xsurround = self.Xsurround_relu(self.Xsurround(input))
        xsurround = self.conv1_1(xsurround)
        return xsurround - xcenter

class Yc1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Ycenter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Ycenter_relu = nn.ReLU(inplace=True)
        self.Ysurround = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=4, dilation=2, groups=in_channels)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.Ysurround_relu = nn.ReLU(inplace=True)
    def forward(self, input):
        ycenter = self.Ycenter_relu(self.Ycenter(input))
        ysurround = self.Ysurround_relu(self.Ysurround(input))
        ysurround = self.conv1_1(ysurround)
        return ysurround - ycenter

class W(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.h = nn.Conv2d(inchannel, inchannel, kernel_size=(1, 3), padding=(0, 1), groups=inchannel)
        self.v = nn.Conv2d(inchannel, inchannel, kernel_size=(3, 1), padding=(1, 0), groups=inchannel)
        self.convh_1 = nn.Conv2d(inchannel, inchannel, kernel_size=1, bias=False)
        self.convv_1 = nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.h(x))
        h = self.convh_1(h)
        v = self.relu(self.v(h))
        v = self.convv_1(v)
        return v

class XYW_S(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)
    def forward(self, x):
        xc = self.x_c(x)
        yc = self.y_c(x)
        w = self.w(x)
        return xc, yc, w, {"xc": xc, "yc": yc, "w": w}

class XYW(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)
    def forward(self, xc, yc, w):
        xc2 = self.x_c(xc)
        yc2 = self.y_c(yc)
        w2 = self.w(w)
        return xc2, yc2, w2, {"xc": xc2, "yc": yc2, "w": w2}

class XYW_E(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)
    def forward(self, xc, yc, w):
        xout = self.x_c(xc)
        yout = self.y_c(yc)
        wout = self.w(w)
        return xout + yout + wout, {"xc": xout, "yc": yout, "w": wout}

class s1(nn.Module):
    def __init__(self, channel=30):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channel, kernel_size=7, padding=6, dilation=2)
        self.xyw1_1 = XYW_S(channel, channel)
        self.xyw1_2 = XYW(channel, channel)
        self.xyw1_3 = XYW_E(channel, channel)
        self.relu = nn.ReLU()
    def forward(self, x):
        temp = self.relu(self.conv1(x))
        xc1, yc1, w1, s1_1 = self.xyw1_1(temp)
        xc2, yc2, w2, s1_2 = self.xyw1_2(xc1, yc1, w1)
        xyw1_3, s1_3 = self.xyw1_3(xc2, yc2, w2)
        out = xyw1_3 + temp
        return out, {"pre": temp, "blocks": [s1_1, s1_2, s1_3]}

class s2(nn.Module):
    def __init__(self, channel=60):
        super().__init__()
        self.xyw2_1 = XYW_S(channel//2, channel)
        self.xyw2_2 = XYW(channel, channel)
        self.xyw2_3 = XYW_E(channel, channel)
        self.shortcut = nn.Conv2d(channel//2, channel, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.pool(x)
        xc1, yc1, w1, s2_1 = self.xyw2_1(x)
        xc2, yc2, w2, s2_2 = self.xyw2_2(xc1, yc1, w1)
        xyw2_3, s2_3 = self.xyw2_3(xc2, yc2, w2)
        out = xyw2_3 + self.shortcut(x)
        return out, {"pre": x, "blocks": [s2_1, s2_2, s2_3]}

class s3(nn.Module):
    def __init__(self, channel=120):
        super().__init__()
        self.xyw3_1 = XYW_S(channel//2, channel)
        self.xyw3_2 = XYW(channel, channel)
        self.xyw3_3 = XYW_E(channel, channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(channel//2, channel, kernel_size=1)
    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        xc1, yc1, w1, s3_1 = self.xyw3_1(x)
        xc2, yc2, w2, s3_2 = self.xyw3_2(xc1, yc1, w1)
        xyw3_3, s3_3 = self.xyw3_3(xc2, yc2, w2)
        out = xyw3_3 + shortcut
        return out, {"pre": x, "blocks": [s3_1, s3_2, s3_3]}

class s4(nn.Module):
    def __init__(self, channel=120):
        super().__init__()
        self.xyw4_1 = XYW_S(channel, channel)
        self.xyw4_2 = XYW(channel, channel)
        self.xyw4_3 = XYW_E(channel, channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(channel, channel, kernel_size=1)
    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        xc1, yc1, w1, s4_1 = self.xyw4_1(x)
        xc2, yc2, w2, s4_2 = self.xyw4_2(xc1, yc1, w1)
        xyw4_3, s4_3 = self.xyw4_3(xc2, yc2, w2)
        out = xyw4_3 + shortcut
        return out, {"pre": x, "blocks": [s4_1, s4_2, s4_3]}

class encode(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = s1()
        self.s2 = s2()
        self.s3 = s3()
        self.s4 = s4()
    def forward(self, x):
        s1_out, s1_dbg = self.s1(x)
        s2_out, s2_dbg = self.s2(s1_out)
        s3_out, s3_dbg = self.s3(s2_out)
        s4_out, s4_dbg = self.s4(s3_out)
        return (s1_out, s2_out, s3_out, s4_out), {"s1": s1_dbg, "s2": s2_dbg, "s3": s3_dbg, "s4": s4_dbg}

class Refine_block2_1(nn.Module):
    def __init__(self, in_channel, out_channel, factor):
        super().__init__()
        self.pre_conv1 = nn.Sequential(
            nn.Conv2d(in_channel[0], out_channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.pre_conv2 = nn.Sequential(
            nn.Conv2d(in_channel[1], out_channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.factor = factor
    def forward(self, x1_in, x2_in):
        x1 = self.pre_conv1(x1_in)
        x2 = self.pre_conv2(x2_in)
        x2 = F.interpolate(x2, scale_factor=self.factor, mode="bilinear", align_corners=False)
        return x1 + x2

class decode_rcf(nn.Module):
    def __init__(self):
        super().__init__()
        self.f43 = Refine_block2_1(in_channel=(120, 120), out_channel=60, factor=2)
        self.f32 = Refine_block2_1(in_channel=(60, 60), out_channel=30, factor=2)
        self.f21 = Refine_block2_1(in_channel=(30, 30), out_channel=24, factor=2)
        self.f = nn.Conv2d(24, 1, kernel_size=1, padding=0)
    def forward(self, x):
        s3 = self.f43(x[2], x[3])
        s2 = self.f32(x[1], s3)
        s1 = self.f21(x[0], s2)
        out = self.f(s1)
        return torch.sigmoid(out)

class XYWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = encode()
        self.decode = decode_rcf()
    def forward(self, x, return_debug=False):
        endpoints, dbg = self.encode(x)
        out = self.decode(endpoints)
        if return_debug:
            return out, dbg
        return out

# ------------------------------
# Helpers
# ------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def list_models():
    candidates = []
    for d in MODELS_DIRS:
        if d.exists():
            for p in d.rglob("*.pth"):
                candidates.append(str(p))
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def load_model(model_path: str, device: str = "cpu"):
    if not model_path:
        raise RuntimeError("Model path not provided")
    model_path = str(Path(model_path))
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = XYWNet()
    model.to(device)
    try:
        # Load weights-only; require XYWNet-like state dict
        state = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']
        if not isinstance(state, dict):
            raise RuntimeError("Unsupported checkpoint format. Expected a state_dict (dict of tensors).")
        # Minimal schema check: ensure expected keys exist
        expected_keys = [
            'encode.s1.conv1.weight',
            'decode.f.weight'
        ]
        missing_expected = [k for k in expected_keys if k not in state]
        if missing_expected:
            raise RuntimeError("Uploaded model is not XYWNet-compatible (missing expected keys).")
        model.load_state_dict(state, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {model_path}: {e}")
    model.eval()
    return model


def _to_vis(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr * 255.0).astype(np.uint8)

# ------------------------------
# >>> FIX APPLIED HERE <<<
# run_inference now correctly preprocesses input
# ------------------------------
def run_inference(model: nn.Module, img_path: Path, device: str = "cpu", with_debug: bool = False):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise RuntimeError("Could not read image")

    # Letterbox with reflect padding and get valid mask
    bgr_lb, valid_mask = letterbox_resize(bgr, (512, 512))

    rgb = cv2.cvtColor(bgr_lb, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        if with_debug:
            pred, dbg = model(tensor, return_debug=True)
        else:
            pred = model(tensor)

    pred = pred[0, 0].cpu().numpy()
    # Align mask to prediction size to avoid broadcast errors
    mh, mw = pred.shape
    mask_resized = cv2.resize(valid_mask, (mw, mh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    # Zero-out padding influence
    pred = pred * mask_resized
    pred_u8 = _to_vis(pred)

    debug_images = {}
    if with_debug:
        for stage_name in ["s1", "s2", "s3", "s4"]:
            info = dbg.get(stage_name, {})
            if not info:
                continue
            pre = info.get("pre")
            blocks = info.get("blocks", [])
            if pre is not None:
                pre_vis_arr = pre[0].mean(dim=0).cpu().numpy()
                ph, pw = pre_vis_arr.shape
                vm_pre = cv2.resize(valid_mask, (pw, ph), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                pre_vis = _to_vis(pre_vis_arr)
                pre_vis = (pre_vis.astype(np.float32) * vm_pre).astype(np.uint8)
                debug_images[f"{stage_name}_pre"] = pre_vis
            for bi, block in enumerate(blocks, start=1):
                for k in ["xc", "yc", "w"]:
                    t = block.get(k)
                    if t is None:
                        continue
                    vis_arr = t[0].mean(dim=0).cpu().numpy()
                    th, tw = vis_arr.shape
                    vm_t = cv2.resize(valid_mask, (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                    vis = _to_vis(vis_arr)
                    vis = (vis.astype(np.float32) * vm_t).astype(np.uint8)
                    debug_images[f"{stage_name}_b{bi}_{k}"] = vis

    return rgb, pred_u8, debug_images

# ------------------------------
# Routes
# ------------------------------

@app.route("/")
def index():
    return render_template("viewer.html")

@app.route("/predict", methods=["POST"])
def predict():
    model_path = request.form.get("model_path")
    model_file = request.files.get("model_file")
    file = request.files.get("image_file")
    show_debug = request.form.get("show_debug") == "on"

    if not file or file.filename == "":
        flash("Please select an image file.", "error")
        return render_template("viewer.html")
    if not allowed_file(file.filename):
        flash("Unsupported file type. Use png/jpg/jpeg.", "error")
        return render_template("viewer.html")

    filename = secure_filename(file.filename)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_name = f"{ts}_{filename}"
    save_path = UPLOAD_DIR / save_name
    file.save(str(save_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved_model_path = None
    if model_file and model_file.filename:
        if not model_file.filename.lower().endswith('.pth'):
            flash("Model file must be a .pth checkpoint.", "error")
            return render_template("viewer.html")
        mp_name = secure_filename(model_file.filename)
        mp_save = UPLOAD_DIR / f"{ts}_{mp_name}"
        model_file.save(str(mp_save))
        resolved_model_path = str(mp_save)
    elif model_path:
        resolved_model_path = model_path
    else:
        flash("Please select or provide a .pth model file.", "error")
        return render_template("viewer.html")

    try:
        model = load_model(resolved_model_path, device=device)
        rgb, pred_u8, debug_images = run_inference(model, save_path, device=device, with_debug=show_debug)
    except Exception as e:
        flash(f"Inference failed: {e}", "error")
        return render_template("viewer.html", selected_model=resolved_model_path, show_debug=show_debug)

    rgb_out = RESULTS_DIR / f"{ts}_input.png"
    pred_out = RESULTS_DIR / f"{ts}_edges.png"
    cv2.imwrite(str(rgb_out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(pred_out), pred_u8)

    debug_urls = {}
    if show_debug:
        for name, img_arr in debug_images.items():
            p = RESULTS_DIR / f"{ts}_{name}.png"
            cv2.imwrite(str(p), img_arr)
            debug_urls[name] = url_for("serve_result", filename=p.name)

    return render_template(
        "viewer.html",
        selected_model=resolved_model_path,
        input_image=url_for("serve_result", filename=rgb_out.name),
        edge_image=url_for("serve_result", filename=pred_out.name),
        show_debug=show_debug,
        debug_urls=debug_urls
    )

@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(str(RESULTS_DIR), filename)

# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
