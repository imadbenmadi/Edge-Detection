import cv2
import numpy as np
from PIL import Image, ImageOps


def resize_image(img, size, is_edge_map=False):
    """
    Aspect-ratio preserving letterbox resize:
    - Scale the image so that it fits within target size without distortion.
    - Pad the remaining area to reach exact target size.
    
    For edge maps (grayscale, potentially soft labels):
    - Downscale with bilinear (preserves fractional labels)
    - Upscale with nearest (avoids creating spurious values)
    - Keep float in [0,1] until padding; convert to uint8 only at the end
    """
    target_w, target_h = size
    w, h = img.size

    # Compute scale while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if is_edge_map:
        # Convert to float array in [0,1]
        arr = np.array(img).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        # Choose interpolation based on scaling direction
        interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_LINEAR
        arr_resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)

        # Pad to target size with zeros
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        arr_padded = np.pad(
            arr_resized,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0.0,
        )

        out = (np.clip(arr_padded, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(out, mode='L')
    else:
        # High-quality downscale for RGB
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        # Pad with black to reach target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img_padded = ImageOps.expand(img_resized, border=pad, fill=(0, 0, 0))
        return img_padded
    
# Uses LANCZOS for downscaling, BICUBIC for upscaling (RGB)
def resize_image_v2(img, size, is_edge_map=False):
    """
    Research-grade, aspect-ratio preserving letterbox resize.

    - Preserves geometry (no distortion)
    - Uses interpolation based on scaling direction
    - Protects edge-label integrity
    - Produces pixel-aligned RGB + edge maps
    - Guaranteed output size
    """

    target_w, target_h = size
    w, h = img.size
    # Safety check
    assert w > 0 and h > 0, "Invalid image size"
    # Aspect-ratio preserving scale
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Center padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top

    if is_edge_map:
        # --- EDGE MAP PIPELINE ---
        arr = np.array(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 255.0  # normalize safely to [0,1]

        interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_LINEAR
        arr_resized = cv2.resize(
            arr,
            (new_w, new_h),
            interpolation=interp
        )
        arr_padded = np.pad(
            arr_resized,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0
        )
        arr_padded = np.clip(arr_padded, 0.0, 1.0)
        out = (arr_padded * 255.0).astype(np.uint8)
        return Image.fromarray(out, mode="L")
    else:
        # --- RGB IMAGE PIPELINE ---
        interp = Image.BICUBIC if scale >= 1.0 else Image.LANCZOS
        img_resized = img.resize(
            (new_w, new_h),
            interp
        )
        img_padded = ImageOps.expand(
            img_resized,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=(0, 0, 0)
        )
        # Final safety check
        assert img_padded.size == (target_w, target_h), \
            f"Output size mismatch: {img_padded.size} vs {(target_w, target_h)}"

        return img_padded
