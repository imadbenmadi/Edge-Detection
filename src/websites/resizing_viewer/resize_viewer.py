from pathlib import Path
from io import BytesIO
import base64

from flask import Flask, render_template, request
from PIL import Image
import numpy as np

from Resize_image import resize_image_v2 as resize_image

app = Flask(__name__)

TARGET_SIZE = (512, 512)  # Fixed output size


def pil_to_data_url(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format='PNG')
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{data}"


@app.route('/', methods=['GET'])
def index():
    return render_template('viewer.html')


@app.route('/preview', methods=['POST'])
def preview():
    files = request.files.getlist('images')
    # Global mode acts as a hint, but we auto-detect per-file
    mode_hint = request.form.get('mode', 'rgb')  # 'rgb' or 'edge'
    size = TARGET_SIZE

    items = []
    for f in files:
        if not f or f.filename == '':
            continue
        try:
            img = Image.open(f.stream)

            # Auto-detect edge vs RGB
            filename_lower = f.filename.lower()
            looks_like_edge_name = any(k in filename_lower for k in ['edge', 'gt', 'mask', 'boundary'])
            # If image already single channel or named like edge, treat as edge
            if img.mode in ['1', 'L'] or looks_like_edge_name or mode_hint == 'edge':
                img = img.convert('L')
                is_edge = True
            else:
                img = img.convert('RGB')
                is_edge = False

            resized = resize_image(img, size, is_edge_map=is_edge)

            items.append({
                'filename': f.filename,
                'original_url': pil_to_data_url(img),
                'resized_url': pil_to_data_url(resized),
                'orig_size': f"{img.size[0]}×{img.size[1]}",
                'new_size': f"{resized.size[0]}×{resized.size[1]}",
            })
        except Exception as e:
            items.append({
                'filename': f.filename,
                'error': str(e)
            })

    return render_template('viewer.html', items=items, mode=mode_hint, target_w=size[0], target_h=size[1])


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
