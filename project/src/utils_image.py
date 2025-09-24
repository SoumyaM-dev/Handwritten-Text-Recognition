# src/utils_image.py
from PIL import Image, ImageOps
import torchvision.transforms.functional as F
import numpy as np

def normalize_and_pad(word_img, target_size=(32, 128), fill=255):
    """
    - Scales word_img so it fits *inside* target_size, preserving aspect ratio.
    - Pads evenly on all sides to exactly target_size.
    - Returns a FloatTensor [1, H, W] with values in [0,1].
    """
    # word_img: np.ndarray HxW (grayscale) or HxWxC (RGB)
    h, w = word_img.shape[:2]
    th, tw = target_size

    # 1) Compute uniform scale to fit in box
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 2) Resize with high‑quality resampling
    mode = "RGB" if (word_img.ndim == 3 and word_img.shape[2] == 3) else "L"
    pil = Image.fromarray(word_img, mode=mode)
    pil = pil.resize((new_w, new_h), Image.LANCZOS)

    # 3) Compute symmetric padding
    pad_w = tw - new_w
    pad_h = th - new_h
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)

    pil = ImageOps.expand(pil, border=(left, top, right, bottom), fill=fill)

    # 4) Convert to tensor [C,H,W] with values 0–1
    tensor = F.to_tensor(pil)
    # if RGB, collapse to single channel by mean
    if tensor.shape[0] == 3:
        tensor = tensor.mean(dim=0, keepdim=True)
    return tensor
