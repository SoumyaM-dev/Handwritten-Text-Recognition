# src/word_inference.py

import torch
from .utils_image import normalize_and_pad

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def crop_and_recognize(img, bboxes, model):
    """
    Given a grayscale image and a list of bounding‐boxes,
    pad each box to 32×128, run through the model, and CTC‐decode.
    """
    from .model_ctc_resnet import ctc_decode  # Local import to avoid circular issues

    tensors = []
    for (x1, y1, x2, y2) in bboxes:
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            tensors.append(torch.zeros((1,32,128), device=DEVICE))
        else:
            t = normalize_and_pad(patch, target_size=(32,128)).to(DEVICE)
            tensors.append(t)

    if not tensors:
        return []

    batch = torch.stack(tensors, dim=0)   # [N,1,32,128]
    with torch.no_grad():
        logits = model(batch)             # [T, N, C]
    return ctc_decode(logits)             # list[str] of length N
