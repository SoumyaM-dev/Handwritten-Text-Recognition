#!/usr/bin/env python3
import os
import sys

# Ensure project root is on PYTHONPATH so `src/` is importable
sys.path.insert(0, os.getcwd())

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Import merged segmentation
from src.merge_segments import merge_segments
from src.utils_image import normalize_and_pad


def main(img_path, gap=70, min_area=500):
    """
    1️⃣ Merge segments (green boxes) with given horizontal gap
    2️⃣ Apply area filter, pad each patch
    3️⃣ Visualize padded patches in a grid

    Args:
        img_path: path to input image
        gap: max horizontal gap (px) to merge adjacent boxes
        min_area: min area of merged box to consider
    """
    # 1️⃣ Segment, merge
    lines, merged_lines, cleaned = merge_segments(
        img_path,
        max_gap=gap,
        show=False
    )

    # 2️⃣ Collect merged boxes above the area threshold
    boxes = []
    for line in merged_lines:
        for (x1, y1, x2, y2) in line:
            if (x2 - x1) * (y2 - y1) >= min_area:
                boxes.append((x1, y1, x2, y2))

    if not boxes:
        print("No patches (try lowering min_area or increasing gap).")
        return

    # 3️⃣ Normalize + pad each patch to 32×128
    patches = []
    for (x1, y1, x2, y2) in boxes:
        patch = cleaned[y1:y2, x1:x2]
        tensor = normalize_and_pad(patch, target_size=(32,128))
        arr = tensor.squeeze(0).cpu().numpy()  # H×W array in [0,1]
        patches.append(arr)

    # 4️⃣ Plot grid of patches with red border
    n = len(patches)
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))

    # Flatten axes
    axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, ax in enumerate(axes_list):
        if idx < n:
            img = patches[idx]
            h, w = img.shape
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            rect = Rectangle((0, 0), w, h, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        else:
            ax.remove()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_patches.py path/to/image.jpg [gap] [min_area]")
        sys.exit(1)
    img_path = sys.argv[1]
    gap = int(sys.argv[2]) if len(sys.argv) > 2 else 70
    min_area = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    main(img_path, gap, min_area)
