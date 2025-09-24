import cv2
import numpy as np
import argparse
from src.preprocess import preprocess_image

# — tune these to suit your data —
MIN_BLOB_AREA      = 100        # minimum component area to keep
LINE_Y_THRESHOLD_F = 0.5        # fraction of median line height for grouping
WORD_MIN_WIDTH_F   = 0.02       # fraction of image width for word filtering


def segment_components(bin_img):
    """
    Find connected components and return bounding boxes
    for those above MIN_BLOB_AREA.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= MIN_BLOB_AREA:
            boxes.append((x, y, x + w, y + h))
    return boxes


def group_lines(boxes):
    """
    Group boxes into lines by clustering on vertical centers.
    """
    if not boxes:
        return []
    centers = [(y1 + y2) / 2 for (_, y1, _, y2) in boxes]
    heights = [(y2 - y1) for (_, y1, _, y2) in boxes]
    median_h = np.median(heights)
    thresh = median_h * LINE_Y_THRESHOLD_F

    items = sorted(zip(centers, boxes), key=lambda cb: cb[0])
    lines = []
    curr_line = [items[0][1]]
    prev_c = items[0][0]
    for c, box in items[1:]:
        if abs(c - prev_c) <= thresh:
            curr_line.append(box)
        else:
            lines.append(sorted(curr_line, key=lambda b: b[0]))
            curr_line = [box]
        prev_c = c
    lines.append(sorted(curr_line, key=lambda b: b[0]))
    return lines


def extract_word_boxes(line_boxes, img_w):
    """
    Filter out narrow boxes unlikely to be full words.
    """
    words = []
    for x1, y1, x2, y2 in line_boxes:
        if (x2 - x1) >= img_w * WORD_MIN_WIDTH_F:
            words.append((x1, y1, x2, y2))
    return words


def segment_image(path, preprocess=False, show=False):
    """
    path: image path
    preprocess: if True, run preprocess_image() first
    show: if True, draw word boxes on image
    Returns: (lines, gray_img)
    """
    # 1) Load image
    if preprocess:
        gray = preprocess_image(path, show=False)
    else:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot open image: {path}")

    # 2) Binarize for connected components (invert so text=1)
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 3) Segment and line-group
    comps = segment_components(bin_img)
    lines = group_lines(comps)

    # 4) Extract word-level boxes
    word_boxes = []
    for line in lines:
        wbs = extract_word_boxes(line, gray.shape[1])
        word_boxes.extend(wbs)

    # 5) Visualize if required
    if show:
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x1, y1, x2, y2) in word_boxes:
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Segmentation", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return lines, gray


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to input image (grayscale or bin)")
    parser.add_argument("--preprocess", action="store_true",
                        help="apply preprocessing before segmenting")
    parser.add_argument("--show", action="store_true",
                        help="visualize word boxes on image")
    args = parser.parse_args()
    segment_image(args.input, preprocess=args.preprocess, show=args.show)
