import cv2
import numpy as np
from src.segment import segment_image

# You can adjust this threshold (in pixels) to merge boxes with small gaps
DEFAULT_MAX_GAP = 0


def merge_line_boxes(line_boxes, max_gap=DEFAULT_MAX_GAP):
    """
    Given a list of boxes for a single line [(x1,y1,x2,y2), ...],
    merge adjacent boxes whose horizontal gap <= max_gap.
    Returns a new list of merged boxes sorted by x1.
    """
    if not line_boxes:
        return []

    # Sort boxes by x1
    boxes = sorted(line_boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]

    for x1, y1, x2, y2 in boxes[1:]:
        prev = merged[-1]
        prev_x1, prev_y1, prev_x2, prev_y2 = prev

        # Compute horizontal gap
        gap = x1 - prev_x2

        if gap <= max_gap:
            # Merge: update the last box
            merged[-1] = [
                min(prev_x1, x1), min(prev_y1, y1),
                max(prev_x2, x2), max(prev_y2, y2)
            ]
        else:
            merged.append([x1, y1, x2, y2])

    # Convert back to tuples
    return [tuple(b) for b in merged]


def merge_segments(image_path, max_gap=DEFAULT_MAX_GAP, show=False):
    """
    Perform segmentation and merge close/overlapping word boxes per line.

    Args:
        image_path: path to input image
        max_gap: max horizontal gap (in pixels) to merge boxes
        show: if True, display red (orig) & green (merged) boxes

    Returns:
        lines: original component-line boxes
        merged_lines: merged-per-line boxes
        gray: grayscale image array
    """
    # 1) initial segmentation
    lines, gray = segment_image(image_path, preprocess=False, show=False)

    # 2) merge per line
    merged_lines = [merge_line_boxes(line, max_gap) for line in lines]

    # 3) visualize
    if show:
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for line in lines:
            for (x1, y1, x2, y2) in line:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 1)
        for merged in merged_lines:
            for (x1, y1, x2, y2) in merged:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Merged Segments', disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return lines, merged_lines, gray


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Merge nearby segment boxes after segmentation"
    )
    parser.add_argument(
        'input', help='path to input image (grayscale or bin)'
    )
    parser.add_argument(
        '--gap', '-g', type=int, default=DEFAULT_MAX_GAP,
        help='max horizontal gap (px) to merge adjacent boxes'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='display red/original and green/merged boxes'
    )
    args = parser.parse_args()

    merge_segments(args.input, max_gap=args.gap, show=args.show)
