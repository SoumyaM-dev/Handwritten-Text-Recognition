import cv2
import numpy as np
import argparse

# ↓ You can tweak these parameters
MEDIAN_BLUR_KSIZE = 3       # for light denoising
CONTRAST_ALPHA    = 1.1     # slight contrast boost
BRIGHTNESS_BETA   = 10      # slight brightness boost
ADAPT_BLOCK       = 21      # block size for adaptive threshold
ADAPT_C           = 10      # constant subtracted from the mean
MORPH_KERNEL      = (3,3)   # kernel for morphology
MIN_SPECKLE       = 50      # minimum area to keep a contour
MAX_DESKEW_ANGLE  = 5       # cap deskew correction to ±5 degrees


def preprocess_image(path, show=False):
    """
    1) Load grayscale image
    2) Median blur to denoise
    3) Slight contrast & brightness boost
    4) Adaptive threshold + morphology to build a clean mask
    5) Remove small speckles
    6) Deskew using mask, capped to ±MAX_DESKEW_ANGLE
    """
    # 1️⃣ Read
    print("📷 Reading image from:", path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")

    # 2️⃣ Denoise
    blurred = cv2.medianBlur(img, MEDIAN_BLUR_KSIZE)

    # 3️⃣ Contrast & brightness
    contrasted = cv2.convertScaleAbs(blurred, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)

    # 4️⃣ Adaptive threshold to binary (invert: text white on black)
    binarized = cv2.adaptiveThreshold(
        contrasted, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=ADAPT_BLOCK,
        C=ADAPT_C
    )
    # 5️⃣ Morphology: open then close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    opened = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened,    cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6️⃣ Remove tiny blobs
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed)
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_SPECKLE:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    # 7️⃣ Deskew based on mask
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        deskewed = contrasted
    else:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        # cap the deskew angle
        angle = max(min(angle, MAX_DESKEW_ANGLE), -MAX_DESKEW_ANGLE)
        h, w = mask.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        # apply to the contrasted image for final output
        deskewed = cv2.warpAffine(
            contrasted, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    # 8️⃣ Visualize
    if show:
        cv2.imshow("Original", img)
        cv2.imshow("Brightened", contrasted)
        cv2.imshow("Deskewed", deskewed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return deskewed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input image path")
    parser.add_argument("--output", default="clean.png", help="output image path")
    parser.add_argument("--show", action="store_true",   help="display stages")
    args = parser.parse_args()

    clean = preprocess_image(args.input, show=args.show)
    cv2.imwrite(args.output, clean)
    print(f"🔹 Saved cleaned image to {args.output}")
