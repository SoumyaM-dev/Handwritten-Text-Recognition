import os
from PIL import Image, ImageOps

# CONFIG
DATA_DIR = "classifier_data"
OUT_DIR = "classifier_data_resized"
IMG_SIZE = (32, 32)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def pad_to_square(img):
    """Pad the image to make it square with equal width and height."""
    w, h = img.size
    max_side = max(w, h)
    pad_w = (max_side - w) // 2
    pad_h = (max_side - h) // 2
    padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
    return ImageOps.expand(img, padding, fill=255)  # white padding

def resize_and_save_all():
    classes = ["letter", "word"]
    for cls in classes:
        src_dir = os.path.join(DATA_DIR, cls)
        dst_dir = os.path.join(OUT_DIR, cls)
        ensure_dir(dst_dir)

        images = os.listdir(src_dir)
        for img_name in images:
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(dst_dir, img_name)

            try:
                img = Image.open(src_path).convert("L")
                img = pad_to_square(img)
                img = img.resize(IMG_SIZE, resample=Image.LANCZOS)
                img.save(dst_path)
            except Exception as e:
                print(f"Failed: {src_path} — {e}")

if __name__ == "__main__":
    print("Resizing with padding...")
    resize_and_save_all()
    print(f"Done! Resized images are in: {OUT_DIR}")
