import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

CLASSIFIER_MODEL_PATH = "letter_word_classifier.h5"
IMG_SIZE = (32, 32)

# Load the model once
model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)

def preprocess_patch(pil_img):
    """Pad, resize, normalize."""
    img = ImageOps.grayscale(pil_img)
    w, h = img.size
    max_side = max(w, h)
    pad = (
        (max_side - w) // 2,
        (max_side - h) // 2
    )
    img = ImageOps.expand(img, (pad[0], pad[1], max_side - w - pad[0], max_side - h - pad[1]), fill=255)
    img = img.resize(IMG_SIZE)
    img = np.array(img).astype('float32') / 255.0
    return img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

def is_letter(pil_patch):
    """Returns True if the patch is a letter, False if word."""
    img = preprocess_patch(pil_patch)
    pred = model.predict(img, verbose=0)[0][0]
    return pred < 0.5  # class 0 = letter, class 1 = word
