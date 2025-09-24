#!/usr/bin/env python3
# src/recognize_words.py

import sys
import os
from PIL import Image
import torch
import numpy as np
import tensorflow as tf

# Ensure project root on PYTHONPATH
sys.path.insert(0, os.getcwd())

from src.model_ctc_resnet import ResNetCRNN, VOCAB, idx2char

# Merged segmentation utility
from src.merge_segments  import merge_segments
# Preprocessing & padding
from src.utils_image     import normalize_and_pad
# Mixed inference helper for words only
from src.word_inference  import crop_and_recognize as word_crop_and_recognize
# Letter CNN and labels, plus predict helper
from inference import predict_letter, LetterCNN, CLASSES

import torch.nn.functional as F

def ctc_greedy_decode(output):
    out = output.permute(1, 0, 2)  # [B, T, C]
    pred = out.softmax(2).argmax(2)  # [B, T]
    results = []
    blank_idx = len(idx2char)
    for p in pred:
        prev = -1
        decoded = []
        for c in p:
            c = c.item()
            if c != prev and c != blank_idx:
                decoded.append(idx2char.get(c, "?"))
            prev = c
        results.append("".join(decoded))
    return results


# Paths & constants
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
WORD_MODEL_PATH   = "word_model.pth"
LETTER_MODEL_PATH = "models/lettercnn_best.pth"
CLASSIFIER_PATH   = "letter_word_classifier.h5"
MIN_WORD_AREA     = 500
WIDTH_THRESHOLD   = 120
MERGE_GAP         = 70

# Global models
word_model       = None
letter_model     = None
classifier_model = None


def load_models():
    global word_model, letter_model, classifier_model

    # Load CTC word model
    word_model = ResNetCRNN(num_classes=len(VOCAB)).to(DEVICE)
    checkpoint = torch.load(WORD_MODEL_PATH, map_location=DEVICE)
    word_model.load_state_dict(checkpoint['model_state'])
    word_model.eval()


    # Load letter CNN
    letter_model = LetterCNN().to(DEVICE)
    letter_model.load_state_dict(
        torch.load(LETTER_MODEL_PATH, map_location=DEVICE)
    )
    letter_model.eval()

    # Load binary (letter vs word) classifier
    classifier_model = tf.keras.models.load_model(CLASSIFIER_PATH)


def classify_segment(patch):
    """Return 'letter' or 'word' based on classifier."""
    img = Image.fromarray(patch).convert('L').resize((32,32))
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1,32,32,1)
    pred = classifier_model.predict(arr, verbose=0)[0][0]
    return 'letter' if pred >= 0.5 else 'word'


def main_crop_and_recognize(cleaned_img, bboxes):
    """Handle mixed letter/word inference for each bbox."""
    results = []
    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        patch = cleaned_img[y1:y2, x1:x2]
        if patch.size == 0:
            results.append('')
            continue

        width = x2 - x1
        if width <= WIDTH_THRESHOLD:
            choice = classify_segment(patch)
            if choice == 'letter':
                # Single-letter prediction
                let = predict_letter(letter_model, Image.fromarray(patch))
                results.append(let)
                continue
            # else fall through to word path

        # Word prediction
        # Word prediction
        t = normalize_and_pad(patch, target_size=(32,128)).to(DEVICE)
        with torch.no_grad():
            logits = word_model(t.unsqueeze(0))
        word = ctc_greedy_decode(logits)[0].strip()
        results.append(word)

    return results


def main(image_path):
    # 1️⃣ Segment & merge
    _, merged_lines, cleaned_img = merge_segments(
        image_path,
        max_gap=MERGE_GAP,
        show=False
    )

    # 2️⃣ Filter tiny boxes
    filtered = []
    for line in merged_lines:
        good = [b for b in line if (b[2]-b[0])*(b[3]-b[1]) >= MIN_WORD_AREA]
        if good:
            filtered.append(good)

    print(f"\n🗂️  Blocks detected: {len(filtered)}\n")

    # 3️⃣ Load models
    load_models()

    # 4️⃣ Recognize and print
    full_text = []
    print("📜 Recognized text:")
    for i, line in enumerate(filtered, 1):
        words = main_crop_and_recognize(cleaned_img, line)
        text = ' '.join(w for w in words if w)
        full_text.append(text)
        print(f" Line {i} ({len(words)}): {text}")

    # 5️⃣ Summary
    total = sum(len(txt.split()) for txt in full_text)
    print(f"\n🔢 Total recognized: {total}\n")
    print("🧾 Final full prediction:\n", "\n".join(full_text))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 -m src.recognize_words path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
