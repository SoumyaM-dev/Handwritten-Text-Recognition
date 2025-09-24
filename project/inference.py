import torch
import torch.nn.functional as F
import tensorflow as tf
from PIL import Image
from torchvision import transforms
import numpy as np

from datasets import CLASSES  # adjust if in different module
from train import LetterCNN, IMG_H, IMG_W, DEVICE  # adjust if train module file path differs

from src.model_ctc_resnet import ResNetCRNN, VOCAB, idx2char

from src.segment import segment_image
from src.preprocess import preprocess_image
from src.utils_image import normalize_and_pad

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


def load_letter_model(checkpoint_path="models/lettercnn_best.pth"):
    model = LetterCNN().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def predict_letter(model, img_input):
    """
    Predict a single letter from a PIL.Image or file path.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Load image
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("L")
    else:
        img = img_input.convert("L")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        idx = logits.argmax(dim=1).item()
    return CLASSES[idx]

# ========== WORD MODEL LOADING & PREDICTION ==========

WORD_MODEL_PATH   = "word_model.pth"
LETTER_MODEL_PATH = "models/lettercnn_best.pth"
CLASSIFIER_PATH   = "letter_word_classifier.h5"
WIDTH_THRESHOLD   = 120
MIN_WORD_AREA     = 500


def load_word_model(model_path=WORD_MODEL_PATH):
    model = ResNetCRNN(num_classes=len(VOCAB)).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Handle both dict and raw state_dict
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model



def predict_word(model, img_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        log_probs = F.log_softmax(logits, dim=2)
    decoded = ctc_greedy_decode(logits)[0].strip()

    return decoded 


# ========== CLASSIFIER LOADING ==========

def load_classifier(model_path=CLASSIFIER_PATH):
    return tf.keras.models.load_model(model_path)

# ========== FULL TEXT PREDICTION PIPELINE ==========

def predict_text(img_path):
    """
    Segment the image into word/letter regions, route each crop through
    the letter or word model via a binary classifier, and return the
    concatenated text.
    """
    # 1) Segment
    lines, cleaned = segment_image(img_path, preprocess=True, show=False)

    # 2) Filter small noise
    lines = [
        [box for box in line if (box[2]-box[0])*(box[3]-box[1]) >= MIN_WORD_AREA]
        for line in lines
    ]

    # 3) Load models
    word_model   = load_word_model()
    letter_model= load_letter_model(LETTER_MODEL_PATH)
    classifier   = load_classifier(CLASSIFIER_PATH)

    # 4) Recognize
    results = []
    for line in lines:
        words = []
        for (x1, y1, x2, y2) in line:
            patch = cleaned[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            width = x2 - x1
            use_letter = False
            if width <= WIDTH_THRESHOLD:
                # Classify region
                arr_img = Image.fromarray(patch).convert("L").resize((32,32))
                arr = (np.array(arr_img)/255.0).reshape(1,32,32,1)
                score = classifier.predict(arr, verbose=0)[0][0]
                if score >= 0.5:
                    use_letter = True
            if use_letter:
                letter = predict_letter(letter_model, Image.fromarray(patch))
                words.append(letter)
            else:
                # Word CTC path
                                # Word CTC path
                t = normalize_and_pad(patch, target_size=(32,128)).to(DEVICE)
                with torch.no_grad():
                    logits = word_model(t.unsqueeze(0))
                decoded = ctc_greedy_decode(logits)[0].strip()
                words.append(decoded)

        if words:
            results.append(" ".join(words))

    # 5) Return full text
    return "\n".join(results)

def predict_words(image_path, gap=70, min_area=500):
    """
    Run the same src/recognize_words pipeline but return the text
    instead of printing it.
    """
    from src.merge_segments import merge_segments
    from src.recognize_words import load_models, main_crop_and_recognize, MIN_WORD_AREA, WIDTH_THRESHOLD

    # 1) segment + merge
    _, merged_lines, cleaned = merge_segments(
        image_path,
        max_gap=gap,
        show=False
    )

    # 2) filter out tiny boxes
    filtered = []
    for line in merged_lines:
        good = [b for b in line if (b[2]-b[0])*(b[3]-b[1]) >= min_area]
        if good:
            filtered.append(good)

    # 3) load all models (word/letter/classifier)
    load_models()

    # 4) recognize every box, collect lines
    full = []
    for line in filtered:
        words = main_crop_and_recognize(cleaned, line)
        full.append(" ".join(w for w in words if w.strip()))

    # 5) return a single string with newlines
    return "\n".join(full)

