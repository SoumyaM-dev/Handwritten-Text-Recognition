# train_words.py
import sys
sys.path.append("src")

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.model_ctc_resnet import ResNetCRNN, idx2char, VOCAB  # new model exports idx2char, VOCAB
from src.word_dataset import WordDataset
from torch import amp
from collections import defaultdict

torch.backends.mkldnn.enabled = False


# ----------------- HELPERS -----------------
def levenshtein(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]

def ctc_greedy_decode(output, idx2char_local):
    out = output.permute(1, 0, 2)  # [B, T, C]
    pred = out.softmax(2).argmax(2)  # [B, T]
    results = []
    blank_idx = len(idx2char_local)
    for p in pred:
        prev = -1
        decoded = []
        for c in p:
            c = c.item()
            if c != prev and c != blank_idx:
                decoded.append(idx2char_local.get(c, "?"))
            prev = c
        results.append("".join(decoded))
    return results

def cer(pred, gt):
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return levenshtein(pred, gt) / len(gt)

# ----------------- CONFIG -----------------
use_cuda = torch.cuda.is_available()
DEVICE = "cuda" if use_cuda else "cpu"
device_type = "cuda" if use_cuda else "cpu"

BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-4
IMG_HEIGHT = 32
IMG_WIDTH = 128

# ----------------- TRANSFORMS -----------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(3, expand=False, fill=0),
    transforms.RandomAffine(0, translate=(0.05, 0.05), shear=2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------- DATA -----------------
full_dataset = WordDataset(
    root_dir="data/words/iam_words",
    label_file="words.txt",
    transform=None
)

val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds, transform):
        self.base_ds = base_ds
        self.transform = transform
    def __len__(self): return len(self.base_ds)
    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        if self.transform and hasattr(self.transform, "__call__"):
            img = self.transform(img)
        return img, label

train_ds = TransformWrapper(train_subset, train_transform)
val_ds   = TransformWrapper(val_subset, val_transform)

def safe_collate(batch):
    batch = [(img, tgt) for img, tgt in batch if (isinstance(tgt, torch.Tensor) and tgt.numel() > 0)]
    if not batch:
        return [], []
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=safe_collate, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=safe_collate, num_workers=2)

# ----------------- MODEL / LOSS / OPT -----------------
model = ResNetCRNN(num_classes=len(VOCAB), backbone_name='resnet18', pretrained=True).to(DEVICE)
ctc_blank = len(VOCAB)
criterion = nn.CTCLoss(blank=ctc_blank, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# OneCycleLR requires steps_per_epoch
steps_per_epoch = max(1, len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR*5, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)

scaler = amp.GradScaler(enabled=use_cuda)
grad_clip = 5.0

# ----------------- TRAIN / EVAL -----------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()
    running_loss = 0.0
    count = 0
    for imgs, targets in loader:
        if len(imgs) == 0:
            continue
        imgs = torch.stack(imgs).to(DEVICE)
        targets = [t.to(DEVICE) for t in targets]
        with amp.autocast(device_type=device_type):
            logits = model(imgs)                  # [T, B, C]
            log_probs = nn.functional.log_softmax(logits, dim=2)
            T, B, C = log_probs.shape
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(DEVICE)
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long).to(DEVICE)
            targets_flat = torch.cat(targets).to(DEVICE)
            loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        count += 1
    return running_loss / max(1, count)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    sum_cer = 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            if len(imgs) == 0:
                continue
            imgs = torch.stack(imgs).to(DEVICE)
            targets = [t.to(DEVICE) for t in targets]

            logits = model(imgs)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            T, B, C = log_probs.shape
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(DEVICE)
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long).to(DEVICE)
            targets_flat = torch.cat(targets).to(DEVICE)
            loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)

            total_loss += loss.item()
            preds = ctc_greedy_decode(logits, idx2char)
            for pred, tgt in zip(preds, targets):
                target_str = ''.join(idx2char[i.item()] for i in tgt if i.item() < len(idx2char))
                sum_cer += cer(pred, target_str)
                total += 1

    return total_loss / max(1, len(loader)), (sum_cer / max(1, total))

# ----------------- MAIN LOOP -----------------
best_cer = 1.0
save_path = "resnet_crnn_best.pth"
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler)
    val_loss, val_cer = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch}/{EPOCHS}  Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}  Val CER: {val_cer:.4f}")

    # checkpoint
    if val_cer < best_cer:
        best_cer = val_cer
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_cer': best_cer
        }, save_path)
        print("→ Saved best model (CER improved)")

print("Training finished. Best CER:", best_cer)
