# datasets.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split

# 1) Discover classes from folders under data/
DATA_ROOT = "data/letters"
CLASSES   = sorted([d.name for d in Path(DATA_ROOT).iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASSES)

class LetterDataset(Dataset):
    def __init__(self, root_dir=DATA_ROOT, transform=None):
        self.transform = transform
        self.samples = []
        for cls in CLASSES:
            cls_idx = CLASSES.index(cls)
            for img_path in Path(root_dir).joinpath(cls).glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_splits(val_frac=0.2, transform=None):
    full = LetterDataset(transform=transform)
    n_val = int(len(full)*val_frac)
    n_tr  = len(full) - n_val
    return random_split(full, [n_tr, n_val])
