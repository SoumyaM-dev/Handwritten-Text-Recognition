# src/word_dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
from src.model_ctc_resnet import VOCAB, char2idx
from PIL import UnidentifiedImageError

# allow loading truncated images in a tolerant way
ImageFile.LOAD_TRUNCATED_IMAGES = True

class WordDataset(Dataset):
    def __init__(self, root_dir, label_file="words.txt", transform=None):
        self.transform = transform
        root = Path(root_dir)
        img_root = root / "words"
        label_path = root / label_file

        labmap = {}
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # skip empty and comment lines
                parts = line.split()
                # typical IAM line: <img_id> <ok|err> <transcription>
                if len(parts) < 2:
                    continue
                if parts[1] != "ok":
                    continue
                key = parts[0]
                transcription = parts[-1]
                labmap[key] = transcription.lower()

        self.labmap = labmap
        all_images = list(img_root.rglob("*.png"))
        self.imgs = []
        for p in all_images:
            if p.stem in labmap:
                try:
                    _ = Image.open(p).convert("L")
                    self.imgs.append(p)
                except UnidentifiedImageError:
                    print(f"⚠️ Skipping unreadable image: {p}")

        if not self.imgs:
            raise FileNotFoundError(f"No labeled images found under {img_root}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        try:
            img = Image.open(img_path).convert("L")  # grayscale
        except Exception as e:
            # fallback: return an empty tensor + empty target so collate will drop it
            print(f"⚠️ Couldn't open image {img_path}: {e}")
            empty_img = torch.zeros((1, 32, 128), dtype=torch.float)  # approximate shape
            return empty_img, torch.tensor([], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        transcription = self.labmap[img_path.stem]
        # build indices, skip unknown chars
        indices = [char2idx[c] for c in transcription if c in char2idx]
        target = torch.tensor(indices, dtype=torch.long)
        return img, target
