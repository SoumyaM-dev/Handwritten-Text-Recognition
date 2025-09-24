# model_letter.py
import torch.nn as nn
import torch
import torchvision.transforms as T
from PIL import Image

LETTER_VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

class LetterModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def letter_preprocess(img):
    transform = T.Compose([
        T.Grayscale(1),
        T.Resize((28, 28)),
        T.ToTensor()
    ])
    return transform(img)
