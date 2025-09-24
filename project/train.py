# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import get_splits, CLASSES

# 1) Hyperparams"""  """
IMG_H, IMG_W   = 32, 32
BATCH_SIZE     = 64
LR             = 1e-3
EPOCHS         = 10
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES    = len(CLASSES)
os.makedirs("models", exist_ok=True)



# 3) Model
class LetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*(IMG_H//4)*(IMG_W//4), 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x)

if __name__ == "__main__":
# 2) Transforms & DataLoaders
    transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds, val_ds = get_splits(transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = LetterCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4) Train & validate
    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        # — Train
        model.train()
        running_loss = 0.0; correct=0; total=0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item()*imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss/total
        train_acc  = correct/total

        # — Validate
        model.eval()
        v_loss=0.0; v_corr=0; v_tot=0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                v_loss += criterion(outputs, labels).item()*imgs.size(0)
                preds = outputs.argmax(dim=1)
                v_corr += (preds==labels).sum().item()
                v_tot += labels.size(0)
        val_loss = v_loss/v_tot
        val_acc  = v_corr/v_tot

        print(f"Epoch {epoch}/{EPOCHS}  "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}  |  "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Save best
        if val_acc>best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"models/lettercnn_best.pth")
            print("→ Saved best model")
