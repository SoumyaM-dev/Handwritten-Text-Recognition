import torchvision.transforms as T
from src.word_dataset import WordDataset

transform = T.Compose([
    T.Resize((32, 128)),
    T.ToTensor(),
    T.Normalize(0.5, 0.5),
])

dataset = WordDataset(
    root_dir="data/words/iam_words",
    transform=transform
)

img, label_tensor = dataset[0]
print("Image shape:", img.shape)
print("Label indices:", label_tensor.tolist())
