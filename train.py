import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from config import *
from data.kanjis_dataset import KanjiPairsDataset
from models.siamese_cnn import KanjiSiameseNetwork, ContrastiveLoss

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
print("load dataset")
# Load full dataset
full_dataset = KanjiPairsDataset(DATA_PATH, transform=transform)

# Split into 70% train, 30% validation
train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
print("create model")
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
model = KanjiSiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

num_epochs = EPOCHS
print("train")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), 'mlmodel/kanji_model.pt')
