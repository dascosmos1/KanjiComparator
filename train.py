import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from config import *
from data.kanjis_dataset import KanjiPairsDataset
from models.siamese_cnn import KanjiSiameseNetwork, ContrastiveLoss
from utils.visualize import (
    plot_loss_curves,
    plot_distance_distributions,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_accuracy_curve,
)

THRESHOLD = 1.0  # distance < threshold → predicted same kanji

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if __name__ == '__main__':
    print("load dataset")
    full_dataset = KanjiPairsDataset(DATA_PATH, transform=train_transform)

    # Val dataset gets a separate instance with no augmentation.
    val_dataset_full = KanjiPairsDataset(DATA_PATH, transform=val_transform)
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, _ = random_split(full_dataset, [train_size, val_size])
    _, val_dataset = random_split(val_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

    print("create model")
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    model = KanjiSiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    train_losses, val_losses, val_accuracies = [], [], []

    # Keep the final epoch's distances for ROC / confusion matrix
    last_epoch_distances, last_epoch_labels = [], []
    last_pos_dists, last_neg_dists = [], []

    print("train")
    for epoch in range(EPOCHS):
        # --- Training ---
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
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        epoch_distances, epoch_labels = [], []
        epoch_pos_dists, epoch_neg_dists = [], []

        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)
                val_loss += loss.item()

                dists = F.pairwise_distance(out1, out2).cpu().numpy()
                labs = label.cpu().numpy()
                epoch_distances.extend(dists)
                epoch_labels.extend(labs)
                epoch_pos_dists.extend(dists[labs == 1].tolist())
                epoch_neg_dists.extend(dists[labs == 0].tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        preds = (np.array(epoch_distances) < THRESHOLD).astype(int)
        accuracy = (preds == np.array(epoch_labels)).mean()
        val_accuracies.append(accuracy)

        mean_pos = np.mean(epoch_pos_dists) if epoch_pos_dists else float('nan')
        mean_neg = np.mean(epoch_neg_dists) if epoch_neg_dists else float('nan')

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {accuracy:.3f} | "
            f"Dist+ {mean_pos:.3f} | Dist- {mean_neg:.3f}"
        )

        scheduler.step(avg_val_loss)

        # Save distance plot every 5 epochs and on the final epoch
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            plot_distance_distributions(epoch_pos_dists, epoch_neg_dists, epoch + 1, THRESHOLD)

        last_epoch_distances = epoch_distances
        last_epoch_labels = epoch_labels
        last_pos_dists = epoch_pos_dists
        last_neg_dists = epoch_neg_dists

    # --- Save model ---
    os.makedirs('mlmodel', exist_ok=True)
    torch.save(model.state_dict(), 'mlmodel/kanji_model.pt')

    # --- Generate plots ---
    print("\ngenerating plots → plots/")
    plot_loss_curves(train_losses, val_losses)
    plot_accuracy_curve(val_accuracies)
    plot_roc_curve(last_epoch_labels, last_epoch_distances)
    plot_confusion_matrix(last_epoch_labels, last_epoch_distances, THRESHOLD)
    print("done — plots saved to plots/")
