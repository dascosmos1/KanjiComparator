"""
Run ROC analysis on a saved model without retraining.
Usage: python analyze_roc.py --model mlmodel/kanji_model.pt
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from config import *
from data.kanjis_dataset import KanjiPairsDataset
from models.siamese_cnn import KanjiSiameseNetwork
from utils.visualize import plot_roc_curve, plot_distance_distributions, plot_confusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlmodel/kanji_model.pt')
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading dataset...")
    full_dataset = KanjiPairsDataset(DATA_PATH, transform=transform)
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, persistent_workers=True)

    print("Loading model...")
    model = KanjiSiameseNetwork().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    all_distances, all_labels = [], []
    pos_dists, neg_dists = [], []

    print("Running inference on validation set...")
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            dists = F.pairwise_distance(out1, out2).cpu().numpy()
            labs  = label.cpu().numpy()
            all_distances.extend(dists)
            all_labels.extend(labs)
            pos_dists.extend(dists[labs == 1].tolist())
            neg_dists.extend(dists[labs == 0].tolist())

    print(f"\nVal set: {len(all_distances)} pairs")
    print(f"  Dist+ mean (same kanji):  {np.mean(pos_dists):.4f}")
    print(f"  Dist- mean (diff kanji):  {np.mean(neg_dists):.4f}")
    print(f"  Decision midpoint:        {(np.mean(pos_dists) + np.mean(neg_dists)) / 2:.4f}")

    roc = plot_roc_curve(all_labels, all_distances)
    plot_distance_distributions(pos_dists, neg_dists, epoch='final')
    plot_confusion_matrix(all_labels, all_distances, threshold=roc['opt_distance'])

    print("\n--- ROC Curve Analysis ---")
    print(f"  AUC:                        {roc['auc']:.4f}")
    print(f"  Optimal distance threshold: {roc['opt_distance']:.4f}")
    print(f"  Optimal similarity score:   {roc['opt_sim_score']:.4f}  ← use this in evaluate.py")
    print(f"  TPR at optimal point:       {roc['opt_tpr']:.4f}  (correctly accepted same-kanji pairs)")
    print(f"  FPR at optimal point:       {roc['opt_fpr']:.4f}  (incorrectly accepted different-kanji pairs)")
    print("\nPlots saved to plots/")
