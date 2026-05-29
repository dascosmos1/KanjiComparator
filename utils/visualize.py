import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


def plot_loss_curves(train_losses, val_losses, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150)
    plt.close()


def plot_distance_distributions(pos_dists, neg_dists, epoch, threshold=1.0, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(pos_dists, bins=50, alpha=0.6, color='steelblue', label='Same kanji (positive)')
    plt.hist(neg_dists, bins=50, alpha=0.6, color='tomato', label='Different kanji (negative)')
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5,
                label=f'Decision threshold ({threshold})')
    plt.xlabel('L2 Distance')
    plt.ylabel('Count')
    plt.title(f'Embedding Distance Distribution — Epoch {epoch}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    label = f'{epoch:02d}' if isinstance(epoch, int) else epoch
    plt.savefig(os.path.join(save_dir, f'distances_epoch_{label}.png'), dpi=150)
    plt.close()


def plot_roc_curve(all_labels, all_distances, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    # Label=1 means same kanji → low distance → negate so higher score = more similar
    scores = -np.array(all_distances)
    fpr, tpr, thresholds = roc_curve(all_labels, scores)
    roc_auc = auc(fpr, tpr)

    # Optimal threshold via Youden's J (maximises TPR - FPR)
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    opt_distance  = -thresholds[opt_idx]          # convert back from negated score
    opt_fpr       = fpr[opt_idx]
    opt_tpr       = tpr[opt_idx]
    opt_sim_score = 1 / (1 + opt_distance)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.scatter([opt_fpr], [opt_tpr], color='red', zorder=5,
                label=f'Optimal threshold\ndistance={opt_distance:.3f} | score={opt_sim_score:.3f}\nTPR={opt_tpr:.3f} | FPR={opt_fpr:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Validation Set (Final Epoch)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150)
    plt.close()

    return {
        'auc':           roc_auc,
        'opt_distance':  opt_distance,
        'opt_sim_score': opt_sim_score,
        'opt_tpr':       opt_tpr,
        'opt_fpr':       opt_fpr,
    }


def plot_confusion_matrix(all_labels, all_distances, threshold=1.0, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    preds = (np.array(all_distances) < threshold).astype(int)
    cm = confusion_matrix(all_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Different', 'Same'])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix (threshold = {threshold})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()


def plot_accuracy_curve(val_accuracies, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(val_accuracies) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_accuracies, marker='o', color='seagreen', label='Val Accuracy')
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random baseline')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Validation Accuracy at Distance Threshold = 1.0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'val_accuracy.png'), dpi=150)
    plt.close()
