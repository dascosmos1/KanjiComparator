import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class KanjiSiameseNetwork(nn.Module):
    def __init__(self):
        super(KanjiSiameseNetwork, self).__init__()

        # Shared encoder (can swap with resnet18, or custom CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),  # 64x64 input → 14x14 after two conv+pool stages
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        embedding = self.cnn(x)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):  # L2-normalized embeddings live in [0,2] distance range
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)
        loss = torch.mean(
            label * torch.pow(distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )
        return loss

