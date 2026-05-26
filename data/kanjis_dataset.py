import os
import torch
import random
import csv
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from config import KANJI_REFERENCE_DIR, KANJI_HANDWRITING_DIR, KANJI_METADATA_CSV


class KanjiPairsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.positive_pairs = []   # (actual_path, handwriting_path, code)
        self.all_handwriting = []  # [(path, code), ...]
        self.handwriting_map = {}  # code -> [paths]

        kanji_dataset_dir = KANJI_REFERENCE_DIR
        handwriting_dataset_dir = KANJI_HANDWRITING_DIR
        metadata_csv = KANJI_METADATA_CSV

        with open(metadata_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = f'{str(row[""]).zfill(5)}.png'
                code = row["unicode"].lower().replace('x', '')
                img_path = os.path.join(handwriting_dataset_dir, filename)
                self.all_handwriting.append((img_path, code))
                self.handwriting_map.setdefault(code, []).append(img_path)

        for actual_filename in os.listdir(kanji_dataset_dir):
            if actual_filename.endswith(".png"):
                code = actual_filename[:-4].lower()
                actual_path = os.path.join(kanji_dataset_dir, actual_filename)
                if code in self.handwriting_map:
                    for handwriting_path in self.handwriting_map[code]:
                        self.positive_pairs.append((actual_path, handwriting_path, code))

    def __len__(self):
        return 2 * len(self.positive_pairs)

    def __getitem__(self, idx):
        n = len(self.positive_pairs)
        if idx < n:
            actual_path, hw_path, _ = self.positive_pairs[idx]
            label = 1
        else:
            actual_path, _, code = self.positive_pairs[idx - n]
            # Fresh negative every call — model sees different negatives each epoch.
            while True:
                neg_path, neg_code = self.all_handwriting[random.randrange(len(self.all_handwriting))]
                if neg_code != code:
                    break
            hw_path = neg_path
            label = 0

        img1 = Image.open(actual_path).convert('L')
        img2 = ImageOps.invert(Image.open(hw_path).convert('L'))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)
