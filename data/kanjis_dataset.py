import os
import torch
import random
import csv
from PIL import Image
from torch.utils.data import Dataset


class KanjiPairsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []

        kanji_dataset_dir = "/Users/david/PycharmProjects/KanjiComparator/kanji-dataset/referenceKanji"
        handwriting_dataset_dir = "/Users/david/PycharmProjects/KanjiComparator/kanji-dataset/handwritingKanji"
        metadata_csv = "/Users/david/PycharmProjects/KanjiComparator/kanji-dataset/handwritingKanji/meta.csv"
        handwriting_map = {}  # code (hex) -> list of handwriting image paths
        all_handwriting = []  # list of all (path, code) for negative sampling
        max_digits = 0
        with open(metadata_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            rows_size = len(rows)
            for row in rows:
                base = len(str(99999))
                max_digits = max(max_digits, base)
                filename = f'{str(row['']).zfill(max_digits)}.png'
                code = row["unicode"].lower().replace('x','')  # match lowercase filenames like '0f9a8'
                img_path = os.path.join(handwriting_dataset_dir, filename)
                all_handwriting.append((img_path, code))
                handwriting_map.setdefault(code, []).append(img_path)

            # Match with actual Kanji dataset

            for actual_filename in os.listdir(kanji_dataset_dir):
                if actual_filename.endswith(".png"):
                    code = actual_filename[:-4].lower()
                    actual_path = os.path.join(kanji_dataset_dir, actual_filename)

                    if code in handwriting_map:
                        # Add matching pair(s)
                        for handwriting_path in handwriting_map[code]:
                            self.pairs.append((actual_path, handwriting_path, 1))

                        # Generate a mismatched sample (negative pair)
                        # Choose a handwriting sample with a different code
                        negative_sample = random.choice([
                            (h_path, h_code) for (h_path, h_code) in all_handwriting if h_code != code
                        ])
                        self.pairs.append((actual_path, negative_sample[0], 0))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)
