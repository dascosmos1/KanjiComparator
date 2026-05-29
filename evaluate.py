import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import argparse
import numpy as np
import os
from models.siamese_cnn import KanjiSiameseNetwork
from utils.helpers import similarity_score
from config import *


def load_image(path, transform, device, debug=False, label=''):
    image = Image.open(path).convert('L')
    mean_intensity = sum(image.getdata()) / (image.width * image.height)
    inverted = mean_intensity <= 127
    if inverted:
        image = ImageOps.invert(image)

    if debug:
        debug_path = f'debug_{label}.png'
        image.resize(IMG_SIZE).save(debug_path)
        print(f'  [{label}] mean intensity: {mean_intensity:.1f} | '
              f'{"inverted" if inverted else "kept as-is"} | '
              f'saved → {debug_path}')

    return transform(image).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1',  type=str, required=True,  help='Reference kanji image')
    parser.add_argument('--img2',  type=str, required=True,  help='Handwritten kanji image')
    parser.add_argument('--model', type=str, default='mlmodel/kanji_model.pt')
    parser.add_argument('--debug', action='store_true',
                        help='Save preprocessed images to debug_ref.png / debug_hw.png')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    model = KanjiSiameseNetwork().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    if args.debug:
        print('\nPreprocessing debug:')
    img1 = load_image(args.img1, transform, device, debug=args.debug, label='ref')
    img2 = load_image(args.img2, transform, device, debug=args.debug, label='hw')

    with torch.no_grad():
        out1, out2 = model(img1, img2)
        distance = F.pairwise_distance(out1, out2).item()
        score = similarity_score(out1, out2)

    THRESHOLD = 0.6123  # Youden's J optimal from ROC curve (AUC=0.983, TPR=96.4%, FPR=7.1%)

    print(f'\nDistance:         {distance:.4f}')
    print(f'  Dist+ mean (same kanji, val set):  ≈ 0.27  → score ≈ 0.787')
    print(f'  Dist- mean (diff kanji, val set):  ≈ 1.35  → score ≈ 0.425')
    print(f'  Decision boundary:                 ≈ 0.81  → score ≈ 0.553')
    print(f'\nSimilarity Score: {score:.3f}  (threshold > {THRESHOLD})')
    print(f'Result:           {"ACCEPTED ✅" if score > THRESHOLD else "TRY AGAIN ❌"}')

    if args.debug:
        print('\nOpen debug_ref.png and debug_hw.png to verify the model sees')
        print('black strokes on a white background for both images.')


if __name__ == "__main__":
    main()
