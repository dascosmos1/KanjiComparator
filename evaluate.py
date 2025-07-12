import torch
from torchvision import transforms
from PIL import Image
import argparse
from models.siamese_cnn import KanjiSiameseNetwork
from utils.helpers import similarity_score
from config import *

# --- Load + preprocess image ---
def load_image(path, transform):
    image = Image.open(path).convert('L')
    return transform(image).unsqueeze(0).to(DEVICE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, required=True, help="Reference kanji image path")
    parser.add_argument('--img2', type=str, required=True, help="User-drawn kanji image path")
    parser.add_argument('--model', type=str, default='kanji_model.pt', help="Trained model path")
    args = parser.parse_args()

    # Transforms (same as training!)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load model
    model = KanjiSiameseNetwork().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    # Load and transform images
    img1 = load_image(args.img1, transform)
    img2 = load_image(args.img2, transform)

    # Get embeddings
    with torch.no_grad():
        out1, out2 = model(img1, img2)
        score = similarity_score(out1, out2)

    print(f"\n🈶 Similarity Score: {score:.3f} ({'ACCEPTED ✅' if score > 0.75 else 'TRY AGAIN ❌'})")

if __name__ == "__main__":
    main()