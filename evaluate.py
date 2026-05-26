import torch
from torchvision import transforms
from PIL import Image, ImageOps
import argparse
from models.siamese_cnn import KanjiSiameseNetwork
from utils.helpers import similarity_score
from config import *

# --- Load + preprocess image ---
def load_image(path, transform):
    """Load grayscale and normalize polarity to black-on-white (matching training).

    Training feeds the model black-on-white for both branches: references are
    already black-on-white, and handwriting (white-on-black) is inverted in the
    dataset. At eval time the user may pass either polarity, so detect it from
    the mean intensity: a light background means the image is already
    black-on-white; a dark background means we need to invert.
    """
    image = Image.open(path).convert('L')
    # Mean > 127 ⇒ majority light pixels ⇒ already black-on-white. Keep as is.
    # Mean ≤ 127 ⇒ majority dark pixels ⇒ white-on-black handwriting. Invert.
    if sum(image.getdata()) / (image.width * image.height) <= 127:
        image = ImageOps.invert(image)
    return transform(image).unsqueeze(0).to(DEVICE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, required=True, help="Reference kanji image path (black-on-white)")
    parser.add_argument('--img2', type=str, required=True, help="User-drawn kanji image path (white-on-black handwriting)")
    parser.add_argument('--model', type=str, default='mlmodel/kanji_model.pt', help="Trained model path")
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

    # Both branches go through the same polarity-normalising loader, so the
    # model always sees black-on-white regardless of the input source.
    img1 = load_image(args.img1, transform)
    img2 = load_image(args.img2, transform)

    # Get embeddings
    with torch.no_grad():
        out1, out2 = model(img1, img2)
        score = similarity_score(out1, out2)

    print(f"\n🈶 Similarity Score: {score:.3f} ({'ACCEPTED ✅' if score > 0.75 else 'TRY AGAIN ❌'})")

if __name__ == "__main__":
    main()