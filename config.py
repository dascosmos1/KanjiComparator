import os

DATA_PATH = 'kanji-dataset'
KANJI_REFERENCE_DIR  = os.path.join(DATA_PATH, 'referenceKanji')
KANJI_HANDWRITING_DIR = os.path.join(DATA_PATH, 'handwritingKanji')
KANJI_METADATA_CSV   = os.path.join(DATA_PATH, 'handwritingKanji', 'meta.csv')

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 25
IMG_SIZE = (64, 64)