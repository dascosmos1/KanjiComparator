import io
import os
import re
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

# KanjiVG SVGs contain a `<g id="kvg:StrokeNumbers_XXXXX"> ... </g>` block with
# the stroke-order digits. Drop it so the rendered PNG only shows the strokes.
STROKE_NUMBERS_RE = re.compile(
    rb'<g\s+id="kvg:StrokeNumbers_[^"]+".*?</g>',
    flags=re.DOTALL,
)


def strip_stroke_numbers(svg_bytes):
    return STROKE_NUMBERS_RE.sub(b"", svg_bytes)


def svg_to_png(svg_path, png_path):
    os.makedirs(png_path, exist_ok=True)
    for file_name in os.listdir(svg_path):
        if not file_name.endswith(".svg"):
            continue

        kanji_file = os.path.join(svg_path, file_name)
        with open(kanji_file, "rb") as f:
            svg_bytes = strip_stroke_numbers(f.read())

        drawing = svg2rlg(io.BytesIO(svg_bytes))

        output_file = os.path.join(png_path, file_name.replace(".svg", ".png"))
        renderPM.drawToFile(drawing, output_file, fmt="PNG")


if __name__ == "__main__":
    from config import KANJI_REFERENCE_DIR
    svg_to_png("kanji", KANJI_REFERENCE_DIR)