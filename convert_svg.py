import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


def svg_to_png(svg_path, png_path):
    files_list = os.listdir(svg_path)

    for file_name in files_list:
        # Read the SVG file
        kanji_file = os.path.join(svg_path,file_name)
        drawing = svg2rlg(kanji_file)

        output_file = os.path.join(png_path, file_name.replace(".svg", ".png"))
        # Convert to PNG
        renderPM.drawToFile(drawing, output_file, fmt="PNG")


svg_to_png("/Users/david/PycharmProjects/KanjiComparator/kanji","/Users/david/PycharmProjects/KanjiComparator/kanji-dataset/referenceKanji")