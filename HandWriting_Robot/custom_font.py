#!/usr/bin/env python3
"""
custom_font.py
===============
Loads hand-drawn letter variants created with letter_designer.html and
converts them into the same glyph-object interface HersheyFonts uses,
so they drop straight into handwriting_bot.py's rendering pipeline and
mix seamlessly with (or replace) the built-in Hershey fonts.

JSON format (produced by letter_designer.html):
{
  "glyphs": {
    "a": [
      {"width": 16, "strokes": [[[x,y],[x,y],...], [[x,y],...]]},   // variant 1
      {"width": 16, "strokes": [...]}                                // variant 2
    ],
    "t": [ {"width": 9, "strokes": [...]} ]
  }
}

Coordinate convention (matches Hershey's raw glyph data, same as what
HandwritingGenerator._render_line expects before its y-flip):
  - y is DOWN-positive (larger y = lower on the page)
  - baseline is at y=9, cap-height top is at y=-12 (21-unit em), matching
    the default HersheyFonts render_options -- so a custom glyph drawn to
    fill that same box will come out the same size as the built-in fonts.
  - x=0 is the left edge of the glyph; `width` is the advance width (how
    far the cursor moves after drawing this letter), same units as x/y.
"""

import json


class _CustomGlyph:
    """Mimics the small subset of HersheyFonts' glyph interface that
    HandwritingGenerator._render_line actually uses: .strokes and .char_width."""

    def __init__(self, strokes, width):
        # strokes: list of polylines, each a list of (x, y) tuples, y-down positive
        self.strokes = strokes
        self.char_width = width


def load_custom_font(path):
    """
    Returns {letter: [ _CustomGlyph, _CustomGlyph, ... ]} -- a list of
    variant glyphs per letter, ready to be mixed into HandwritingGenerator's
    per-letter random choice pool.
    """
    with open(path, "r") as f:
        data = json.load(f)

    glyphs_data = data.get("glyphs", {})
    result = {}
    for letter, variants in glyphs_data.items():
        glyph_list = []
        for variant in variants:
            strokes = [[tuple(pt) for pt in stroke] for stroke in variant["strokes"]]
            width = variant.get("width", 16)
            if strokes:  # skip empty/unfinished variants
                glyph_list.append(_CustomGlyph(strokes, width))
        if glyph_list:
            result[letter] = glyph_list

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 custom_font.py <path_to_font.json>")
        sys.exit(1)

    glyphs = load_custom_font(sys.argv[1])
    print(f"Loaded {len(glyphs)} letters with custom variants:")
    for letter, variants in sorted(glyphs.items()):
        print(f"  '{letter}': {len(variants)} variant(s)")