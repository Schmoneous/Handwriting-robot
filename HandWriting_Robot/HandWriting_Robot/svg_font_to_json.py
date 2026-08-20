#!/usr/bin/env python3
"""
svg_font_to_json.py
=====================
Converts an SVG font file (the format used by the free single-line
plotter fonts like EMS Casual Hand, EMS Tech, EMS Nixish, etc.) into the
same JSON glyph format used by custom_font.py / letter_designer.html --
so it drops straight into handwriting_bot.py's font_variants pool.

These SVG fonts are already single-stroke (designed for pen plotters),
so this is a fairly clean conversion: each glyph's path data is split
into subpaths (each "M"/"moveto" starts a new subpath = a new pen
stroke), and curves are flattened into short line segments.

Usage:
    python3 svg_font_to_json.py EMS_Casual_Hand.svg -o casual_hand.json

Then use it in the pipeline:
    python3 handwriting_ebb.py "text" -o letter.ebb \\
        --font-variants-json casual_hand.json --port ... --send
    (or load it as a custom_font_path / merge into font_variants -- see
    the printed instructions at the end of this script's output)
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET


# --------------------------------------------------------------------------
# Minimal SVG path 'd' parser -> list of flattened subpaths (polylines)
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"([MLHVCSQTAZmlhvcsqtaz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _tokenize(d):
    tokens = []
    for cmd, num in _TOKEN_RE.findall(d):
        if cmd:
            tokens.append(cmd)
        elif num:
            tokens.append(float(num))
    return tokens


def _bezier_cubic(p0, p1, p2, p3, n=10):
    pts = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        x = (mt**3) * p0[0] + 3 * (mt**2) * t * p1[0] + 3 * mt * (t**2) * p2[0] + (t**3) * p3[0]
        y = (mt**3) * p0[1] + 3 * (mt**2) * t * p1[1] + 3 * mt * (t**2) * p2[1] + (t**3) * p3[1]
        pts.append((x, y))
    return pts


def _bezier_quad(p0, p1, p2, n=8):
    pts = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        x = (mt**2) * p0[0] + 2 * mt * t * p1[0] + (t**2) * p2[0]
        y = (mt**2) * p0[1] + 2 * mt * t * p1[1] + (t**2) * p2[1]
        pts.append((x, y))
    return pts


def parse_path(d):
    """Parse an SVG path 'd' string into a list of subpaths, each a list
    of (x, y) points with curves flattened to line segments."""
    tokens = _tokenize(d)
    subpaths = []
    cur = None
    pos = (0.0, 0.0)
    start = (0.0, 0.0)
    last_cmd = None
    last_ctrl = None  # for smooth curve reflection (S/T)
    i = 0

    def push_point(pt):
        nonlocal cur
        if cur is None:
            cur = []
            subpaths.append(cur)
        cur.append(pt)

    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str):
            cmd = tok
            i += 1
        else:
            # repeat last command (implicit repetition), don't advance i
            cmd = last_cmd

        is_rel = cmd.islower()
        C = cmd.upper()

        if C == "M":
            x, y = tokens[i], tokens[i + 1]
            i += 2
            if is_rel:
                x, y = pos[0] + x, pos[1] + y
            pos = (x, y)
            start = pos
            cur = None  # force new subpath
            push_point(pos)
            last_cmd = "l" if is_rel else "L"  # subsequent bare coords act as lineto

        elif C == "L":
            x, y = tokens[i], tokens[i + 1]
            i += 2
            if is_rel:
                x, y = pos[0] + x, pos[1] + y
            pos = (x, y)
            push_point(pos)
            last_cmd = cmd

        elif C == "H":
            x = tokens[i]; i += 1
            x = pos[0] + x if is_rel else x
            pos = (x, pos[1])
            push_point(pos)
            last_cmd = cmd

        elif C == "V":
            y = tokens[i]; i += 1
            y = pos[1] + y if is_rel else y
            pos = (pos[0], y)
            push_point(pos)
            last_cmd = cmd

        elif C == "C":
            x1, y1, x2, y2, x, y = tokens[i:i + 6]
            i += 6
            if is_rel:
                x1, y1 = pos[0] + x1, pos[1] + y1
                x2, y2 = pos[0] + x2, pos[1] + y2
                x, y = pos[0] + x, pos[1] + y
            for pt in _bezier_cubic(pos, (x1, y1), (x2, y2), (x, y)):
                push_point(pt)
            pos = (x, y)
            last_ctrl = (x2, y2)
            last_cmd = cmd

        elif C == "S":
            x2, y2, x, y = tokens[i:i + 4]
            i += 4
            if is_rel:
                x2, y2 = pos[0] + x2, pos[1] + y2
                x, y = pos[0] + x, pos[1] + y
            if last_ctrl and last_cmd.upper() in ("C", "S"):
                x1, y1 = 2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]
            else:
                x1, y1 = pos
            for pt in _bezier_cubic(pos, (x1, y1), (x2, y2), (x, y)):
                push_point(pt)
            pos = (x, y)
            last_ctrl = (x2, y2)
            last_cmd = cmd

        elif C == "Q":
            x1, y1, x, y = tokens[i:i + 4]
            i += 4
            if is_rel:
                x1, y1 = pos[0] + x1, pos[1] + y1
                x, y = pos[0] + x, pos[1] + y
            for pt in _bezier_quad(pos, (x1, y1), (x, y)):
                push_point(pt)
            pos = (x, y)
            last_ctrl = (x1, y1)
            last_cmd = cmd

        elif C == "T":
            x, y = tokens[i], tokens[i + 1]
            i += 2
            if is_rel:
                x, y = pos[0] + x, pos[1] + y
            if last_ctrl and last_cmd.upper() in ("Q", "T"):
                x1, y1 = 2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]
            else:
                x1, y1 = pos
            for pt in _bezier_quad(pos, (x1, y1), (x, y)):
                push_point(pt)
            pos = (x, y)
            last_ctrl = (x1, y1)
            last_cmd = cmd

        elif C == "A":
            # Arc -- approximate with a straight line to the endpoint.
            # (Rare in single-line plotter fonts; good enough fallback.)
            rx, ry, rot, large, sweep, x, y = tokens[i:i + 7]
            i += 7
            if is_rel:
                x, y = pos[0] + x, pos[1] + y
            push_point((x, y))
            pos = (x, y)
            last_cmd = cmd

        elif C == "Z":
            if cur and cur[0] != pos:
                push_point(start)
            pos = start
            cur = None
            last_cmd = cmd

        else:
            # Unknown command -- skip its expected arg count conservatively
            i += 1

    return [s for s in subpaths if len(s) >= 1]


# --------------------------------------------------------------------------
# SVG font -> our JSON glyph format
# --------------------------------------------------------------------------

def convert_svg_font(svg_path, target_em_height=21.0, target_cap_line=-12.0, target_base_line=9.0):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    font_el = root.find(f".//{ns}font")
    if font_el is None:
        raise ValueError("No <font> element found -- is this a valid SVG font file?")

    face_el = font_el.find(f"{ns}font-face")
    units_per_em = float(face_el.get("units-per-em", 1000)) if face_el is not None else 1000.0
    ascent = float(face_el.get("ascent", units_per_em * 0.8)) if face_el is not None else units_per_em * 0.8

    default_adv_x = float(font_el.get("horiz-adv-x", units_per_em * 0.6))

    # Scale factor: map the font's ascent (cap-height-ish reference) to our
    # target em height, so it comes out the same visual size as Hershey fonts.
    scale = target_em_height / ascent

    def transform(pt):
        x, y = pt
        out_x = x * scale
        out_y = target_base_line - y * scale
        return [round(out_x, 3), round(out_y, 3)]

    glyphs = {}
    skipped = []
    for glyph_el in font_el.findall(f"{ns}glyph"):
        unicode_attr = glyph_el.get("unicode")
        if not unicode_attr or len(unicode_attr) != 1:
            continue  # skip ligatures / multi-char glyphs / unnamed glyphs
        d = glyph_el.get("d")
        if not d:
            continue  # e.g. space character with no path

        adv_x = float(glyph_el.get("horiz-adv-x", default_adv_x))
        try:
            subpaths = parse_path(d)
        except Exception as e:
            skipped.append((unicode_attr, str(e)))
            continue

        strokes = [[transform(pt) for pt in sub] for sub in subpaths]
        strokes = [s for s in strokes if len(s) >= 1]
        if not strokes:
            continue

        glyphs.setdefault(unicode_attr, []).append({
            "width": round(adv_x * scale, 3),
            "strokes": strokes,
        })

    return glyphs, skipped


def main():
    p = argparse.ArgumentParser(description="Convert an SVG font to the custom_font.py JSON format.")
    p.add_argument("svg_path", help="Path to the SVG font file (e.g. EMS Casual Hand.svg).")
    p.add_argument("-o", "--output", default="converted_font.json")
    args = p.parse_args()

    glyphs, skipped = convert_svg_font(args.svg_path)

    with open(args.output, "w") as f:
        json.dump({"glyphs": glyphs}, f, indent=2)

    print(f"[info] converted {len(glyphs)} letters -> {args.output}")
    if skipped:
        print(f"[warn] {len(skipped)} glyph(s) failed to parse and were skipped: "
              f"{[s[0] for s in skipped]}")
    print()
    print("To use this font in the handwriting pipeline, load it as a custom font:")
    print(f'    cfg["custom_font_path"] = "{args.output}"')
    print(f'    cfg["custom_font_exclusive"] = True   # use ONLY this font, no Hershey mixing')
    print(f"or from the command line:")
    print(f"    python3 handwriting_ebb.py \"text\" -o letter.ebb "
          f"--custom-font {args.output} --custom-font-exclusive --port ... --send")


if __name__ == "__main__":
    main()