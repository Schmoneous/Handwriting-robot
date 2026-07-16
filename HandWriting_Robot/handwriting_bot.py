#!/usr/bin/env python3
"""
handwriting_bot.py
===================
Text -> human-like pen strokes -> GRBL G-code, for the DrawCore V1.2 board
(or any GRBL-based pen plotter).

Pipeline:
  1. Convert text to per-letter Hershey font strokes (single-line "engraving" font).
  2. Humanize:
       - per-letter random micro-variants (scale/skew/rotate jitter, so the
         same letter never looks pixel-identical twice)
       - smooth Perlin-noise jitter along every stroke (hand tremor, not white noise)
       - slow baseline drift + slight per-word slant variation
       - irregular letter/word spacing
       - randomized feedrate per stroke (speed variation)
  3. Emit GRBL-flavored G-code (pen up/down via Z moves by default, or via
     M3/M5 spindle-style commands if you tell it your pen-lift mechanism uses that).
  4. Optional: stream the G-code straight to the DrawCore over USB serial.

Usage examples
--------------
Generate a gcode file only:
    python3 handwriting_bot.py "Dear friend, thanks for everything." -o letter.gcode

Generate AND stream directly to the board:
    python3 handwriting_bot.py "Dear friend..." -o letter.gcode --port /dev/ttyACM0 --send

Tune the "humanness":
    python3 handwriting_bot.py "text" -o out.gcode --jitter 0.35 --slant-var 3 --seed 42

Dependencies (already installed in this environment):
    pip install hershey-fonts vnoise pyserial --break-system-packages
"""

import argparse
import math
import random
import sys
import time

from HersheyFonts import HersheyFonts
import vnoise


# --------------------------------------------------------------------------
# Config defaults -- tweak these to taste, or override via CLI flags
# --------------------------------------------------------------------------

# Common paper sizes: (full sheet width_mm, full sheet height_mm). The generator
# uses these to derive page_width_mm/page_height_mm (usable writing area) by
# subtracting the configured margins (x_offset_mm/y_offset_mm on each side).
PAPER_SIZES = {
    "Letter (8.5x11in)":  (215.9, 279.4),
    "Legal (8.5x14in)":   (215.9, 355.6),
    "A4":                 (210.0, 297.0),
    "A5":                 (148.0, 210.0),
    "Index Card (4x6in)": (101.6, 152.4),
    "Half Letter":        (139.7, 215.9),
}


def paper_size_to_cfg(name, margin_mm=15.0):
    """Convert a PAPER_SIZES entry into page_width_mm/page_height_mm/offsets,
    accounting for a margin on all sides."""
    if name not in PAPER_SIZES:
        raise ValueError(f"Unknown paper size '{name}'. Options: {list(PAPER_SIZES.keys())}")
    sheet_w, sheet_h = PAPER_SIZES[name]
    return dict(
        page_width_mm=max(10.0, sheet_w - 2 * margin_mm),
        page_height_mm=max(10.0, sheet_h - 2 * margin_mm),
        x_offset_mm=margin_mm,
        y_offset_mm=margin_mm,
    )


DEFAULTS = dict(
    font="cursive",          # used if font_variants is empty/not set
    font_variants=["cursive", "scriptc"],  # confirmed: every lowercase letter differs
                             # between these two, same em-height -- so picking randomly
                             # between them per letter gives REAL shape variation, not
                             # just scaled/skewed copies of one glyph. Set to [] or a
                             # single-item list to disable and use `font` only.
    font_size_mm=6.0,        # cap-height of letters, in mm
    line_spacing_mm=10.0,    # distance between baselines when text wraps
    page_width_mm=180.0,     # usable writing width -- wrap width AND a hard bound check
    page_height_mm=250.0,    # usable writing height -- checked after generation; you'll
                             # get a warning (not a silent failure) if text exceeds this
    x_offset_mm=10.0,
    y_offset_mm=15.0,
    tab_width_mm=15.0,       # how far a Tab character advances the cursor (aligned to
                             # a grid from the start of the line, like a text editor)

    # --- humanization knobs ---
    jitter_amp_mm=0.25,      # amplitude of Perlin-noise hand-tremor jitter
    jitter_freq=0.35,        # spatial frequency of the noise (higher = shakier/finer)
    variant_scale_jitter=0.06,   # +/- random per-letter scale variation (fraction)
    variant_rot_jitter_deg=2.5,  # +/- random per-letter rotation (degrees)
    variant_skew_jitter_deg=3.0, # +/- random per-letter shear (degrees)
    baseline_drift_amp_mm=0.6,   # slow up/down wander of the baseline
    baseline_drift_freq=0.05,    # how quickly the drift wanders along the line
    slant_base_deg=4.0,          # base italic-ish slant
    slant_var_deg=2.0,           # +/- random slant variation per word
    space_jitter_frac=0.25,      # +/- fractional jitter on letter spacing
    word_gap_extra_mm=1.5,       # base extra gap for spaces
    word_gap_jitter_frac=0.3,
    extra_letter_spacing_mm=0.0, # flat extra gap added between every letter (not just
                                 # words) -- helps print-style fonts avoid crowding/
                                 # overlapping strokes that can look like unwanted joins
    min_letter_advance_mm=2.2,   # floor on cursor advance per letter, regardless of the
                                 # glyph's own advance width. Some fonts define very
                                 # narrow letters (I, l, i) with tiny advance widths --
                                 # without a floor, the next letter starts almost on top
                                 # of them and can visually fuse/overlap with it.

    # --- motion / gcode ---
    pen_up_z=3.0,             # mm, Z height when pen is lifted
    pen_down_z=0.0,           # mm, Z height when pen is touching paper
    travel_feed=3000,         # mm/min, rapid pen-up moves
    draw_feed_base=1400,      # mm/min, base drawing feed
    draw_feed_jitter_frac=0.15,  # +/- fractional speed variation per stroke
    pen_lift_mode="z",        # "z" (Z-axis servo/solenoid) or "spindle" (M3/M5 laser-style)

    # If new lines are appearing ABOVE the previous line instead of below it
    # (a common CoreXY/axis-orientation surprise), set this True. This only
    # changes which way the vertical cursor advances between lines -- it does
    # NOT flip individual letter shapes (unlike flip_y in the EBB output stage).
    reverse_line_direction=False,

    # Custom/hand-drawn or SVG-converted letter variants (see custom_font.py).
    # These MUST be present here (even as None/False) for the CLI flags in
    # handwriting_ebb.py to actually reach the generator's config.
    custom_font_path=None,
    custom_font_exclusive=False,
    custom_font_exclude_letters="",  # e.g. "EJ" -- these letters skip the custom
                                      # font entirely and always use Hershey instead,
                                      # even when custom_font_exclusive is True.
                                      # Useful when specific glyphs in a custom/converted
                                      # font read ambiguously (e.g. a cursive E that
                                      # looks like R) and a hand-redraw isn't done yet.
)

# Convenience bundle for print-style fonts (like EMS Casual Hand) where a
# forced cursive slant/skew and tight spacing can distort letterforms into
# looking like other letters. Applied via --print-style in handwriting_ebb.py,
# or manually: cfg.update(PRINT_STYLE_OVERRIDES)
PRINT_STYLE_OVERRIDES = dict(
    slant_base_deg=0.0,
    slant_var_deg=0.5,           # tiny natural wobble only, no forced cursive lean
    variant_skew_jitter_deg=0.5, # was 3.0 -- minimal shear, upright letters stay upright
    variant_rot_jitter_deg=1.2,  # was 2.5 -- less per-letter rotation
    extra_letter_spacing_mm=0.8, # breathing room so strokes don't crowd/overlap
    word_gap_extra_mm=2.2,       # slightly wider word gaps too
)

# Fonts built into HersheyFonts worth knowing about:
#   "cursive"  -> joined-up script, most "handwriting-like" out of the box
#   "scripts"  -> simple script / small
#   "futural"  -> plain single-stroke sans, very robotic looking (good baseline test)
#   "rowmans"  -> simplex roman
# Run `python3 -c "from HersheyFonts import HersheyFonts; print(HersheyFonts().default_font_names)"`
# to see the full list.


# --------------------------------------------------------------------------
# Noise helper
# --------------------------------------------------------------------------

class Jitter:
    """Smooth 2D Perlin-noise offset generator (hand-tremor style jitter)."""

    def __init__(self, seed=0, amp_mm=0.25, freq=0.35):
        self.n = vnoise.Noise()
        self.seed = seed
        self.amp = amp_mm
        self.freq = freq
        # offset each axis into a different region of noise space so x/y
        # jitter aren't correlated
        self.ox = seed * 1000.0
        self.oy = seed * 1000.0 + 5000.0

    def offset(self, x, y):
        nx = self.n.noise2(self.ox + x * self.freq, y * self.freq * 0.3) * self.amp
        ny = self.n.noise2(self.oy + x * self.freq, y * self.freq * 0.3) * self.amp
        return nx, ny


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def apply_transform(points, scale, rot_deg, skew_deg):
    """Apply scale + rotation + shear about the origin to a list of (x, y)."""
    rot = math.radians(rot_deg)
    skew = math.radians(skew_deg)
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    out = []
    for x, y in points:
        # shear (skew) along x based on y
        x = x + y * math.tan(skew)
        # scale
        x *= scale
        y *= scale
        # rotate
        xr = x * cos_r - y * sin_r
        yr = x * sin_r + y * cos_r
        out.append((xr, yr))
    return out


# --------------------------------------------------------------------------
# Core: text -> humanized stroke groups
# --------------------------------------------------------------------------

class HandwritingGenerator:
    def __init__(self, cfg, rng_seed=None):
        self.cfg = cfg
        self.rng = random.Random(rng_seed)
        self.jitter = Jitter(seed=rng_seed or 0,
                              amp_mm=cfg["jitter_amp_mm"],
                              freq=cfg["jitter_freq"])

        variant_names = cfg.get("font_variants") or [cfg["font"]]
        self.hf_variants = []
        for fname in variant_names:
            hf = HersheyFonts()
            hf.load_default_font(fname)
            self.hf_variants.append(hf)
        # keep self.hf pointing at the first variant for render_options/unit_scale
        self.hf = self.hf_variants[0]
        # Hershey glyphs are defined on a ~21-unit em (cap_line..bottom_line);
        # figure out a scale factor to hit the requested cap-height in mm.
        cap = abs(self.hf.render_options["cap_line"])
        base = self.hf.render_options["base_line"]
        em_height = base - self.hf.render_options["cap_line"]
        self.unit_scale = cfg["font_size_mm"] / em_height

        # Optional: your own hand-drawn letter variants (see custom_font.py /
        # the letter_designer.html tool). Uses the same coordinate convention
        # as Hershey glyphs (y-down positive, baseline/cap-height matching
        # render_options above), so it drops straight into the same pipeline.
        self.custom_glyphs = {}
        custom_path = cfg.get("custom_font_path")
        if custom_path:
            from custom_font import load_custom_font
            self.custom_glyphs = load_custom_font(custom_path)

        self.custom_exclude = set()
        for c in (cfg.get("custom_font_exclude_letters") or ""):
            self.custom_exclude.add(c.lower())
            self.custom_exclude.add(c.upper())

    def wrap_text(self, text, width_mm=None, font_scale=1.0):
        """Greedy word-wrap to a given width (defaults to page_width_mm), honoring existing newlines."""
        cfg = self.cfg
        width_mm = width_mm if width_mm is not None else cfg["page_width_mm"]
        effective_size = cfg["font_size_mm"] * font_scale
        max_chars_guess = max(6, int(width_mm / (effective_size * 0.55)))
        lines = []
        for paragraph in text.split("\n"):
            words = paragraph.split(" ")
            cur = ""
            for w in words:
                trial = (cur + " " + w).strip()
                if len(trial) > max_chars_guess and cur:
                    lines.append(cur)
                    cur = w
                else:
                    cur = trial
            lines.append(cur)
        return lines

    def _render_line(self, line, x_start, y_cursor, font_scale=1.0):
        """
        Render one already-wrapped line of text at the given position.
        Returns the list of strokes produced for this line (mm coordinates).
        This is the shared per-character humanization logic used by both
        generate() (flat text) and generate_blocks() (structured input).
        """
        cfg = self.cfg
        strokes_out = []
        x_cursor = x_start
        drift_phase = self.rng.uniform(0, 1000)
        current_slant = cfg["slant_base_deg"] + self.rng.uniform(
            -cfg["slant_var_deg"], cfg["slant_var_deg"])
        glyph_scale = self.unit_scale * font_scale

        for ch in line:
            if ch == "\t":
                # advance to the next tab stop, grid-aligned to the start of
                # this line (x_start), like a text editor's tab behavior
                rel = x_cursor - x_start
                stop_index = int(rel // cfg["tab_width_mm"]) + 1
                x_cursor = x_start + stop_index * cfg["tab_width_mm"]
                continue

            if ch == " ":
                gap = cfg["word_gap_extra_mm"] * (
                    1 + self.rng.uniform(-cfg["word_gap_jitter_frac"],
                                          cfg["word_gap_jitter_frac"]))
                x_cursor += cfg["font_size_mm"] * font_scale * 0.35 + gap
                current_slant = cfg["slant_base_deg"] + self.rng.uniform(
                    -cfg["slant_var_deg"], cfg["slant_var_deg"])
                continue

            candidates = []
            custom_list = None
            if ch not in self.custom_exclude:
                custom_list = self.custom_glyphs.get(ch) or self.custom_glyphs.get(ch.lower())
            if custom_list:
                candidates.extend(custom_list)
            if not custom_list or not self.cfg.get("custom_font_exclusive"):
                for hf in self.hf_variants:
                    hershey_glyphs = list(hf.glyphs_for_text(ch))
                    if hershey_glyphs:
                        candidates.append(hershey_glyphs[0])
            if not candidates:
                continue
            g = self.rng.choice(candidates)

            s = 1.0 + self.rng.uniform(-cfg["variant_scale_jitter"],
                                        cfg["variant_scale_jitter"])
            rot = self.rng.uniform(-cfg["variant_rot_jitter_deg"],
                                    cfg["variant_rot_jitter_deg"])
            skew = current_slant + self.rng.uniform(
                -cfg["variant_skew_jitter_deg"], cfg["variant_skew_jitter_deg"])

            drift = self.jitter.n.noise1(
                drift_phase + x_cursor * cfg["baseline_drift_freq"]
            ) * cfg["baseline_drift_amp_mm"]

            for raw_stroke in g.strokes:
                pts = [(px, -py) for px, py in raw_stroke]
                pts = apply_transform(pts, s * glyph_scale, rot, skew)

                world_pts = []
                for px, py in pts:
                    wx = x_cursor + px
                    wy = y_cursor + py + drift
                    jx, jy = self.jitter.offset(wx, wy)
                    world_pts.append((wx + jx, wy + jy))
                strokes_out.append(world_pts)

            advance = max(g.char_width * glyph_scale + cfg["extra_letter_spacing_mm"],
                          cfg["min_letter_advance_mm"])
            letter_spacing_jitter = advance * self.rng.uniform(
                -cfg["space_jitter_frac"] * 0.15, cfg["space_jitter_frac"] * 0.15)
            x_cursor += advance + letter_spacing_jitter

        return strokes_out

    def generate(self, text):
        """
        Returns a list of "pen strokes": each stroke is a list of (x_mm, y_mm)
        points that should be drawn pen-down, continuously, in order. Moves
        between strokes are implicit pen-up travels.
        """
        cfg = self.cfg
        strokes_out = []
        lines = self.wrap_text(text)
        y_cursor = cfg["y_offset_mm"]
        line_dir = -1 if cfg.get("reverse_line_direction") else 1
        for line in lines:
            strokes_out.extend(self._render_line(line, cfg["x_offset_mm"], y_cursor))
            y_cursor += line_dir * cfg["line_spacing_mm"]
        return strokes_out

    def generate_blocks(self, blocks):
        """
        Like generate(), but takes structured blocks (see text_extractor.extract_blocks):
        [{"type": "heading"|"bullet"|"numbered"|"paragraph"|"blank", "text": str, "level": int}, ...]

        Headings are rendered larger, bullets/numbered items get a real
        marker + indent, paragraph breaks (blank blocks) add extra vertical
        gap -- so the robot's output preserves the document's structure
        instead of flattening everything into one undifferentiated block
        of handwriting.
        """
        cfg = self.cfg
        strokes_out = []
        y_cursor = cfg["y_offset_mm"]
        numbered_counter = 0
        line_dir = -1 if cfg.get("reverse_line_direction") else 1

        HEADING_SCALE = {1: 1.6, 2: 1.35, 3: 1.15}
        INDENT_MM = 8.0

        prev_type = None
        for block in blocks:
            btype = block["type"]

            if btype == "blank":
                y_cursor += line_dir * cfg["line_spacing_mm"] * 0.6
                numbered_counter = 0
                prev_type = btype
                continue

            if btype != "numbered":
                numbered_counter = 0

            if btype == "heading":
                scale = HEADING_SCALE.get(block["level"], 1.15)
                if prev_type not in (None, "blank"):
                    y_cursor += line_dir * cfg["line_spacing_mm"] * 0.4  # extra gap before heading
                lines = self.wrap_text(block["text"], font_scale=scale)
                for line in lines:
                    strokes_out.extend(self._render_line(line, cfg["x_offset_mm"], y_cursor, font_scale=scale))
                    y_cursor += line_dir * cfg["line_spacing_mm"] * scale
                y_cursor += line_dir * cfg["line_spacing_mm"] * 0.3  # extra gap after heading

            elif btype == "bullet":
                x_indent = cfg["x_offset_mm"] + INDENT_MM
                marker = "- "
                lines = self.wrap_text(marker + block["text"],
                                        width_mm=cfg["page_width_mm"] - INDENT_MM)
                for i, line in enumerate(lines):
                    x_start = x_indent if i == 0 else x_indent + INDENT_MM * 0.6
                    strokes_out.extend(self._render_line(line, x_start, y_cursor))
                    y_cursor += line_dir * cfg["line_spacing_mm"]

            elif btype == "numbered":
                numbered_counter += 1
                x_indent = cfg["x_offset_mm"] + INDENT_MM
                marker = f"{numbered_counter}. "
                lines = self.wrap_text(marker + block["text"],
                                        width_mm=cfg["page_width_mm"] - INDENT_MM)
                for i, line in enumerate(lines):
                    x_start = x_indent if i == 0 else x_indent + INDENT_MM * 0.6
                    strokes_out.extend(self._render_line(line, x_start, y_cursor))
                    y_cursor += line_dir * cfg["line_spacing_mm"]

            else:  # paragraph
                lines = self.wrap_text(block["text"])
                for line in lines:
                    strokes_out.extend(self._render_line(line, cfg["x_offset_mm"], y_cursor))
                    y_cursor += line_dir * cfg["line_spacing_mm"]

            prev_type = btype

        return strokes_out


# --------------------------------------------------------------------------
# Page bounds checking
# --------------------------------------------------------------------------

def check_page_bounds(strokes, cfg):
    """
    Check generated strokes against the configured page_width_mm/page_height_mm.
    Measures the actual footprint (span) of the geometry rather than assuming
    a fixed positive direction -- calibration settings like
    reverse_line_direction/flip_x/flip_y can make content grow in either
    direction depending on the specific machine, so a coordinate being
    negative isn't inherently a problem; what matters is the total span.

    Returns a dict: {fits, used_width_mm, used_height_mm, available_width_mm,
    available_height_mm, message}. Doesn't modify strokes or raise -- callers
    decide whether to warn, block, or just log it.
    """
    if not strokes:
        return dict(fits=True, used_width_mm=0, used_height_mm=0,
                     available_width_mm=cfg["page_width_mm"],
                     available_height_mm=cfg["page_height_mm"],
                     message="No strokes generated.")

    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]

    used_width = max(xs) - min(xs)
    used_height = max(ys) - min(ys)
    available_width = cfg["page_width_mm"]
    available_height = cfg["page_height_mm"]

    over_x = used_width > available_width
    over_y = used_height > available_height
    fits = not (over_x or over_y)

    if fits:
        message = (f"Fits comfortably: uses {used_width:.0f}x{used_height:.0f}mm "
                    f"of {available_width:.0f}x{available_height:.0f}mm available.")
    else:
        problems = []
        if over_y:
            problems.append(f"text needs {used_height:.0f}mm of height but only "
                             f"{available_height:.0f}mm is available (overflows by "
                             f"{used_height - available_height:.0f}mm)")
        if over_x:
            problems.append(f"text needs {used_width:.0f}mm of width but only "
                             f"{available_width:.0f}mm is available (overflows by "
                             f"{used_width - available_width:.0f}mm)")
        message = "Does NOT fit: " + "; ".join(problems) + "."

    return dict(fits=fits, used_width_mm=used_width, used_height_mm=used_height,
                 available_width_mm=available_width, available_height_mm=available_height,
                 message=message)


# --------------------------------------------------------------------------
# G-code emission
# --------------------------------------------------------------------------

def strokes_to_gcode(strokes, cfg, rng_seed=None):
    rng = random.Random(rng_seed)
    lines = []
    lines.append("; Generated by handwriting_bot.py")
    lines.append("G21 ; mm units")
    lines.append("G90 ; absolute positioning")
    lines.append(f"G0 F{cfg['travel_feed']}")

    def pen_up():
        if cfg["pen_lift_mode"] == "z":
            lines.append(f"G0 Z{cfg['pen_up_z']:.3f}")
        else:
            lines.append("M5 ; pen up (spindle/laser off)")

    def pen_down():
        if cfg["pen_lift_mode"] == "z":
            lines.append(f"G1 Z{cfg['pen_down_z']:.3f} F{cfg['travel_feed']}")
        else:
            lines.append("M3 S1000 ; pen down (spindle/laser on)")

    pen_up()

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        x0, y0 = stroke[0]
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f}")
        pen_down()

        feed = cfg["draw_feed_base"] * (
            1 + rng.uniform(-cfg["draw_feed_jitter_frac"], cfg["draw_feed_jitter_frac"]))
        lines.append(f"G1 F{feed:.0f}")
        for x, y in stroke[1:]:
            lines.append(f"G1 X{x:.3f} Y{y:.3f}")

        pen_up()

    lines.append("G0 X0 Y0")
    lines.append("M2 ; program end")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Optional: stream G-code to a GRBL board over serial
# --------------------------------------------------------------------------

def stream_gcode(gcode_text, port, baud=115200, verbose=True):
    import serial  # pyserial

    lines = [l for l in gcode_text.splitlines() if l.strip() and not l.strip().startswith(";")]

    with serial.Serial(port, baud, timeout=5) as ser:
        time.sleep(2)  # let GRBL reset/wake
        ser.reset_input_buffer()
        ser.write(b"\r\n\r\n")
        time.sleep(1)
        ser.reset_input_buffer()

        for i, line in enumerate(lines):
            cmd = line.split(";")[0].strip()
            if not cmd:
                continue
            ser.write((cmd + "\n").encode())
            resp = ser.readline().decode(errors="replace").strip()
            while resp == "":
                resp = ser.readline().decode(errors="replace").strip()
            if verbose:
                print(f"[{i+1}/{len(lines)}] {cmd}  ->  {resp}")
            if "error" in resp.lower():
                print(f"!! GRBL reported an error on line: {cmd}", file=sys.stderr)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_cfg_from_args(args):
    cfg = dict(DEFAULTS)
    for key in cfg:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


def main():
    p = argparse.ArgumentParser(description="Text -> human-like G-code for GRBL pen plotters (DrawCore).")
    p.add_argument("text", nargs="?", help="Text to write. If omitted, reads from --text-file.")
    p.add_argument("--text-file", help="Read text to write from this file instead of the CLI arg.")
    p.add_argument("-o", "--output", default="handwriting.gcode", help="Output .gcode file path.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (for reproducible output).")

    p.add_argument("--font", help=f"HersheyFonts font name (default: {DEFAULTS['font']}).")
    p.add_argument("--font-size-mm", type=float, dest="font_size_mm")
    p.add_argument("--page-width-mm", type=float, dest="page_width_mm")

    p.add_argument("--jitter", type=float, dest="jitter_amp_mm", help="Hand-tremor jitter amplitude, mm.")
    p.add_argument("--slant-var", type=float, dest="slant_var_deg", help="Per-word slant variation, degrees.")
    p.add_argument("--pen-lift-mode", choices=["z", "spindle"], dest="pen_lift_mode")

    p.add_argument("--port", help="Serial port for the DrawCore board (e.g. /dev/ttyACM0 or COM5).")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--send", action="store_true", help="Stream the generated G-code to --port after writing it.")

    args = p.parse_args()

    if args.text_file:
        with open(args.text_file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        p.error("Provide text as an argument or via --text-file.")
        return

    cfg = build_cfg_from_args(args)
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    print(f"[info] using random seed {seed} (pass --seed {seed} to reproduce this exact output)")

    gen = HandwritingGenerator(cfg, rng_seed=seed)
    strokes = gen.generate(text)
    gcode = strokes_to_gcode(strokes, cfg, rng_seed=seed)

    with open(args.output, "w") as f:
        f.write(gcode)
    print(f"[info] wrote {args.output}  ({len(strokes)} strokes, {len(gcode.splitlines())} gcode lines)")

    if args.send:
        if not args.port:
            p.error("--send requires --port")
        print(f"[info] streaming to {args.port} @ {args.baud} baud ...")
        stream_gcode(gcode, args.port, args.baud)
        print("[info] done.")


if __name__ == "__main__":
    main()