#!/usr/bin/env python3
"""
handwriting_ebb.py
===================
Text -> human-like pen strokes -> EBB (EiBotBoard) commands, for your
DrawCore V1.2 board (confirmed: EBB firmware v2.6.5, CoreXY mechanism).

This reuses the exact same humanization pipeline as handwriting_bot.py
(HersheyFonts strokes + Perlin jitter + variant letters + baseline drift),
but outputs real EBB serial commands instead of GRBL G-code:
    - XM,<duration_ms>,<dxA_steps>,<dxB_steps>   CoreXY-aware relative move
    - SP,<0|1>                                    pen up / pen down (servo)
    - EM,1,1                                      enable motors
    - CS                                          zero step counters

IMPORTANT -- you must calibrate two things before real output will be
correctly scaled/oriented:

  1. steps_per_mm  -- how many motor steps correspond to 1mm of real travel.
     Calibrate by: zero position (EM,1,1 then CS), send one XM move of a
     known step count, physically measure how far the pen moved, then
         steps_per_mm = steps_sent / mm_measured
     Set this via --steps-per-mm.

  2. pen_up_value / pen_down_value -- which SP,<value> lifts vs lowers the
     pen on your servo. Defaults assume SP,0 = up, SP,1 = down (the common
     EBB/EggBot convention) -- VERIFY this on your machine with --test-pen
     before running a real job, since it's easy to have this backwards.

Usage
-----
Verify which SP value lifts the pen (does this FIRST, before anything else):
    python3 handwriting_ebb.py --port /dev/tty.usbmodem11401 --test-pen

Generate an EBB command file only (no hardware needed):
    python3 handwriting_ebb.py "Dear friend" -o letter.ebb

Generate AND stream to the board:
    python3 handwriting_ebb.py "Dear friend" -o letter.ebb --port /dev/tty.usbmodem11401 --send

Write out the contents of a text file, Word doc, or PDF instead of typing text:
    python3 handwriting_ebb.py --file myletter.docx -o letter.ebb --port /dev/tty.usbmodem11401 --send
    python3 handwriting_ebb.py --file myletter.pdf  -o letter.ebb --port /dev/tty.usbmodem11401 --send
    python3 handwriting_ebb.py --file myletter.txt  -o letter.ebb --port /dev/tty.usbmodem11401 --send

Dependencies (already installed):
    pip install hershey-fonts vnoise pyserial python-docx pypdf
"""

import argparse
import random
import sys
import time

# Reuse the stroke generator from the GRBL version -- same humanization logic.
from handwriting_bot import (HandwritingGenerator, DEFAULTS as BASE_DEFAULTS,
                               PRINT_STYLE_OVERRIDES, PAPER_SIZES, paper_size_to_cfg,
                               check_page_bounds)
from text_extractor import extract_text, extract_blocks, clean_text


EBB_DEFAULTS = dict(BASE_DEFAULTS)
EBB_DEFAULTS.update(
    steps_per_mm=80.0,       # calibrated on your machine
    pen_up_value=1,          # confirmed: SP,1 = pen up on your board
    pen_down_value=0,        # confirmed: SP,0 = pen down on your board
    pen_move_settle_ms=500,  # how long to wait after an SP command before moving --
                             # confirmed via testing that 200ms was too short and let
                             # the pen drag between letters before fully lifting
    travel_speed_mm_s=45.0,  # pen-up rapid speed
    draw_speed_mm_s=18.0,    # pen-down drawing speed
    draw_speed_jitter_frac=0.15,
    min_move_ms=10,          # EBB has a practical minimum move duration
    flip_x=True,             # confirmed: needed to fix left-right mirroring on your board
    flip_y=False,            # do NOT use this to fix line-stacking direction -- it mirrors
                             # letter shapes too. Use reverse_line_direction instead.
    reverse_line_direction=True,  # confirmed: needed so new lines go down, not up
)


# --------------------------------------------------------------------------
# Strokes (mm) -> EBB command list
# --------------------------------------------------------------------------

def strokes_to_ebb_commands(strokes, cfg, rng_seed=None):
    rng = random.Random(rng_seed)
    steps_per_mm = cfg["steps_per_mm"]
    pen_up_val = cfg["pen_up_value"]
    pen_down_val = cfg["pen_down_value"]

    cmds = []
    cmds.append("EM,1,1")     # enable motors, default microstepping
    cmds.append("CS")         # zero here -- caller is responsible for having
                               # physically homed to a known reference first
    cmds.append(f"SP,{pen_up_val}")

    cur_x, cur_y = 0.0, 0.0
    pen_is_down = False

    def move_to(x, y, speed_mm_s):
        nonlocal cur_x, cur_y
        dx_mm = x - cur_x
        dy_mm = y - cur_y
        dist = (dx_mm ** 2 + dy_mm ** 2) ** 0.5
        if dist < 1e-6:
            return
        dx_steps = round(dx_mm * steps_per_mm) * (-1 if cfg["flip_x"] else 1)
        dy_steps = round(dy_mm * steps_per_mm) * (-1 if cfg["flip_y"] else 1)
        if dx_steps == 0 and dy_steps == 0:
            cur_x, cur_y = x, y
            return
        duration_ms = max(cfg["min_move_ms"], round((dist / speed_mm_s) * 1000))
        # XM is CoreXY-aware: internally Axis1 = A+B, Axis2 = A-B
        cmds.append(f"XM,{duration_ms},{dx_steps},{dy_steps}")
        cur_x, cur_y = x, y

    for stroke in strokes:
        if len(stroke) < 2:
            continue

        # Travel (pen up) to the start of this stroke.
        if pen_is_down:
            cmds.append(f"SP,{pen_up_val}")
            pen_is_down = False
        x0, y0 = stroke[0]
        move_to(x0, y0, cfg["travel_speed_mm_s"])

        # Pen down, draw the stroke.
        cmds.append(f"SP,{pen_down_val}")
        pen_is_down = True

        speed = cfg["draw_speed_mm_s"] * (
            1 + rng.uniform(-cfg["draw_speed_jitter_frac"], cfg["draw_speed_jitter_frac"]))
        for x, y in stroke[1:]:
            move_to(x, y, speed)

    if pen_is_down:
        cmds.append(f"SP,{pen_up_val}")

    # Return to the zeroed origin at the end, pen up.
    move_to(0.0, 0.0, cfg["travel_speed_mm_s"])

    return cmds


# --------------------------------------------------------------------------
# Serial streaming (EBB line protocol: send command + CR, expect "OK")
# --------------------------------------------------------------------------

def stream_ebb(commands, port, baud=115200, verbose=True, pen_settle_s=0.2):
    import serial

    with serial.Serial(port, baud, timeout=5) as ser:
        time.sleep(2)  # let the board finish USB enumeration/reset
        ser.reset_input_buffer()

        for i, cmd in enumerate(commands):
            ser.write((cmd + "\r").encode())
            deadline = time.time() + 5
            resp_lines = []
            while time.time() < deadline:
                line = ser.readline().decode(errors="replace").strip()
                if line:
                    resp_lines.append(line)
                    if line.upper().startswith("OK") or "err" in line.lower():
                        break
            resp = " | ".join(resp_lines) if resp_lines else "<no response>"
            if verbose:
                print(f"[{i+1}/{len(commands)}] {cmd}  ->  {resp}")
            if any("err" in r.lower() for r in resp_lines):
                print(f"!! Board reported an error on: {cmd}", file=sys.stderr)

            if cmd.startswith("SP,"):
                time.sleep(pen_settle_s)
            elif cmd.startswith("XM,") or cmd.startswith("SM,"):
                # IMPORTANT: the board's "OK" response means the command was
                # accepted/queued, NOT that the physical motion has finished
                # executing yet. If we send the next command (especially a
                # pen-lift SP right after the last stroke of a line) before
                # this move's own declared duration has actually elapsed, it
                # can cut the current stroke off mid-motion -- this is what
                # causes letters at the start/end of a line to look
                # incomplete. So we explicitly wait out the move's own
                # duration here, on top of waiting for OK.
                try:
                    duration_ms = int(cmd.split(",")[1])
                    time.sleep(duration_ms / 1000.0)
                except (IndexError, ValueError):
                    pass


def test_pen(port, baud=115200):
    """Toggle SP,0 and SP,1 with pauses so you can visually confirm which is up/down."""
    import serial

    with serial.Serial(port, baud, timeout=5) as ser:
        time.sleep(2)
        ser.reset_input_buffer()
        for val in (0, 1, 0, 1):
            print(f"Sending SP,{val} ... watch the pen now.")
            ser.write(f"SP,{val}\r".encode())
            time.sleep(0.3)
            resp = ser.readline().decode(errors="replace").strip()
            print(f"  -> {resp}")
            input("  Press Enter to continue to the next test...")
    print(
        "\nBased on what you observed: whichever value lifted the pen is your "
        "pen_up_value, and whichever lowered it is pen_down_value. Pass these "
        "in with --pen-up-value and --pen-down-value if they're not the "
        "defaults (0=up, 1=down)."
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Text -> human-like EBB commands for CoreXY DrawCore.")
    p.add_argument("text", nargs="?", help="Text to write.")
    p.add_argument("--text-file", help="Read raw text from a plain .txt file (no parsing).")
    p.add_argument("--file", help="Read text from a .txt, .docx, or .pdf file (auto-detected by extension).")
    p.add_argument("-o", "--output", default="handwriting.ebb", help="Output command file path.")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--steps-per-mm", type=float, dest="steps_per_mm",
                    help="Calibrated steps/mm (see docstring for calibration procedure).")
    p.add_argument("--pen-up-value", type=int, dest="pen_up_value")
    p.add_argument("--pen-down-value", type=int, dest="pen_down_value")
    p.add_argument("--font", help=f"HersheyFonts font name (default: {BASE_DEFAULTS['font']}).")
    p.add_argument("--custom-font", dest="custom_font_path",
                    help="Path to a JSON font file exported from letter_designer.html.")
    p.add_argument("--custom-font-exclusive", dest="custom_font_exclusive",
                    action="store_const", const=True,
                    help="Only use your custom letters where defined (no mixing with Hershey fonts for those letters).")
    p.add_argument("--exclude-letters", dest="custom_font_exclude_letters",
                    help="Letters that should always fall back to Hershey instead of the custom font "
                         "(e.g. \"EJ\" if those specific glyphs read ambiguously). Case-insensitive.")
    p.add_argument("--print-style", action="store_true",
                    help="Disable forced cursive slant/skew and add breathing room between letters -- "
                         "use this for upright print-style fonts (like EMS Casual Hand) instead of true cursive.")
    p.add_argument("--pen-settle-ms", type=int, dest="pen_move_settle_ms",
                    help="How long (ms) to pause after each pen up/down command before the next move. "
                         "Increase this if letters are dragging/connecting when they shouldn't (default: 200).")
    p.add_argument("--font-size-mm", type=float, dest="font_size_mm")
    p.add_argument("--page-width-mm", type=float, dest="page_width_mm")
    p.add_argument("--page-height-mm", type=float, dest="page_height_mm")
    p.add_argument("--paper-size", choices=list(PAPER_SIZES.keys()),
                    help="Preset paper size -- sets width/height/margins automatically. "
                         "Overridden by --page-width-mm/--page-height-mm if also given.")
    p.add_argument("--margin-mm", type=float, default=15.0,
                    help="Margin used with --paper-size (default: 15mm on all sides).")
    p.add_argument("--jitter", type=float, dest="jitter_amp_mm")
    p.add_argument("--slant-var", type=float, dest="slant_var_deg")
    p.add_argument("--flip-x", dest="flip_x", action="store_const", const=True,
                    help="Flip X direction if output is mirrored left-right.")
    p.add_argument("--flip-y", dest="flip_y", action="store_const", const=True,
                    help="Flip Y direction if output is mirrored top-bottom.")
    p.add_argument("--reverse-line-direction", dest="reverse_line_direction",
                    action="store_const", const=True,
                    help="If new lines appear ABOVE the previous line instead of below it.")
    p.add_argument("--no-reverse-line-direction", dest="reverse_line_direction",
                    action="store_const", const=False,
                    help="Force normal line direction (override the default).")

    p.add_argument("--port", help="Serial port, e.g. /dev/tty.usbmodem11401")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--send", action="store_true", help="Stream commands to --port after writing the file.")
    p.add_argument("--test-pen", action="store_true",
                    help="Just toggle the pen up/down a few times to identify SP values, then exit.")
    p.add_argument("--flat-text", action="store_true",
                    help="With --file: ignore document structure (headings/bullets/lists) and just "
                         "write all the text as plain flowing paragraphs.")

    args = p.parse_args()

    if args.test_pen:
        if not args.port:
            p.error("--test-pen requires --port")
        test_pen(args.port, args.baud)
        return

    use_blocks = False
    blocks = None
    text = None

    if args.file:
        if args.flat_text:
            print(f"[info] extracting text from {args.file} (flat, structure ignored) ...")
            try:
                raw = extract_text(args.file)
            except Exception as e:
                p.error(f"failed to extract text from '{args.file}': {e}")
                return
            text = clean_text(raw)
            preview = text[:200].replace("\n", " / ")
            print(f"[info] extracted {len(text)} characters. Preview: {preview}...")
        else:
            print(f"[info] extracting structured content from {args.file} ...")
            try:
                blocks = extract_blocks(args.file)
            except Exception as e:
                p.error(f"failed to extract structure from '{args.file}': {e}")
                return
            use_blocks = True
            non_blank = [b for b in blocks if b["type"] != "blank"]
            print(f"[info] extracted {len(non_blank)} content blocks:")
            for b in non_blank[:8]:
                preview = b["text"][:60]
                print(f"    [{b['type']}] {preview}")
            if len(non_blank) > 8:
                print(f"    ... and {len(non_blank) - 8} more")
    elif args.text_file:
        with open(args.text_file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        p.error("Provide text, use --file <path> / --text-file <path>, or --test-pen.")
        return

    cfg = dict(EBB_DEFAULTS)
    if args.print_style:
        cfg.update(PRINT_STYLE_OVERRIDES)
        print("[info] --print-style applied: cursive slant/skew disabled, extra letter spacing added")
    if args.paper_size:
        cfg.update(paper_size_to_cfg(args.paper_size, margin_mm=args.margin_mm))
        print(f"[info] paper size '{args.paper_size}' applied "
              f"(usable area {cfg['page_width_mm']:.0f}x{cfg['page_height_mm']:.0f}mm "
              f"with {args.margin_mm:.0f}mm margins)")
    for key in cfg:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    print(f"[info] using random seed {seed} (pass --seed {seed} to reproduce this exact output)")

    gen = HandwritingGenerator(cfg, rng_seed=seed)
    if use_blocks:
        strokes = gen.generate_blocks(blocks)
    else:
        strokes = gen.generate(text)

    bounds = check_page_bounds(strokes, cfg)
    if bounds["fits"]:
        print(f"[info] {bounds['message']}")
    else:
        print(f"[WARNING] {bounds['message']}")
        print("[WARNING] The robot may try to write outside your paper or physical travel area. "
              "Consider shortening the text, reducing font size, or choosing a larger paper size.")

    commands = strokes_to_ebb_commands(strokes, cfg, rng_seed=seed)

    with open(args.output, "w") as f:
        f.write("\n".join(commands) + "\n")
    print(f"[info] wrote {args.output}  ({len(strokes)} strokes, {len(commands)} EBB commands)")
    print(f"[info] using steps_per_mm={cfg['steps_per_mm']} -- if this isn't calibrated yet, "
          f"the physical size/scale of the writing WILL be wrong.")

    if args.send:
        if not args.port:
            p.error("--send requires --port")
        print(f"[info] streaming to {args.port} @ {args.baud} baud ...")
        stream_ebb(commands, args.port, args.baud, pen_settle_s=cfg["pen_move_settle_ms"] / 1000.0)
        print("[info] done.")


if __name__ == "__main__":
    main()