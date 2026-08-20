"""
preview_document.py
======================
Renders a full-page visual preview of what the robot would physically
write on real paper -- without needing the robot connected at all.

Run from your project directory:
    python preview_document.py

EDIT TEXT / ENGINE / OUTPUT_PNG BELOW before running -- the defaults are
just placeholder content so the script runs out of the box.

WHY flip_x/flip_y ARE NOT APPLIED HERE
-----------------------------------------
flip_x/flip_y (used in strokes_to_ebb_commands) exist ONLY to cancel out
THIS specific robot's own physical mechanical mirroring during actual
execution -- the correction and the hardware's inherent mirroring cancel
each other out, so real paper output ends up correctly oriented using the
RAW (uncorrected) world coordinates. An earlier version of this script
applied those corrections to the preview image directly, which produced
backwards/reordered output when checked visually (confirmed: text came out
mirrored and in reversed line order) -- since there's no physical robot
present in a plain 2D preview to cancel the correction back out. This
version renders the raw strokes directly, which is what actually matches
final paper appearance.
"""

from PIL import Image, ImageDraw

# --- EDIT THESE before running, or you'll just get this placeholder text ---
TEXT = """
The difference between a DFA and an NFA is that a DFA seems to be a finite acceptor in which every transition state must be defined for each internal state. For example, let's say that we have an alphabet that accepts just a set of a and b. Each internal state needs to be defined as what it would do if it got an a or b input. As for an NFA, you don’t need to explain every transition state for the internal states. You can just define the state that you need. Also, NFAs allow for things such as lamba transitions, enabling secondary arguments. When it comes to what I find more straightforward to construct for me, DFA’s reason is that, for NFA’s, I just have a hard time knowing when to use a lambda function to skip processes. And for DFA, all I know is that I have to define what to do for each input for an internal state. When converting an NFA to a DFA, I use the table method since it helps me keep track of everything as I go.  


Procedure Bob is used mainly in this course to convert an NFA to a DFA that allows only one accepting state. Procedure Mark reduces the number of states that a DFA contains.
"""
ENGINE = "rnn"   # "hershey" or "rnn" -- change this to actually preview RNN output
VERIFY = True      # True = run through the SAME verification/retry pipeline the GUI's
                      # Verify checkbox uses (only meaningful with ENGINE="rnn") -- set
                      # this True if you want the preview to match exactly what "Write
                      # with robot" would send when Verify is checked, including retries.
OUTPUT_PNG = "document_preview.png"


def render_document_preview(strokes, cfg, dpi: float = 150.0, margin_px: int = 20):
    """
    Renders strokes DIRECTLY, with no flip_x/flip_y correction applied --
    representing what the final output looks like on real paper.

    IMPORTANT: flip_x/flip_y exist ONLY to cancel out THIS specific robot's
    own physical mechanical mirroring during actual execution -- the
    correction and the hardware's inherent mirroring cancel each other out,
    so the net result on real paper ends up correctly oriented using the
    RAW (uncorrected) world coordinates. Applying flip_x/flip_y to a plain
    2D image preview (with no physical robot present to cancel it back out)
    shows the deliberately-mirrored pre-cancellation intermediate state
    instead -- which is backwards and was confirmed wrong via direct visual
    inspection. Uses PIL's natural downward-y pixel convention directly,
    matching this pipeline's documented "y increases downward = world/page
    coordinates" convention (see handwriting_bot.py's Hershey glyph
    negation comment).
    """
    all_pts = [p for s in strokes for p in s]
    if not all_pts:
        raise ValueError("No strokes to render -- check TEXT isn't empty.")

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    mm_to_px = dpi / 25.4
    width_px = int((max_x - min_x) * mm_to_px) + margin_px * 2
    height_px = int((max_y - min_y) * mm_to_px) + margin_px * 2

    img = Image.new("L", (width_px, height_px), color=255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        pts_px = [((x - min_x) * mm_to_px + margin_px, (y - min_y) * mm_to_px + margin_px)
                  for x, y in stroke]
        draw.line(pts_px, fill=0, width=2, joint="curve")

    return img


if __name__ == "__main__":
    from handwriting_bot import HandwritingGenerator, check_page_bounds
    from handwriting_ebb import EBB_DEFAULTS

    cfg = dict(EBB_DEFAULTS)

    if ENGINE == "rnn":
        from handwriting_rnn import RNNHandwritingGenerator
        gen = RNNHandwritingGenerator(cfg, rng_seed=1, bias=0.75, style=9)

        if VERIFY:
            from handwriting_verify import HandwritingVerifier, VerifiedRNNHandwritingGenerator
            print("Loading handwriting-recognition model for verification "
                  "(first use only, may take a moment)...")
            verifier = HandwritingVerifier()
            gen = VerifiedRNNHandwritingGenerator(gen, verifier, similarity_threshold=0.55)
    else:
        gen = HandwritingGenerator(cfg, rng_seed=1)
        if VERIFY:
            print("Note: VERIFY only applies to ENGINE='rnn' -- ignoring for Hershey.")

    strokes = gen.generate(TEXT)

    flagged = getattr(gen, "flagged_lines", None)
    if flagged:
        print(f"\n{len(flagged)} line(s) did not pass verification even after retries:")
        for f in flagged:
            print(f"  expected {f['text']!r}, model read back {f['recognized']!r} "
                  f"(similarity {f['similarity']:.2f})")
        print()

    bounds = check_page_bounds(strokes, cfg)
    print(bounds["message"])
    if not bounds["fits"]:
        print("WARNING: this would not fit on the configured page size.")

    img = render_document_preview(strokes, cfg)
    img.save(OUTPUT_PNG)
    print(f"Saved {OUTPUT_PNG} ({img.size[0]}x{img.size[1]}px)")
    print(f"Engine used: {ENGINE}")