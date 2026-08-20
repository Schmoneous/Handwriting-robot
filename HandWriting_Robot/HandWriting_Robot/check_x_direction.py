from handwriting_bot import HandwritingGenerator, check_page_bounds
from handwriting_ebb import EBB_DEFAULTS
from handwriting_rnn import RNNHandwritingGenerator

cfg = dict(EBB_DEFAULTS)
gen = RNNHandwritingGenerator(cfg, rng_seed=1, bias=0.75, style=9)

line = "hello world"
strokes = gen._line_to_strokes(line, x_start=10.0, y_baseline=20.0)

# Flatten all points in stroke order (the order they'd actually be drawn)
all_pts = [p for s in strokes for p in s]
first_x = all_pts[0][0]
last_x = all_pts[-1][0]

print(f"Line: {line!r}")
print(f"First point x = {first_x:.2f}mm")
print(f"Last point x  = {last_x:.2f}mm")
print()
if last_x > first_x:
    print("x INCREASES left-to-right through the stroke sequence -- CORRECT direction.")
else:
    print("x DECREASES left-to-right through the stroke sequence -- MIRRORED. "
          "This confirms a real sign bug in _line_to_strokes, not a rendering issue.")

# Also check: does x ever go strongly negative relative to x_start, which
# would indicate the pen is moving backward from the start point rather
# than forward through the word?
print(f"\nx range across whole line: {min(p[0] for p in all_pts):.2f} to {max(p[0] for p in all_pts):.2f}")
print(f"x_start was: 10.00")
