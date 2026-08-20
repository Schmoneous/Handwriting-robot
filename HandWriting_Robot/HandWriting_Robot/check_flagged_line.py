import matplotlib.pyplot as plt
from handwriting_rnn import RNNHandwritingGenerator

cfg = dict(
    font_size_mm=8.0, page_width_mm=180.0, page_height_mm=250.0,
    x_offset_mm=10.0, y_offset_mm=15.0, line_spacing_mm=10.0,
    reverse_line_direction=False,
)

gen = RNNHandwritingGenerator(cfg, rng_seed=1, bias=0.75, style=9)

# One of the actual flagged lines from the real run
line = "The difference between a DFA and an NFA is that a"
strokes = gen._line_to_strokes(line, x_start=10.0, y_baseline=20.0)

fig, ax = plt.subplots(figsize=(14, 3))
for stroke in strokes:
    xs = [p[0] for p in stroke]
    ys = [p[1] for p in stroke]
    ax.plot(xs, ys, 'k-', linewidth=1.5)

ax.invert_yaxis()
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f"RNN output for: {line!r}")

plt.savefig('flagged_line_check.png', dpi=150, bbox_inches='tight')
print("Saved flagged_line_check.png -- does this actually look like readable cursive?")
