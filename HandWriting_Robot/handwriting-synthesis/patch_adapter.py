"""
Run this once from inside your handwriting-synthesis repo directory
(same folder as rnn_handwriting_adapter.py) to fix the column-order bug:

    python patch_adapter.py

It rewrites the two functions that had the wrong assumption about the
stroke array's column layout, based on the diagnostic output confirming
this fork uses [dx, dy, pen_lift_bit] instead of [pen_lift_bit, dx, dy].
"""

import re

PATH = "rnn_handwriting_adapter.py"

with open(PATH, "r") as f:
    content = f.read()

old_cumulative = '''def _cumulative_points(stroke: np.ndarray) -> np.ndarray:
    """
    Converts a (T, 3) [pen_lift, dx, dy] array into absolute (x, y) points
    via cumulative sum. Returns (T, 2) array.
    """
    xy = np.cumsum(stroke[:, 1:3], axis=0)
    return xy'''

new_cumulative = '''def _cumulative_points(stroke: np.ndarray) -> np.ndarray:
    """
    Converts a (T, 3) [dx, dy, pen_lift] array into absolute (x, y) points
    via cumulative sum. Returns (T, 2) array.

    NOTE: column layout confirmed empirically against this fork's actual
    _sample() output. This fork uses [dx, dy, pen_lift_bit] -- NOT the
    [pen_lift_bit, dx, dy] order used by the original Graves paper's
    reference implementation. Different forks apparently reordered these
    columns, so if you swap to a different fork later, re-run the
    diagnostic (print stroke[:10], check which column is binary 0/1)
    before assuming this layout still holds.
    """
    xy = np.cumsum(stroke[:, 0:2], axis=0)
    return xy'''

old_strokes_loop = '''    xy_units = _cumulative_points(stroke)
    xy_mm = xy_units / cal.model_units_per_mm

    ox, oy = origin_mm
    points: List[PenPoint] = []

    pen_down = True  # RNN strokes start pen-down by convention
    for row, (x, y) in zip(stroke, xy_mm):
        lift_bit = row[0]
        px = x + ox
        py = y + oy
        if cal.flip_x:
            px = -px
        points.append(PenPoint(x_mm=px, y_mm=py, pen_down=pen_down))
        # After a point flagged with lift_bit==1, the NEXT point begins a
        # new stroke with the pen lifted for the travel move, then back down.
        pen_down = lift_bit < 0.5

    return points'''

new_strokes_loop = '''    xy_units = _cumulative_points(stroke)
    xy_mm = xy_units / cal.model_units_per_mm

    ox, oy = origin_mm
    points: List[PenPoint] = []

    pen_down = True  # RNN strokes start pen-down by convention
    for row, (x, y) in zip(stroke, xy_mm):
        lift_bit = row[2]  # column 2, not column 0 -- see _cumulative_points note
        px = x + ox
        py = y + oy
        if cal.flip_x:
            px = -px
        points.append(PenPoint(x_mm=px, y_mm=py, pen_down=pen_down))
        # After a point flagged with lift_bit==1, the NEXT point begins a
        # new stroke with the pen lifted for the travel move, then back down.
        pen_down = lift_bit < 0.5

    return points'''

if old_cumulative not in content:
    raise SystemExit("ERROR: could not find _cumulative_points block to patch -- file may already be patched or modified.")
if old_strokes_loop not in content:
    raise SystemExit("ERROR: could not find strokes_to_pen_points loop to patch -- file may already be patched or modified.")

content = content.replace(old_cumulative, new_cumulative)
content = content.replace(old_strokes_loop, new_strokes_loop)

with open(PATH, "w") as f:
    f.write(content)

print("Patched successfully:")
print("  - _cumulative_points now sums columns [0:2] instead of [1:3]")
print("  - strokes_to_pen_points now reads pen_lift bit from column 2 instead of column 0")