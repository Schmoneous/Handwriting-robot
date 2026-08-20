"""
rnn_handwriting_adapter.py
============================
Bridges the sjvasquez-style handwriting-synthesis RNN (Graves 2013) into
DrawCore's existing EBB / CoreXY pipeline.

DEPENDENCY SETUP
-----------------
The original repo (sjvasquez/handwriting-synthesis) requires TensorFlow 1.6
and uses tensorflow.contrib, which was removed in TF2. Use the TF2-migrated
fork instead:

    git clone https://github.com/otuva/handwriting-synthesis vendor/handwriting-synthesis
    cd vendor/handwriting-synthesis
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

Then this module imports it as:

    from handwriting_synthesis.hand import Hand

Add vendor/handwriting-synthesis to your PYTHONPATH, or pip install -e it
into your existing DrawCore venv (it has no console entrypoint, so an editable
install is fine and keeps everything in one environment).

WHY WE INTERCEPT BEFORE SVG
-----------------------------
Hand.write() internally does roughly:

    strokes = self._sample(lines, biases, styles)   # <-- raw pen data
    self._draw(strokes, lines, filename, ...)        # <-- renders SVG

`strokes` is a list (one array per line) of shape (T, 3) arrays:
    column 0 : pen-lift bit (1 = end of stroke / pen up, 0 = pen down/drawing)
    column 1 : dx  (relative x offset from previous point, model units)
    column 2 : dy  (relative y offset from previous point, model units)

That's structurally identical to what your Hershey pipeline already produces
(a sequence of moves + pen states) -- so instead of letting Hand.write() burn
cycles rendering SVG we don't need, we call the private sampler directly and
feed the raw strokes into our own coordinate normalizer + EBB generator.

CALIBRATION
------------
The RNN's coordinate units are arbitrary (an artifact of how the IAM dataset
was normalized during training), NOT millimeters. Before first use, run
`calibrate_scale()` once to establish model-units -> mm, the same way you
already calibrated steps_per_mm=80 by physical measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1. RNN backend wrapper
# ---------------------------------------------------------------------------

class RNNStrokeSource:
    """
    Thin wrapper around handwriting_synthesis.hand.Hand that returns raw
    stroke arrays instead of writing SVG files.
    """

    def __init__(self):
        # Imported lazily so this module can be imported (e.g. for the
        # coordinate-math / EBB parts) even in environments where the
        # RNN's TF2 dependency isn't installed yet.
        from handwriting_synthesis.hand import Hand
        self._hand = Hand()

    def sample_lines(
        self,
        lines: Sequence[str],
        biases: Optional[Sequence[float]] = None,
        styles: Optional[Sequence[int]] = None,
    ) -> List[np.ndarray]:
        """
        Returns a list of (T, 3) arrays, one per input line, in RAW model
        units: [pen_lift_bit, dx, dy] per row.

        bias:  0.0 (messy/loose) -> 1.0 (neat/controlled). Good "humanness" dial.
        style: integer index (0-12ish) selecting a learned handwriting style
               from the pretrained checkpoint.
        """
        if biases is None:
            biases = [0.75 for _ in lines]
        if styles is None:
            styles = [9 for _ in lines]

        # Private method, but this is the whole reason we're bypassing
        # Hand.write()'s SVG path. If a future fork renames this, the fix
        # is a one-line change here.
        strokes = self._hand._sample(lines, biases, styles)
        return strokes


# ---------------------------------------------------------------------------
# 2. Coordinate normalization: model units -> mm -> steps
# ---------------------------------------------------------------------------

@dataclass
class DrawCoreCalibration:
    """Mirrors your already-established hardware constants."""
    steps_per_mm: float = 80.0
    flip_x: bool = True
    reverse_line_direction: bool = True
    min_letter_advance_mm: float = 2.2
    pen_move_settle_ms: int = 500

    # RNN-specific calibration (you set this via calibrate_scale())
    model_units_per_mm: float = 1.0  # placeholder until calibrated


def calibrate_scale(
    stroke_source: RNNStrokeSource,
    target_line_height_mm: float,
    sample_text: str = "Hxgpqy",  # tall ascenders + descenders for full extent
) -> float:
    """
    Run once per style/bias you plan to use (line height varies slightly by
    style). Generates one sample line, measures its bounding-box height in
    model units, and returns model_units_per_mm.

    Usage:
        cal = DrawCoreCalibration()
        src = RNNStrokeSource()
        cal.model_units_per_mm = calibrate_scale(src, target_line_height_mm=8.0)
    """
    strokes = stroke_source.sample_lines([sample_text], biases=[0.75], styles=[9])
    pts = _cumulative_points(strokes[0])
    height_units = pts[:, 1].max() - pts[:, 1].min()
    if height_units <= 0:
        raise ValueError("Degenerate sample; try a longer sample_text.")
    return height_units / target_line_height_mm


def _cumulative_points(stroke: np.ndarray) -> np.ndarray:
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
    return xy


@dataclass
class PenPoint:
    x_mm: float
    y_mm: float
    pen_down: bool  # True = drawing, False = travel move


def strokes_to_pen_points(
    stroke: np.ndarray,
    cal: DrawCoreCalibration,
    origin_mm: Tuple[float, float] = (0.0, 0.0),
) -> List[PenPoint]:
    """
    Converts one raw RNN stroke array (one line of text) into a list of
    PenPoint in DrawCore's mm space, relative to origin_mm (typically your
    detected paper-corner offset from the vision system).
    """
    xy_units = _cumulative_points(stroke)
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

    return points


# ---------------------------------------------------------------------------
# 3. EBB command generation
# ---------------------------------------------------------------------------
# NOTE: Replace `mm_to_steps` / the SP,XM formatting below with your existing
# EBB serial-write functions if you already have equivalents -- this is
# written standalone so it can be dropped in and adapted, not to duplicate
# work you've already built and calibrated.

def mm_to_steps(mm: float, cal: DrawCoreCalibration) -> int:
    return round(mm * cal.steps_per_mm)


def pen_points_to_ebb_commands(
    points: Sequence[PenPoint],
    cal: DrawCoreCalibration,
    move_step_freq: int = 2000,
    pen_up_cmd: str = "SP,0",
    pen_down_cmd: str = "SP,1",
) -> List[str]:
    """
    Converts a sequence of PenPoints into EBB command strings.

    Uses absolute-style XM deltas (matches your existing convention of
    issuing relative stepper moves between consecutive points). Emits
    SP pen commands only on state transitions to avoid redundant serial
    writes, and honors reverse_line_direction by processing points in
    reverse if set.
    """
    if cal.reverse_line_direction:
        points = list(reversed(points))

    commands: List[str] = []
    last_x_steps = last_y_steps = 0
    last_pen_down: Optional[bool] = None

    for i, pt in enumerate(points):
        x_steps = mm_to_steps(pt.x_mm, cal)
        y_steps = mm_to_steps(pt.y_mm, cal)

        if last_pen_down is None:
            # First point: establish pen state before any move.
            commands.append(pen_down_cmd if pt.pen_down else pen_up_cmd)
        elif pt.pen_down != last_pen_down:
            commands.append(pen_down_cmd if pt.pen_down else pen_up_cmd)

        dx = x_steps - last_x_steps
        dy = y_steps - last_y_steps
        if dx != 0 or dy != 0:
            commands.append(f"XM,{move_step_freq},{dx},{dy}")

        last_x_steps, last_y_steps = x_steps, y_steps
        last_pen_down = pt.pen_down

    commands.append(pen_up_cmd)  # always end pen-up
    return commands


# ---------------------------------------------------------------------------
# 4. Hybrid writer: mix RNN and Hershey per line
# ---------------------------------------------------------------------------

@dataclass
class HybridWriter:
    """
    High-level entry point for the "combined" system. Each line of text can
    independently choose the RNN engine or your existing Hershey engine,
    letting you e.g. use the RNN for prose (max humanness) and Hershey for
    anything the RNN's training vocabulary handles poorly (unusual symbols,
    numerals in specific fonts, etc.) -- while both paths converge on the
    same EBB command stream.

    Wire `hershey_line_to_points` to whatever function your existing pipeline
    uses to turn a line of text into a list of PenPoint (mm space, same
    convention as strokes_to_pen_points above). That's the only piece of your
    current code this module needs a handle to.
    """
    calibration: DrawCoreCalibration
    hershey_line_to_points: Callable[[str], List[PenPoint]]
    rnn_source: RNNStrokeSource = field(default_factory=RNNStrokeSource)

    def render_line(
        self,
        text: str,
        engine: str = "rnn",          # "rnn" | "hershey"
        bias: float = 0.75,
        style: int = 9,
        origin_mm: Tuple[float, float] = (0.0, 0.0),
    ) -> List[PenPoint]:
        if engine == "hershey":
            return self.hershey_line_to_points(text)

        if engine == "rnn":
            strokes = self.rnn_source.sample_lines([text], biases=[bias], styles=[style])
            return strokes_to_pen_points(strokes[0], self.calibration, origin_mm=origin_mm)

        raise ValueError(f"Unknown engine: {engine!r}")

    def render_document(
        self,
        lines: Sequence[str],
        engine_per_line: Optional[Sequence[str]] = None,
        bias: float = 0.75,
        style: int = 9,
        line_height_mm: float = 8.0,
        left_margin_mm: float = 10.0,
        top_margin_mm: float = 10.0,
    ) -> List[str]:
        """
        Renders a full multi-line document and returns the flattened EBB
        command list, ready to stream to the board. Handles line-to-line
        vertical offset itself; horizontal origin resets to left_margin_mm
        each line (no auto word-wrap -- feed pre-wrapped lines, same as your
        current Hershey pipeline expects).
        """
        if engine_per_line is None:
            engine_per_line = ["rnn"] * len(lines)

        all_commands: List[str] = []
        for i, (text, engine) in enumerate(zip(lines, engine_per_line)):
            origin = (left_margin_mm, top_margin_mm + i * line_height_mm)
            points = self.render_line(text, engine=engine, bias=bias, style=style, origin_mm=origin)
            all_commands.extend(
                pen_points_to_ebb_commands(points, self.calibration)
            )
        return all_commands


# ---------------------------------------------------------------------------
# 5. Bias-derived jitter extraction (true hybrid: RNN dynamics -> Hershey jitter)
# ---------------------------------------------------------------------------

def extract_local_jitter_profile(stroke: np.ndarray, window: int = 5) -> np.ndarray:
    """
    A different flavor of 'hybrid': instead of choosing RNN OR Hershey per
    line, use the RNN's own stroke curvature as a *reference* for how your
    Perlin-noise jitter amplitude should vary along a stroke -- e.g. more
    jitter mid-stroke, less at stroke start/end, matching real hand tremor
    dynamics instead of uniform noise.

    Returns a per-point curvature magnitude array (same length as stroke),
    normalized 0-1, which you can multiply into your existing Perlin jitter
    amplitude before generating Hershey coordinates. This lets you keep your
    full Hershey/EMS-Casual-Hand font pipeline exactly as-is while making the
    jitter itself less uniform/synthetic-feeling.
    """
    pts = _cumulative_points(stroke)
    if len(pts) < 3:
        return np.zeros(len(pts))

    # Discrete curvature approximation via angle change between consecutive
    # segment vectors.
    curvature = np.zeros(len(pts))
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        curvature[i] = math.acos(cos_angle)

    # Smooth with a simple moving average, then normalize.
    kernel = np.ones(window) / window
    smoothed = np.convolve(curvature, kernel, mode="same")
    max_val = smoothed.max()
    return smoothed / max_val if max_val > 0 else smoothed


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cal = DrawCoreCalibration()
    src = RNNStrokeSource()

    # One-time calibration for an 8mm line height at style 9 / bias 0.75.
    cal.model_units_per_mm = calibrate_scale(src, target_line_height_mm=8.0)
    print(f"Calibrated: {cal.model_units_per_mm:.3f} model units per mm")

    # Direct single-line render (RNN only).
    strokes = src.sample_lines(["Testing DrawCore RNN integration"], biases=[0.75], styles=[9])
    points = strokes_to_pen_points(strokes[0], cal, origin_mm=(10.0, 10.0))
    commands = pen_points_to_ebb_commands(points, cal)
    print(f"Generated {len(commands)} EBB commands for line 1")

    # Hybrid document example (requires wiring your real Hershey function in):
    #
    # def my_hershey_line_to_points(text: str) -> List[PenPoint]:
    #     ...  # your existing Hershey + Perlin pipeline, returning PenPoints
    #
    # writer = HybridWriter(calibration=cal, hershey_line_to_points=my_hershey_line_to_points)
    # cmds = writer.render_document(
    #     lines=["Dear Mr. Hanser,", "Progress update attached below.", "42.7 units measured"],
    #     engine_per_line=["rnn", "rnn", "hershey"],  # numerals -> Hershey, prose -> RNN
    # )
