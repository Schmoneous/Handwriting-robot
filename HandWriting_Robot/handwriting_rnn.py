#!/usr/bin/env python3
"""
handwriting_rnn.py
====================
RNN-based (Graves 2013 handwriting-synthesis) alternative to
HandwritingGenerator in handwriting_bot.py. Produces the exact same output
format -- a list of strokes, each a list of (x_mm, y_mm) points in final
page coordinates -- so it's a drop-in replacement anywhere HandwritingGenerator
is used. strokes_to_ebb_commands(), check_page_bounds(), and stream_ebb() in
handwriting_ebb.py need ZERO changes to support this: they only care about
the strokes format, not which engine produced it.

DEPENDENCY SETUP
-----------------
Requires the TF2-migrated fork (the original needs TF 1.6, incompatible
with modern environments):

    git clone https://github.com/otuva/handwriting-synthesis
    cd handwriting-synthesis
    python3.11 -m venv venv   # TF 2.13 has no wheels for Python 3.13
    source venv/bin/activate
    # edit requirements.txt: tensorflow==2.12.0 -> tensorflow==2.13.0
    # (2.12.0 has no macOS arm64 wheel on PyPI)
    pip install -r requirements.txt

Then copy this file (and rnn_handwriting_adapter's RNNStrokeSource logic,
now inlined below) into your DrawCore project directory alongside
handwriting_bot.py / handwriting_ebb.py / handwriting_gui.py, and make sure
the handwriting-synthesis repo root is on PYTHONPATH (or pip install -e it
into the SAME venv you run the GUI from -- everything needs to share one
Python environment since handwriting_bot.py's deps and the RNN's TF deps
both need to be importable together).

STROKE FORMAT NOTE (found empirically -- see conversation history)
---------------------------------------------------------------------
This fork's Hand._sample() returns raw arrays of shape (T, 3) per line,
with columns [dx, dy, pen_lift_bit] -- NOT [pen_lift_bit, dx, dy] like the
original Graves paper reference implementation. If you swap to a different
fork later, re-verify this column order before trusting output (print
strokes[0][:10] and confirm which column is binary 0.0/1.0 -- that's the
pen-lift bit).
"""

from __future__ import annotations

import contextlib
import os
import random
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Makes `import handwriting_synthesis` work regardless of the current working
# directory the GUI (or any script importing this module) was launched from --
# previously this only worked if you happened to be cd'd into the cloned
# handwriting-synthesis repo itself.
#
# Set this once in your shell profile (~/.zshrc):
#     export HANDWRITING_SYNTHESIS_PATH=/full/path/to/handwriting-synthesis
#
# or hardcode the fallback default below to your actual clone location.
_HANDWRITING_SYNTHESIS_PATH = os.environ.get(
    "HANDWRITING_SYNTHESIS_PATH",
    "",  # <-- optionally hardcode your absolute path here as a fallback, e.g.
         #     "/Users/ivebens_eliacin/Downloads/HandWriting_Robot/handwriting-synthesis"
)
if _HANDWRITING_SYNTHESIS_PATH and _HANDWRITING_SYNTHESIS_PATH not in sys.path:
    if not os.path.isdir(_HANDWRITING_SYNTHESIS_PATH):
        print(f"[handwriting_rnn] WARNING: HANDWRITING_SYNTHESIS_PATH is set to "
              f"'{_HANDWRITING_SYNTHESIS_PATH}' but that directory doesn't exist. "
              f"RNN engine imports will fail until this is fixed.")
    sys.path.insert(0, _HANDWRITING_SYNTHESIS_PATH)


@contextlib.contextmanager
def _repo_cwd():
    """
    Temporarily changes the working directory to the handwriting-synthesis
    repo root, then restores the original cwd afterward -- even if an
    exception is raised inside the block.

    This fork resolves MULTIPLE things via paths relative to cwd, not just
    at model-construction time:
      - the pretrained checkpoint (loaded once, in Hand.__init__)
      - per-style stroke/character .npy files (loaded lazily, at sample
        time -- e.g. model/style/style-9-strokes.npy)
    So every call into the model -- construction AND sampling -- needs to
    happen inside this context, not just __init__. Wrapping only __init__
    fixes checkpoint loading but leaves later _sample() calls broken with a
    similar "file not found" error for style-specific files.
    """
    repo_root = _HANDWRITING_SYNTHESIS_PATH
    if repo_root and os.path.isdir(repo_root):
        original_cwd = os.getcwd()
        os.chdir(repo_root)
        try:
            yield
        finally:
            os.chdir(original_cwd)
    else:
        # No path configured -- fall back to hoping cwd is already correct.
        yield


# Hard ceiling confirmed directly from the installed model's source:
# handwriting_synthesis/hand/Hand.py has `chars = np.zeros([num_samples, 120])`
# -- a fixed-size buffer with NO room for anything longer, regardless of
# physical page width or font_size_mm. Exceeding it crashes with
# "ValueError: could not broadcast input array from shape (N,) into shape
# (120,)" deep inside the model's _sample() method. wrap_text() previously
# only checked physical page-width fit, which could pack MORE than 120
# characters onto one line if font_size_mm was small enough to make many
# characters "fit" the configured page width -- this cap prevents that
# regardless of font size. A small safety margin (10 chars) is kept below
# the hard 120 limit in case of any off-by-one in the model's own internal
# handling (e.g. an implicit end-of-sequence marker).
RNN_MAX_CHARS_PER_LINE = 120
RNN_SAFE_MAX_CHARS_PER_LINE = 110


class RNNStrokeSource:
    """Thin wrapper around handwriting_synthesis.hand.Hand that returns raw
    (T, 3) stroke arrays [dx, dy, pen_lift_bit] instead of writing SVG."""

    def __init__(self):
        from handwriting_synthesis.hand import Hand
        with _repo_cwd():
            self._hand = Hand()

    def sample_lines(
        self,
        lines: Sequence[str],
        biases: Optional[Sequence[float]] = None,
        styles: Optional[Sequence[int]] = None,
    ) -> List[np.ndarray]:
        if biases is None:
            biases = [0.75 for _ in lines]
        if styles is None:
            styles = [9 for _ in lines]
        with _repo_cwd():
            return self._hand._sample(lines, biases, styles)


def _calibrate_scale(
    source: RNNStrokeSource,
    target_line_height_mm: float,
    bias: float,
    style: int,
    sample_text: str = "Hxgpqy",
) -> float:
    """
    Returns model_units_per_mm for the given bias/style (line height varies
    slightly by learned style, so calibration is cached per bias/style pair
    by the caller, not globally).
    """
    strokes = source.sample_lines([sample_text], biases=[bias], styles=[style])
    xy = np.cumsum(strokes[0][:, 0:2], axis=0)
    height_units = xy[:, 1].max() - xy[:, 1].min()
    if height_units <= 0:
        raise ValueError("Degenerate calibration sample; try different sample_text.")
    return height_units / target_line_height_mm


def _calibrate_char_width(
    source: RNNStrokeSource,
    scale: float,
    bias: float,
    style: int,
    sample_text: str = "the quick brown fox jumps over the lazy dog",
) -> float:
    """
    Returns average mm-per-character for this bias/style, measured directly
    from real generated stroke width -- NOT a guessed ratio. This matters
    because RNN's continuous cursive strokes are noticeably more compact
    per character than Hershey's discrete glyphs; reusing Hershey's
    font_size_mm * 0.55 wrap-width heuristic caused lines to wrap far too
    early (confirmed: real output was wrapping at ~73.5mm instead of using
    the full configured page width), since that heuristic assumed each
    character takes up more horizontal space than the RNN actually uses.

    sample_text intentionally avoids Q/X/Z (which get lowercased anyway --
    see _sanitize_for_rnn) and mixes common letters/spaces for a
    representative average.
    """
    strokes = source.sample_lines([sample_text], biases=[bias], styles=[style])
    xy = np.cumsum(strokes[0][:, 0:2], axis=0)
    width_units = xy[:, 0].max() - xy[:, 0].min()
    width_mm = width_units / scale
    return width_mm / len(sample_text)


class RNNHandwritingGenerator:
    """
    Drop-in alternative to handwriting_bot.HandwritingGenerator. Same
    constructor shape (cfg, rng_seed), same .generate(text) contract:
    returns a list of strokes, each a list of (x_mm, y_mm) tuples, already
    positioned in final page coordinates (respecting cfg's x_offset_mm,
    y_offset_mm, page_width_mm wrapping, line_spacing_mm, and
    reverse_line_direction) -- identical to what HandwritingGenerator.generate()
    produces, so it can be substituted anywhere in handwriting_gui.py or
    handwriting_ebb.py without those files needing to know the difference.

    NOTE: generate_blocks() (structured documents with headings/bullets/
    numbered lists) IS implemented -- see below -- mirroring
    HandwritingGenerator.generate_blocks()'s layout logic exactly, so
    switching Engine between Hershey and RNN preserves document structure
    either way.
    """

    def __init__(self, cfg, rng_seed: Optional[int] = None, bias: float = 0.75, style: int = 9):
        self.cfg = cfg
        self.rng = random.Random(rng_seed)
        self.bias = bias
        self.style = style
        self._source = RNNStrokeSource()
        self._scale_cache = {}  # (bias, style) -> model_units_per_mm
        self._char_width_cache = {}  # (bias, style) -> mm_per_char
        self._supported_chars = self._load_supported_alphabet()

    def _load_supported_alphabet(self) -> set:
        """
        Reads the model's actual supported character set directly from the
        installed package, rather than hardcoding it here -- so if you ever
        swap checkpoints/forks with a different trained alphabet, this stays
        correct automatically instead of silently going stale.
        """
        with _repo_cwd():
            from handwriting_synthesis.drawing import operations
        return set(operations.alphabet)

    def _sanitize_for_rnn(self, text: str) -> str:
        """
        The model's trained alphabet is missing a small number of characters
        (confirmed: uppercase Q, X, Z aren't in the training vocabulary,
        even though their lowercase forms are). Unsupported characters
        silently map to a fallback token inside the model rather than
        erroring, which would draw the WRONG letter with no warning -- so we
        fix this upstream instead: any unsupported uppercase letter whose
        lowercase form IS supported gets lowercased before generation. This
        keeps everything in one consistent "hand" (no fallback engine, no
        visual mismatch), at the cost of losing capitalization specifically
        for those rare letters (e.g. "Quiz" -> "quiz").

        Any character that's unsupported AND has no usable lowercase fallback
        is left as-is and will still hit the model's default substitution
        behavior -- this covers the two known real cases without silently
        mishandling some future unexpected character.
        """
        out = []
        for ch in text:
            if ch in self._supported_chars:
                out.append(ch)
            elif ch.lower() in self._supported_chars and ch != ch.lower():
                out.append(ch.lower())
            else:
                out.append(ch)
        return "".join(out)

    def _get_scale(self) -> float:
        key = (self.bias, self.style)
        if key not in self._scale_cache:
            self._scale_cache[key] = _calibrate_scale(
                self._source,
                target_line_height_mm=self.cfg["font_size_mm"],
                bias=self.bias,
                style=self.style,
            )
        return self._scale_cache[key]

    def _get_mm_per_char(self) -> float:
        key = (self.bias, self.style)
        if key not in self._char_width_cache:
            self._char_width_cache[key] = _calibrate_char_width(
                self._source,
                scale=self._get_scale(),
                bias=self.bias,
                style=self.style,
            )
        return self._char_width_cache[key]

    def wrap_text(self, text: str, width_mm: Optional[float] = None, font_scale: float = 1.0) -> List[str]:
        """
        Greedy word-wrap using the RNN's REAL measured average character
        width (see _calibrate_char_width), not a ratio borrowed from the
        Hershey pipeline. The old font_size_mm * 0.55 heuristic assumed
        characters were wider than the RNN actually draws them, so lines
        were wrapping far too early (confirmed: ~73.5mm instead of using
        the full page width) -- this fixes that by measuring the real
        thing instead of guessing.

        Also enforces RNN_SAFE_MAX_CHARS_PER_LINE regardless of physical
        page width fit -- the model has a hard 120-character buffer (see
        RNN_MAX_CHARS_PER_LINE above) that has nothing to do with font
        size or page width, and a small font_size_mm can otherwise pack
        more characters onto one "physically fitting" line than the model
        can actually accept, crashing generation entirely.

        font_scale: same convention as HandwritingGenerator.wrap_text --
        pass a value >1.0 for headings (wider characters at a larger size
        need fewer characters per line to fit the same physical width).
        """
        cfg = self.cfg
        width_mm = width_mm if width_mm is not None else cfg["page_width_mm"]
        mm_per_char = self._get_mm_per_char() * font_scale
        width_based_guess = max(6, int(width_mm / mm_per_char))
        max_chars_guess = min(width_based_guess, RNN_SAFE_MAX_CHARS_PER_LINE)

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

        # Safety net: the word-based loop above only breaks BETWEEN words,
        # so a single word (or a bullet/numbered marker + word) longer than
        # max_chars_guess on its own would still slip through over the
        # limit. Force-split anything that's still too long -- this is a
        # last-resort hard character split, not word-aware, but it
        # guarantees the model's 120-char buffer is never exceeded no
        # matter what.
        safe_lines = []
        for line in lines:
            if len(line) <= RNN_SAFE_MAX_CHARS_PER_LINE:
                safe_lines.append(line)
            else:
                for i in range(0, len(line), RNN_SAFE_MAX_CHARS_PER_LINE):
                    safe_lines.append(line[i:i + RNN_SAFE_MAX_CHARS_PER_LINE])

        return safe_lines

    def _line_to_strokes(
        self, line_text: str, x_start: float, y_baseline: float, font_scale: float = 1.0
    ) -> List[List[Tuple[float, float]]]:
        """
        Generates one line of RNN handwriting and splits it into individual
        pen-down segments (matching handwriting_bot.py's stroke convention:
        each returned stroke is one continuous pen-down path, with implicit
        pen-up travel between list entries).

        font_scale: renders this line at font_size_mm * font_scale instead
        of the base size, used for headings. Implemented by dividing the
        base calibrated scale by font_scale -- the model always generates
        the same underlying stroke geometry regardless of target size, so
        scaling the mm-conversion factor is equivalent to (and cheaper
        than) recalibrating from scratch for every heading level.
        """
        if not line_text.strip():
            return []

        effective_scale = self._get_scale() / font_scale
        sanitized = self._sanitize_for_rnn(line_text)

        # Last-resort defensive guard, regardless of how this method was
        # reached (wrap_text's cap, generate_blocks' marker concatenation,
        # or any other caller). This has repeatedly crashed in practice
        # with "could not broadcast input array from shape (N,) into shape
        # (120,)" from unexpected code paths that were hard to pin down
        # exactly -- rather than keep chasing every possible source, this
        # guarantees it can never happen again, while still surfacing a
        # clear, traceable warning (with the actual offending text) instead
        # of a cryptic numpy error deep inside the model.
        if len(sanitized) > RNN_SAFE_MAX_CHARS_PER_LINE:
            print(f"[handwriting_rnn] WARNING: line was {len(sanitized)} chars "
                  f"(limit {RNN_SAFE_MAX_CHARS_PER_LINE}), truncating. This means "
                  f"something upstream of _line_to_strokes let an oversized line "
                  f"through -- original text: {line_text!r}")
            sanitized = sanitized[:RNN_SAFE_MAX_CHARS_PER_LINE]

        raw = self._source.sample_lines([sanitized], biases=[self.bias], styles=[self.style])[0]

        segments: List[List[Tuple[float, float]]] = []
        current: List[Tuple[float, float]] = []
        cum_x = cum_y = 0.0

        for row in raw:
            dx, dy, lift = float(row[0]), float(row[1]), float(row[2])
            cum_x += dx
            cum_y += dy
            current.append((cum_x, cum_y))
            if lift > 0.5:
                if len(current) >= 2:
                    segments.append(current)
                current = []
        if len(current) >= 2:
            segments.append(current)

        strokes = []
        for seg in segments:
            pts_mm = [(x_start + ux / effective_scale, y_baseline + uy / effective_scale) for ux, uy in seg]
            strokes.append(pts_mm)
        return strokes

    def _jittered_x_start(self, base_x: float) -> float:
        """
        Returns base_x plus a small random +/- offset (line_start_x_jitter_mm),
        so successive lines don't all begin at the exact same horizontal
        position -- real handwriting has natural left-margin inconsistency
        between lines/sentences that a perfectly fixed x_start doesn't capture.
        Uses self.rng (seeded), so this is reproducible given the same seed.
        """
        jitter = self.cfg.get("line_start_x_jitter_mm", 0.0)
        if jitter <= 0:
            return base_x
        return base_x + self.rng.uniform(-jitter, jitter)

    def generate(self, text: str) -> List[List[Tuple[float, float]]]:
        cfg = self.cfg
        strokes_out: List[List[Tuple[float, float]]] = []
        lines = self.wrap_text(text)
        y_cursor = cfg["y_offset_mm"]
        line_dir = -1 if cfg.get("reverse_line_direction") else 1
        for line in lines:
            x_start = self._jittered_x_start(cfg["x_offset_mm"])
            strokes_out.extend(self._line_to_strokes(line, x_start, y_cursor))
            y_cursor += line_dir * cfg["line_spacing_mm"]
        return strokes_out

    def generate_blocks(self, blocks) -> List[List[Tuple[float, float]]]:
        """
        Structured-document counterpart to generate(), mirroring
        HandwritingGenerator.generate_blocks() exactly -- same block types
        (heading/bullet/numbered/paragraph/blank), same heading scale
        factors and indent amount, so switching Engine between Hershey and
        RNN produces a document with the same structural layout, just
        drawn by a different hand.

        blocks: as produced by text_extractor.extract_blocks() --
        [{"type": "heading"|"bullet"|"numbered"|"paragraph"|"blank",
          "text": str, "level": int}, ...]
        """
        cfg = self.cfg
        strokes_out: List[List[Tuple[float, float]]] = []
        y_cursor = cfg["y_offset_mm"]
        numbered_counter = 0
        line_dir = -1 if cfg.get("reverse_line_direction") else 1

        # Kept identical to HandwritingGenerator.generate_blocks()'s constants
        # so both engines lay out headings/indents the same way. If you ever
        # change these in handwriting_bot.py, update here too.
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
                    y_cursor += line_dir * cfg["line_spacing_mm"] * 0.4
                lines = self.wrap_text(block["text"], font_scale=scale)
                for line in lines:
                    x_start = self._jittered_x_start(cfg["x_offset_mm"])
                    strokes_out.extend(
                        self._line_to_strokes(line, x_start, y_cursor, font_scale=scale))
                    y_cursor += line_dir * cfg["line_spacing_mm"] * scale
                y_cursor += line_dir * cfg["line_spacing_mm"] * 0.3

            elif btype == "bullet":
                x_indent = cfg["x_offset_mm"] + INDENT_MM
                marker = "- "
                lines = self.wrap_text(marker + block["text"], width_mm=cfg["page_width_mm"] - INDENT_MM)
                for i, line in enumerate(lines):
                    base_x = x_indent if i == 0 else x_indent + INDENT_MM * 0.6
                    x_start = self._jittered_x_start(base_x)
                    strokes_out.extend(self._line_to_strokes(line, x_start, y_cursor))
                    y_cursor += line_dir * cfg["line_spacing_mm"]

            elif btype == "numbered":
                numbered_counter += 1
                x_indent = cfg["x_offset_mm"] + INDENT_MM
                marker = f"{numbered_counter}. "
                lines = self.wrap_text(marker + block["text"], width_mm=cfg["page_width_mm"] - INDENT_MM)
                for i, line in enumerate(lines):
                    base_x = x_indent if i == 0 else x_indent + INDENT_MM * 0.6
                    x_start = self._jittered_x_start(base_x)
                    strokes_out.extend(self._line_to_strokes(line, x_start, y_cursor))
                    y_cursor += line_dir * cfg["line_spacing_mm"]

            else:  # paragraph
                lines = self.wrap_text(block["text"])
                for line in lines:
                    x_start = self._jittered_x_start(cfg["x_offset_mm"])
                    strokes_out.extend(self._line_to_strokes(line, x_start, y_cursor))
                    y_cursor += line_dir * cfg["line_spacing_mm"]

            prev_type = btype

        return strokes_out


# ---------------------------------------------------------------------------
# Standalone smoke test (safe to run without touching the GUI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from handwriting_bot import DEFAULTS
    from handwriting_ebb import EBB_DEFAULTS, strokes_to_ebb_commands
    from handwriting_bot import check_page_bounds

    cfg = dict(EBB_DEFAULTS)
    gen = RNNHandwritingGenerator(cfg, rng_seed=42, bias=0.75, style=9)
    strokes = gen.generate("Testing DrawCore RNN integration")

    bounds = check_page_bounds(strokes, cfg)
    print(bounds["message"])

    commands = strokes_to_ebb_commands(strokes, cfg, rng_seed=42)
    print(f"Generated {len(strokes)} strokes, {len(commands)} EBB commands")
    print("First 10 commands:")
    for c in commands[:10]:
        print(" ", c)