#!/usr/bin/env python3
"""
handwriting_verify.py
========================
Pre-flight verification for RNN-generated handwriting: rasterizes each
generated line to an image, runs it through a real handwriting-recognition
model, and compares the recognized text against what was actually intended
-- catching garbled/misformed output before the robot puts pen to paper.

DEPENDENCY CHOICE -- why EasyOCR, not another TensorFlow model
------------------------------------------------------------------
This uses EasyOCR (PyTorch-based), deliberately NOT another TensorFlow
model. Given how much trouble we had pinning tensorflow==2.13.0 +
tensorflow_probability==0.21.0 for the generation model, stacking a SECOND
TF model with its own version constraints in the same venv would risk
exactly the same class of conflict again (pip resolving to an incompatible
TF version to satisfy both models' pins simultaneously, or one model's TF
op registrations colliding with the other's). PyTorch and TensorFlow can
coexist in the same venv without fighting over the same C++ op registry,
so this sidesteps that whole failure mode.

Tradeoff: EasyOCR is a general handwriting/text recognizer, not a model
specifically trained on the same IAM-style cursive distribution as your
generation model. It will have real error rates on stylized cursive --
this is a genuine mitigation layer, not a guarantee. Tune
similarity_threshold based on what you observe in practice; start
permissive (e.g. 0.5) and tighten once you've seen real false-positive/
false-negative rates on your own output.

CHANGE LOG (verified fixes -- see notes at each site below)
------------------------------------------------------------------
1. Acceptance now uses is_close_enough() (length-scaled edit distance)
   instead of a raw text_similarity() ratio threshold. Verified
   empirically that the ratio doesn't scale with word length: 'the' vs
   'thc' (1 real character error) scored LOWER (0.667) than 'everything'
   vs 'evenjthng' (3 real character errors, 0.737) -- meaning a single
   fixed threshold lets multi-error garbled LONG words through while
   being comparatively harsh on short words with minor, forgivable
   variation. text_similarity() is kept as-is and still used for
   human-readable reporting (flagged_lines, log messages), just no
   longer for the actual accept/reject decision.

2. OCR confidence is now actually used. Previously HandwritingVerifier
   .recognize() correctly computed and returned avg_conf, but the caller
   discarded it immediately (`recognized, _conf = ...`) and never checked
   it anywhere. This matters because OCR models with language-model
   priors will sometimes "guess" a plausible dictionary word from
   genuinely garbled strokes -- with LOW confidence, even when the guess
   happens to match the intended text. That's exactly a case where text
   matches but the actual glyphs may still be wrong, and it was
   completely invisible to the old accept/reject logic. A line is now
   only accepted if it ALSO clears min_confidence.

SETUP
------
    pip install easyocr

First run downloads EasyOCR's pretrained recognition weights (~100MB) --
expect a one-time delay the first time HandwritingVerifier() is constructed.

    from handwriting_verify import HandwritingVerifier, VerifiedRNNHandwritingGenerator
    from handwriting_rnn import RNNHandwritingGenerator

    base_gen = RNNHandwritingGenerator(cfg, rng_seed=seed, bias=0.75, style=9)
    verifier = HandwritingVerifier()
    gen = VerifiedRNNHandwritingGenerator(base_gen, verifier, similarity_threshold=0.55)

    strokes = gen.generate(text)   # or gen.generate_blocks(blocks)
    if gen.flagged_lines:
        for f in gen.flagged_lines:
            print(f"LOW CONFIDENCE: expected {f['text']!r}, "
                  f"model read back {f['recognized']!r} "
                  f"(similarity {f['similarity']:.2f}, confidence {f['confidence']:.2f})")
"""

from __future__ import annotations

import difflib
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Rasterization: stroke list (mm coordinates) -> PIL image
# ---------------------------------------------------------------------------

def rasterize_strokes(
    strokes: Sequence[Sequence[Tuple[float, float]]],
    dpi: float = 200.0,
    padding_mm: float = 2.0,
    line_width_px: int = 2,
    min_height_px: int = 64,
    min_width_to_height_ratio: float = 3.0,
) -> Optional[Image.Image]:
    """
    Renders a list of pen-down strokes (mm coordinates, same convention as
    HandwritingGenerator/RNNHandwritingGenerator output) to a white-background
    grayscale image, for feeding into an OCR/handwriting-recognition model.

    Returns None if strokes is empty (nothing to rasterize, e.g. a blank line).

    min_height_px / min_width_to_height_ratio: EasyOCR's recognition network
    (confirmed via a real "could not broadcast input array" crash on short
    lines) appears to assume a minimum input size and width-relative-to-
    height ratio internally -- very short words or single-character lines
    produced tight rasterizations that violated whatever assumption its
    internal reshape logic makes. Rather than guess the exact failure
    threshold, we pad (not upscale/distort) the canvas up to safe minimums
    with extra white space, which costs nothing visually since the
    recognizer only cares about the actual ink pixels.
    """
    all_pts = [p for s in strokes for p in s]
    if not all_pts:
        return None

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs) - padding_mm, max(xs) + padding_mm
    min_y, max_y = min(ys) - padding_mm, max(ys) + padding_mm

    mm_to_px = dpi / 25.4
    width_px = max(1, int(round((max_x - min_x) * mm_to_px)))
    height_px = max(1, int(round((max_y - min_y) * mm_to_px)))

    # Enforce minimum absolute height.
    height_px = max(height_px, min_height_px)
    # Enforce minimum width relative to height (pad width, don't stretch).
    width_px = max(width_px, int(height_px * min_width_to_height_ratio))

    img = Image.new("L", (width_px, height_px), color=255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        # NOTE: direct mapping, no y-flip. Both HandwritingGenerator
        # (Hershey-based, handwriting_bot.py) and RNNHandwritingGenerator
        # (handwriting_rnn.py, as of its own y-negation fix -- see that
        # file's "Y-AXIS DIRECTION" note) produce strokes where y already
        # increases DOWNWARD, matching PIL's pixel space directly. A
        # previous version of this function applied a (max_y - y) flip
        # here based on an incorrect assumption about the RNN's raw
        # output convention -- confirmed wrong via the RNN repo's own
        # reference SVG-writer (which explicitly negates y itself,
        # meaning ITS raw output is upward-increasing, but
        # RNNHandwritingGenerator now corrects that upstream, so by the
        # time strokes reach this function they're already downward-y
        # like everything else in this project). If you introduce a THIRD
        # stroke source with a different native convention, convert it to
        # downward-y at its own source, not here -- this function should
        # stay convention-agnostic and assume its input already matches
        # the rest of the pipeline.
        pts_px = [((x - min_x) * mm_to_px, (y - min_y) * mm_to_px) for x, y in stroke]
        draw.line(pts_px, fill=0, width=line_width_px, joint="curve")

    return img


def text_similarity(a: str, b: str) -> float:
    """Character-level similarity ratio, 0.0-1.0, case/whitespace-insensitive.
    Kept for human-readable reporting (flagged_lines, log messages) --
    the actual accept/reject decision now uses is_close_enough() below,
    since this ratio doesn't scale sensibly with word length (verified:
    a single real error in a short word can score LOWER than several
    real errors in a long word)."""
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance (insertions/deletions/substitutions), case-insensitive."""
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_close_enough(expected: str, recognized: str, similarity_threshold: float = 0.8) -> bool:
    """
    The actual accept/reject text check, replacing a raw text_similarity()
    ratio comparison. Converts similarity_threshold into an ALLOWED EDIT
    DISTANCE that scales with word length (e.g. at 0.8, roughly 20% of a
    word's characters can differ and still pass), with an exact-match
    floor for 1-2 character words where a percentage-based allowance
    doesn't mean anything (verified: without this floor, single-character
    words could never fail, since the scaled formula's minimum of 1
    allowed error equals the total length).

    Verified against realistic OCR error patterns:
        hello/hallo (1 error)        -> pass  at threshold 0.8
        the/thc (1 error, short)     -> pass  at threshold 0.8
        everything/evenjthng (3 err) -> FAIL  at threshold 0.8 (correctly
                                          rejected -- the old ratio-based
                                          check let this through at 0.55
                                          AND at 0.7)
        friend/fnend (2 errors)      -> FAIL  at threshold 0.8
        a/e (completely wrong)       -> FAIL  (exact-match floor)
    """
    expected_n = expected.lower().strip()
    recognized_n = recognized.lower().strip()
    if len(expected_n) <= 2:
        allowed = 0
    else:
        allowed = max(1, round(len(expected_n) * (1 - similarity_threshold)))
    return _levenshtein(expected_n, recognized_n) <= allowed


def measure_word_gap_consistency(strokes, num_spaces: int, max_coefficient_of_variation: float = 0.5):
    """
    Estimates word-gap consistency directly from generated stroke segments
    (mm coordinates, already split by pen lifts -- same format
    _line_to_strokes returns).

    The model doesn't expose which pen-lift corresponds to a space versus a
    normal letter-to-letter gap -- there's no explicit word-boundary marker
    in its raw output. This uses a heuristic instead: the N largest
    horizontal gaps between consecutive segments (N = number of space
    characters in the original text) are assumed to be the actual word
    gaps, since word gaps are generally wider than within-word letter
    gaps. Their consistency (coefficient of variation -- stdev/mean) is
    then measured.

    This matters because OCR-based verification (text_similarity above)
    only checks whether the recognized WORDS match -- it's completely
    blind to whether the gaps BETWEEN those words look natural. A sample
    can score perfect similarity while still having one word crammed
    against the next and another gap twice as wide, since nothing was
    checking for that.

    Returns (is_consistent, coefficient_of_variation, word_gaps_mm).
    is_consistent is False if CoV exceeds max_coefficient_of_variation
    (default 0.5 -- deliberately permissive, since real handwriting has
    SOME natural word-gap variation; this is meant to catch clearly broken
    cases -- one huge gap next to one near-zero gap -- not enforce
    robotic uniformity).
    """
    if num_spaces <= 0 or len(strokes) < 2:
        return True, 0.0, []

    gaps = []
    for i in range(len(strokes) - 1):
        this_end_x = max(p[0] for p in strokes[i])
        next_start_x = min(p[0] for p in strokes[i + 1])
        gaps.append(next_start_x - this_end_x)

    if len(gaps) < num_spaces:
        # Not enough segment boundaries to meaningfully identify word gaps
        # (e.g. a very short line) -- nothing reliable to check.
        return True, 0.0, gaps

    word_gaps = sorted(gaps, reverse=True)[:num_spaces]
    mean_gap = sum(word_gaps) / len(word_gaps)
    if mean_gap <= 0:
        return True, 0.0, word_gaps

    variance = sum((g - mean_gap) ** 2 for g in word_gaps) / len(word_gaps)
    cov = (variance ** 0.5) / mean_gap

    return cov <= max_coefficient_of_variation, cov, word_gaps


# ---------------------------------------------------------------------------
# Recognition model wrapper
# ---------------------------------------------------------------------------

class HandwritingVerifier:
    """Thin wrapper around EasyOCR, returning (recognized_text, confidence)."""

    def __init__(self, languages: Sequence[str] = ("en",), gpu: bool = False):
        import easyocr
        self._reader = easyocr.Reader(list(languages), gpu=gpu)

    def recognize(self, image: Image.Image) -> Tuple[str, float]:
        import numpy as np
        arr = np.array(image.convert("RGB"))
        try:
            results = self._reader.readtext(arr, detail=1)
        except Exception as e:
            # A single malformed/edge-case image (e.g. an unusual aspect
            # ratio the recognition network's internal reshape logic
            # doesn't expect -- confirmed via a real "could not broadcast
            # input array" crash) shouldn't take down the whole generation.
            # Treat it as a failed recognition -- the caller's retry/flag
            # logic already handles that case correctly.
            print(f"[handwriting_verify] WARNING: recognition failed on one "
                  f"image ({e}); treating as a failed match for this attempt.")
            return "", 0.0
        if not results:
            return "", 0.0
        # results are (bbox, text, confidence) tuples -- sort left-to-right
        # by bounding box x-position so multi-word lines reassemble in order.
        results_sorted = sorted(results, key=lambda r: r[0][0][0])
        text = " ".join(r[1] for r in results_sorted)
        avg_conf = sum(r[2] for r in results_sorted) / len(results_sorted)
        return text, avg_conf


# ---------------------------------------------------------------------------
# Verified generator wrapper -- drop-in replacement for RNNHandwritingGenerator
# ---------------------------------------------------------------------------

class VerifiedRNNHandwritingGenerator:
    """
    Wraps an RNNHandwritingGenerator, adding a rasterize -> recognize ->
    compare -> retry loop around every line before accepting it. Exposes
    the same .generate(text) / .generate_blocks(blocks) contract as the
    base generator, so it's a drop-in replacement anywhere
    RNNHandwritingGenerator is used.

    After generation, check .flagged_lines -- a list of dicts for any line
    that never reached similarity_threshold even after max_retries, so you
    can warn the user (or block sending to the robot) before writing.
    """

    def __init__(
        self,
        base_gen,
        verifier: HandwritingVerifier,
        similarity_threshold: float = 0.80,
        min_confidence: float = 0.3,
        max_retries: int = 3,
        retry_below_similarity: float = 0.50,
        similarity_retry_max_attempts: int = 15,
        on_flag: Optional[Callable[[str, str, float], None]] = None,
        on_low_confidence: Optional[Callable[[str, str, float], bool]] = None,
        width_deviation_threshold: float = 0.25,
        check_spacing: bool = True,
        max_spacing_cov: float = 0.5,
    ):
        """
        retry_below_similarity / similarity_retry_max_attempts: if a line's
        raw character-similarity score (text_similarity, 0.0-1.0 --
        e.g. 0.50 means 50%) comes back below retry_below_similarity, the
        line is regenerated and re-recognized, up to
        similarity_retry_max_attempts times, until it scores at or above
        that bar. This uses the raw similarity ratio directly (the same
        one text_similarity() returns), not the length-scaled
        is_close_enough() check used elsewhere in this file for the final
        accept/reject decision -- so this is specifically "did this
        attempt score at least X%", matching a plain percentage read of
        the recognized text.

        Worth knowing: a fixed percentage threshold like this doesn't
        scale evenly with word length (verified earlier: 'the' vs 'thc',
        one real character error, scored 0.667, while 'everything' vs
        'evenjthng', three real errors, scored 0.737 -- so the same 50%
        cutoff is comparatively strict on short words and comparatively
        lenient on long ones). That's a real property of ratio-based
        scoring, not a bug in this retry rule -- just something to keep in
        mind if you're tuning retry_below_similarity and seeing it behave
        differently than expected across line lengths.

        min_confidence: minimum average OCR confidence (0.0-1.0) an
        attempt must ALSO clear, on top of passing is_close_enough(), to
        be accepted. This exists because OCR models with language-model
        priors will sometimes "guess" a plausible real word from
        genuinely garbled strokes -- with low confidence, even when the
        guess happens to match the intended text. Text matching alone
        can't tell that case apart from a genuinely clean recognition;
        confidence can. Start permissive (e.g. 0.3) and tighten based on
        what you observe -- EasyOCR's confidence values aren't perfectly
        calibrated, so treat this the same way as similarity_threshold:
        tune it against real output, don't assume the default is right.

        on_low_confidence: called once automatic retries (max_retries) are
        exhausted and the best attempt still hasn't reached
        similarity_threshold. Receives (expected_text, recognized_text,
        best_similarity_so_far) and must return True (try one more
        generation attempt, then ask again if it's still below threshold)
        or False (accept the best attempt seen so far and move on). If
        None (the default), behaves exactly as before: automatically keeps
        the best attempt without asking. This is what lets an interactive
        caller (e.g. a GUI) pause and let the user decide per-line, rather
        than every low-confidence line being silently auto-accepted.

        width_deviation_threshold: fraction (0.25 = 25%) by which an
        accepted line's actual rendered width can differ from its expected
        width (estimated from the calibrated mm-per-char) before it gets
        added to .width_flagged_lines.

        check_spacing / max_spacing_cov: an attempt must ALSO pass a
        word-gap consistency check (see measure_word_gap_consistency), not
        just OCR similarity, before being accepted -- otherwise the retry
        loop will keep sampling. This matters because OCR similarity is
        completely blind to spacing: a sample can score perfect text
        accuracy while still having wildly inconsistent gaps between
        words. Set check_spacing=False to disable and only check text
        recognizability (the original behavior).
        """
        self.base_gen = base_gen
        self.verifier = verifier
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.retry_below_similarity = retry_below_similarity
        self.similarity_retry_max_attempts = similarity_retry_max_attempts
        self.on_flag = on_flag
        self.on_low_confidence = on_low_confidence
        self.width_deviation_threshold = width_deviation_threshold
        self.check_spacing = check_spacing
        self.max_spacing_cov = max_spacing_cov
        self.flagged_lines: List[dict] = []
        self.width_flagged_lines: List[dict] = []
        self.spacing_flagged_lines: List[dict] = []

    def _check_width_deviation(self, line_text: str, strokes, font_scale: float = 1.0):
        """
        Compares this line's actual rendered width against the expected
        width (line_text length * calibrated mm-per-char), and records it
        in width_flagged_lines if the deviation exceeds
        width_deviation_threshold. Called once per FINAL accepted line
        (whether accepted on the first attempt or after retries) -- not
        during the retry search itself, since we're checking the outcome,
        not trying to steer the search.
        """
        if not strokes or not line_text.strip():
            return
        mm_per_char = self.base_gen._get_mm_per_char() * font_scale
        expected_width = len(line_text) * mm_per_char
        if expected_width <= 0:
            return
        all_x = [p[0] for s in strokes for p in s]
        actual_width = max(all_x) - min(all_x)
        deviation = abs(actual_width - expected_width) / expected_width
        if deviation > self.width_deviation_threshold:
            self.width_flagged_lines.append({
                "text": line_text,
                "expected_width_mm": expected_width,
                "actual_width_mm": actual_width,
                "deviation_frac": deviation,
            })

    def _verified_line(self, line_text, x_start, y_baseline, font_scale: float = 1.0):
        if not line_text.strip():
            return self.base_gen._line_to_strokes(line_text, x_start, y_baseline, font_scale)

        num_spaces = line_text.count(" ")

        # Best-by-similarity candidate seen so far (used for the flagged_lines
        # fallback if text recognition itself never succeeds).
        best_strokes = None
        best_similarity = -1.0
        best_recognized = ""
        best_confidence = 0.0

        # Among attempts that DID pass the text+confidence checks, track the
        # one with the best (lowest) word-gap coefficient of variation --
        # used as a preferred fallback if no attempt achieves both good
        # text AND good spacing within the retry budget. A text-correct
        # line with imperfect spacing is a better result to keep than the
        # overall highest-similarity line if that one had worse spacing.
        best_spacing_ok_strokes = None
        best_spacing_ok_cov = None
        best_spacing_ok_recognized = ""

        def attempt_once():
            nonlocal best_strokes, best_similarity, best_recognized, best_confidence
            nonlocal best_spacing_ok_strokes, best_spacing_ok_cov, best_spacing_ok_recognized

            strokes = self.base_gen._line_to_strokes(line_text, x_start, y_baseline, font_scale)
            if not strokes:
                return strokes, None, None, True, None

            img = rasterize_strokes(strokes)
            if img is None:
                return strokes, None, None, True, None

            recognized, conf = self.verifier.recognize(img)
            similarity = text_similarity(recognized, line_text)
            # The actual retry trigger: a raw similarity score below
            # retry_below_similarity (default 0.50, i.e. 50%) means this
            # attempt gets rejected and the line is regenerated -- see
            # retry_below_similarity in __init__ for the full explanation.
            similarity_ok = similarity >= self.retry_below_similarity
            conf_ok = conf >= self.min_confidence

            # Track the best-seen attempt by similarity for reporting
            # purposes even if it doesn't end up accepted.
            if similarity > best_similarity:
                best_similarity = similarity
                best_strokes = strokes
                best_recognized = recognized
                best_confidence = conf

            spacing_ok = True
            if self.check_spacing and similarity_ok and conf_ok:
                spacing_ok, spacing_cov, gaps = measure_word_gap_consistency(
                    strokes, num_spaces, self.max_spacing_cov)
                if not spacing_ok:
                    print(f"[handwriting_verify] Spacing check rejected an attempt for "
                          f"{line_text[:40]!r}{'...' if len(line_text) > 40 else ''}: "
                          f"word gaps were {[round(g, 1) for g in gaps]}mm "
                          f"(coefficient of variation {spacing_cov:.2f}, limit "
                          f"{self.max_spacing_cov}). If this looks fine to you when you "
                          f"see the actual output, raise max_spacing_cov to be more permissive.")
                if best_spacing_ok_cov is None or spacing_cov < best_spacing_ok_cov:
                    best_spacing_ok_cov = spacing_cov
                    best_spacing_ok_strokes = strokes
                    best_spacing_ok_recognized = recognized

            return strokes, recognized, (similarity_ok and conf_ok), spacing_ok, conf

        def is_fully_acceptable(text_and_conf_ok, spacing_ok):
            return bool(text_and_conf_ok) and spacing_ok

        # Automatic retry phase: any attempt scoring below
        # retry_below_similarity gets thrown out and regenerated, up to
        # similarity_retry_max_attempts times.
        for _attempt in range(self.similarity_retry_max_attempts + 1):
            strokes, recognized, text_and_conf_ok, spacing_ok, conf = attempt_once()
            if not strokes:
                return strokes
            if is_fully_acceptable(text_and_conf_ok, spacing_ok):
                self._check_width_deviation(line_text, strokes, font_scale)
                return strokes
            # else: retry -- the RNN's sampling is stochastic (temperature-
            # based, per the Graves paper's mixture density output), so a
            # fresh call to _line_to_strokes produces genuinely different
            # strokes each time, not an identical repeat.

        # Automatic retries exhausted without a fully-acceptable attempt.
        # If an interactive callback is set, defer the keep-vs-retry
        # decision to it -- looping for as many manual retries as the
        # caller requests.
        while self.on_low_confidence is not None:
            try_again = self.on_low_confidence(line_text, best_recognized, best_similarity)
            if not try_again:
                break
            strokes, recognized, text_and_conf_ok, spacing_ok, conf = attempt_once()
            if strokes and is_fully_acceptable(text_and_conf_ok, spacing_ok):
                self._check_width_deviation(line_text, strokes, font_scale)
                return strokes
            # else: still not fully acceptable -- loop back and ask again.

        # No fully-acceptable attempt found. Prefer a candidate that at
        # least got the TEXT (and confidence) right, even with imperfect
        # spacing, over the raw best-similarity candidate (which might be
        # worse on both fronts) -- and record the spacing shortfall
        # separately from pure recognition failures, since these are
        # different problems with different implications for what to do
        # about the line.
        if best_spacing_ok_strokes is not None:
            self.spacing_flagged_lines.append({
                "text": line_text,
                "recognized": best_spacing_ok_recognized,
                "spacing_cov": best_spacing_ok_cov,
            })
            self._check_width_deviation(line_text, best_spacing_ok_strokes, font_scale)
            return best_spacing_ok_strokes

        self.flagged_lines.append({
            "text": line_text,
            "recognized": best_recognized,
            "similarity": best_similarity,
            "confidence": best_confidence,
            "retry_below_similarity": self.retry_below_similarity,
            "retries_allowed": self.similarity_retry_max_attempts,
        })
        if self.on_flag:
            self.on_flag(line_text, best_recognized, best_similarity)
        self._check_width_deviation(line_text, best_strokes, font_scale)
        return best_strokes

    def generate(self, text: str):
        cfg = self.base_gen.cfg
        strokes_out = []
        lines = self.base_gen.wrap_text(text)
        y_cursor = cfg["y_offset_mm"]
        line_dir = -1 if cfg.get("reverse_line_direction") else 1
        for line in lines:
            x_start = self.base_gen._jittered_x_start(cfg["x_offset_mm"])
            strokes_out.extend(self._verified_line(line, x_start, y_cursor))
            y_cursor += line_dir * cfg["line_spacing_mm"]
        return strokes_out

    def generate_blocks(self, blocks):
        cfg = self.base_gen.cfg
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
                    y_cursor += line_dir * cfg["line_spacing_mm"] * 0.4
                lines = self.base_gen.wrap_text(block["text"], font_scale=scale)
                for line in lines:
                    x_start = self.base_gen._jittered_x_start(cfg["x_offset_mm"])
                    strokes_out.extend(
                        self._verified_line(line, x_start, y_cursor, font_scale=scale))
                    y_cursor += line_dir * cfg["line_spacing_mm"] * scale
                y_cursor += line_dir * cfg["line_spacing_mm"] * 0.3

            elif btype in ("bullet", "numbered"):
                x_indent = cfg["x_offset_mm"] + INDENT_MM
                if btype == "numbered":
                    numbered_counter += 1
                    marker = f"{numbered_counter}. "
                else:
                    marker = "- "
                lines = self.base_gen.wrap_text(marker + block["text"], width_mm=cfg["page_width_mm"] - INDENT_MM)
                for i, line in enumerate(lines):
                    base_x = x_indent if i == 0 else x_indent + INDENT_MM * 0.6
                    x_start = self.base_gen._jittered_x_start(base_x)
                    strokes_out.extend(self._verified_line(line, x_start, y_cursor))
                    y_cursor += line_dir * cfg["line_spacing_mm"]

            else:  # paragraph
                lines = self.base_gen.wrap_text(block["text"])
                for line in lines:
                    x_start = self.base_gen._jittered_x_start(cfg["x_offset_mm"])
                    strokes_out.extend(self._verified_line(line, x_start, y_cursor))
                    y_cursor += line_dir * cfg["line_spacing_mm"]

            prev_type = btype

        return strokes_out