#!/usr/bin/env python3
"""
smoke_test_verify.py
=======================
Real runtime test of the numpy 1.24.3 pin -- exercises actual EasyOCR
recognition (array processing, not just import), and confirms TensorFlow
still works too. Run this once after any dependency change in this venv
before trusting HandwritingVerifier in the real pipeline.

    python smoke_test_verify.py
"""

from PIL import Image, ImageDraw

print("Step 1: confirming TensorFlow still imports and runs a basic op...")
import tensorflow as tf
result = tf.constant([1, 2, 3]) + tf.constant([4, 5, 6])
print(f"  TF op result: {result.numpy()}  (should be [5 7 9])")

print("\nStep 2: creating a real test image with visible text-like strokes...")
img = Image.new("L", (300, 100), color=255)
draw = ImageDraw.Draw(img)
# draw simple block letter shapes so there's SOMETHING plausibly readable --
# this doesn't need to be pretty, just needs to exercise real array ops
draw.text((20, 30), "TEST", fill=0)
img.save("smoke_test_input.png")
print("  Saved smoke_test_input.png for reference")

print("\nStep 3: running EasyOCR on it (this is the real numpy-array-heavy step)...")
import easyocr
import numpy as np

reader = easyocr.Reader(["en"], gpu=False)
arr = np.array(img.convert("RGB"))
results = reader.readtext(arr, detail=1)

print(f"  EasyOCR returned {len(results)} result(s):")
for bbox, text, conf in results:
    print(f"    recognized: {text!r}  (confidence {conf:.2f})")

print("\nSMOKE TEST PASSED -- both TensorFlow and EasyOCR ran real operations successfully.")