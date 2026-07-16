import sys
sys.path.insert(0, '.')
from handwriting_bot import HandwritingGenerator, DEFAULTS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cfg = dict(DEFAULTS)
gen = HandwritingGenerator(cfg, rng_seed=307141)  # match the seed it printed
strokes = gen.generate("your text here")

fig, ax = plt.subplots(figsize=(10, 4))
for s in strokes:
    xs = [p[0] for p in s]
    ys = [p[1] for p in s]
    ax.plot(xs, ys, color='black', linewidth=1.2)
ax.invert_yaxis()
ax.set_aspect('equal')
ax.axis('off')
plt.savefig('preview.png', dpi=200, bbox_inches='tight')
print("saved preview.png")
