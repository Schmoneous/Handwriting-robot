import matplotlib.pyplot as plt
from rnn_handwriting_adapter import RNNStrokeSource, DrawCoreCalibration, strokes_to_pen_points

src = RNNStrokeSource()
strokes = src.sample_lines(["Testing DrawCore integration"], biases=[0.75], styles=[9])

cal = DrawCoreCalibration()
cal.model_units_per_mm = 1.0
points = strokes_to_pen_points(strokes[0], cal)

xs = [p.x_mm for p in points if p.pen_down]
ys = [p.y_mm for p in points if p.pen_down]

plt.plot(xs, ys, 'k.-', markersize=1)
plt.gca().invert_yaxis()
plt.axis('equal')
plt.savefig('preview2.png')
print('saved preview2.png')