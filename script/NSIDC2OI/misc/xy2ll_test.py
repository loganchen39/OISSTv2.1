from xy2ll import xy2ll
import numpy as np

np.set_printoptions(threshold=np.inf) # To print all numpy array elements.

# xs = np.linspace(-3333000, -500, 6666)
# ys = np.linspace(500, -3333000, 6666)
xs = np.linspace(-3333000, 3333000, 13333)
ys = np.linspace(3333000, -3333000, 13333)
print("xs", xs)
print("ys", ys)


to_lat = []
to_lon = []

for x, y in zip(xs,ys):
    lat, lon = xy2ll(x, y, -1, 0, 71)
    to_lat.append(lat)
    to_lon.append(lon)
    print('x=', x, ', y=', y, ', lat=', lat, ', lon=', lon)
