# x = np.linspace(0, 5, 1000, endpoint=False)
# y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)
# plt.plot(x, y)
# plt.show()

# x, y = np.mgrid[-73.2:-72.8:0.001, -40.4:-40.0:0.001]
# pos = np.empty(x.shape + (2,))
# pos[:,:,0] = x
# pos[:,:,1] = y
# rv = multivariate_normal([-73.02144, -40.26028], [[0.0036552549, 0.0002310531], [0.0002310531, 0.0011807548]])
# plt.contour(x, y, rv.pdf(pos))
# plt.show()

# print(rv.pdf([-73.02144, -40.26]))
#
# norm = multivariate_normal([0, 0], [[3, 0], [0, 3]])
# print(norm.pdf([0, 0]))

import pandas as pd
from scipy.stats import multivariate_normal
import json
import matplotlib.pyplot as plt
import numpy as np

# Define datapoints parameters
x_range = 130 * 2
y_range = 147 * 2
x_min = -73.58
x_max = -72.24
y_min = -40.31
y_max = -38.99
x_step = (x_max - x_min) / x_range
y_step = (y_max - y_min) / y_range
# Place to store data
datapts = []
visual_check = True
# Read in distribution data
print("Reading in data")
df = pd.read_csv("Base_region5_dists.csv")
print("Sorting data")
dist_params = {i+1: {"mu": [df.mu_x[i], df.mu_y[i]],
               "sigma": [[df.sigma_x[i], df.co_sigma[i]],
                         [df.co_sigma[i], df.sigma_y[i]]],
               "alpha": df.weight[i]}
         for i in range(len(df))}
print("Creating random variables")
dists = [{"rv": multivariate_normal(params["mu"], params["sigma"]),
          "alpha": params["alpha"]} for params in dist_params.values()]
print("Forming data points and evaluating")
for i in range(x_range):
    print("Data points range {}".format(i))
    for j in range(y_range):
        x_coord = x_min + i*x_step
        y_coord = y_min + j * y_step
        weight = sum([dist["alpha"] * dist["rv"].pdf([x_coord, y_coord]) for dist in dists])
        datapts.append({"x": x_coord,
                        "y": y_coord,
                        "w": weight})
if visual_check:
    plt.plot([pt['w'] for pt in datapts])
    plt.show()
print("Storing output data in JSON format")
with open("point_weights.json", 'w') as f:
    json.dump(datapts, f, indent=4)