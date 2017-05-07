import pandas as pd
from math import sqrt
import json


# Constants to aid creation of discrete grid
boxes = []
dx, dy = 0.01, 0.01
x0, y0 = -73.73, -39.755
x_count, y_count = int((47/2)*5), int((32/2)*5)
prec = 3
k = 1

# Generate discrete grid over selected terrain
for i in range(x_count):
    for j in range(y_count):
        boxes.append({
            "p1": [round(x0 + dx * i, prec), round(y0 - dy * j, prec)],
            "p2": [round(x0 + dx * (i + 1), prec), round(y0 - dy * j, prec)],
            "p3": [round(x0 + dx * (i + 1), prec), round(y0 - dy * (j + 1), prec)],
            "p4": [round(x0 + dx * i, prec), round(y0 + dy * (j + 1), prec)],
            "pc": [round(x0 + dx * (i + .5), prec), round(y0 + dy * (j + .5), prec)]
        })
# Import cluster centroids and package into a dict
df = pd.read_csv("base_freclog_sur.csv")
# Obtain cluster data from CSV read by panda
clusters = {df.id[i]: { "centroid": [df.latitude[i], df.longitude[i]],
                        "freq": df.freq[i],
                        "freq_rel": df.freq_rel[i],
                        "level": df.levels[i],
                        "freq_log": df.log_rel[i],
                        "cluster_risk": df.colores[i]}
            for i in range(len(df))}
# Generate probability for each box
for box in boxes:
    box_prob = []
    for cl_i in clusters:
        # Obtain distance from box to each cluster and run probability equation
        dist = sqrt((box["pc"][0] - clusters[cl_i]["centroid"][0])**2 + (box["pc"][1] - clusters[cl_i]["centroid"][1])**2)
        box_prob.append((k * clusters[cl_i]["cluster_risk"] / len(clusters)) / (1 + dist**2))
    # Sum probabilities for all clusters. This value has already been averaged out, so we just sum
    box["risk"] = sum(box_prob)
    # Set box boundaries and delete unnecessary data
    box["north"] = box["p1"][1]
    box["south"] = box["p3"][1]
    box["east"] = box["p3"][0]
    box["west"] = box["p1"][0]
    del box["pc"], box['p1'], box['p2'], box['p3'], box['p4']
# Obtain risks, make relative to max value and spread data out further by dividing by range
box_risks = [box["risk"] for box in boxes]
box_risks_norm = [risk/max(box_risks) for risk in box_risks]
box_final_risks = [(riskn-min(box_risks_norm))/(max(box_risks_norm) - min(box_risks_norm)) for riskn in box_risks_norm]
# Classify risks in quintiles and append integer value to color risk list
color_risks = []
for frisk in box_final_risks:
    if frisk <= 0.2:
        color_risks.append(0)
    elif frisk <= 0.4:
        color_risks.append(1)
    elif frisk <= 0.6:
        color_risks.append(2)
    elif frisk <= 0.8:
        color_risks.append(3)
    else:
        color_risks.append(4)
# Update risks to color risks
for i, box in enumerate(boxes):
    box["risk"] = color_risks[i]
# Dump resulting data into JSON file
with open('boxes_coords.json', 'w') as f:
    json.dump(boxes, f, indent=4)
