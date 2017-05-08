import pandas as pd
import json

# Import cluster centroids and package into a dict
df = pd.read_csv("base_freclog_sur_radio.csv")
df2 = pd.read_csv("base_freclog_sur.csv")
# Obtain cluster data from CSV read by panda
clusters = [{   "lat": df.latitude[i],
                "lng": df.longitude[i],
                "radius": df.rad[i],
                "risk": df2.colores[i]}
            for i in range(len(df))]
print(clusters)
# Dump resulting data into JSON file
with open('circles_coords.json', 'w') as f:
    json.dump(clusters, f, indent=4)