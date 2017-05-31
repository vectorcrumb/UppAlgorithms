import serial
from time import sleep
import numpy as np
import requests

is_conn = False
ser_port = "COM10"
baud = 115200
url_feed = "https://befire.herokuapp.com/sensor_data"

num_sensors = 30
x_min = -73.58
x_max = -72.24
y_min = -40.31
y_max = -38.99
x_real = 0
y_real = 0

lon_pos = list(np.random.uniform(x_min, x_max, num_sensors))
lat_pos = list(np.random.uniform(y_min, y_max, num_sensors))

smoke_params = {"mu": 1, "sigma": 4}
temp_params = {"mu": 1.5, "sigma": 1.2}
humid_params = {"mu": 1, "sigma": .8}
mult = [.05, .05, .001]

sensor_pos = {
    "pos": [lon, lat] for lon, lat in zip(lon_pos, lat_pos)
}

ser = serial.Serial(ser_port, baud)

while not is_conn:
    print("connecting")
    serin = ser.readline()
    if bool(serin):
        is_conn = True
sleep(1)


while True:
    # Wait for real sensor, get data, verify and convert to int
    while not ser.in_waiting:
        sleep(0.01)
    d_in = ser.readline().decode()
    sensors = d_in.strip().split(",")
    if len(sensors) != 3:
        continue
    sensors = list(map(float, sensors))
    sens_vals = [sensor * mult[i] for i, sensor in enumerate(sensors)]
    # Simulate real data
    smokes = [np.random.normal(smoke_params["mu"], smoke_params["sigma"]) for a in range(num_sensors)]
    temps = [np.random.normal(temp_params["mu"], temp_params["sigma"]) for b in range(num_sensors)]
    humids = [np.random.normal(humid_params["mu"], humid_params["sigma"]) for c in range(num_sensors)]
    sim_sensors = [item for sublist in zip(smokes, temps, humids) for item in sublist]
    sim_lon = [lon for sub_lon in zip(lon_pos, lon_pos, lon_pos) for lon in sub_lon]
    sim_lat = [lat for sub_lat in zip(lat_pos, lat_pos, lat_pos) for lat in sub_lat]
    # Fit into payload
    payload = {
        "sensors": [{"lat": lat, "lng": lon, "weight": sim_val}
                    for lat, lon, sim_val in zip(sim_lat, sim_lon, sim_sensors)]
    }
    for sen_val in sens_vals:
        payload["sensors"].append({"lat": y_real, "lng": x_real, "weight": sen_val})
    print("Sending data: {}".format(payload.keys()))
    r = requests.post(url_feed, json=payload)
    print(r)
