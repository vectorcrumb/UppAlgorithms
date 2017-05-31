import requests

fires = [0 for _ in range(3)]
url_feed = "http://e5aa819f.ngrok.io/feed"

payload = {
    "cameras": [{"id": i + 1, "status": fire} for i, fire in enumerate(fires)]
}
r = requests.post(url_feed, json=payload)

print("Shut off fires!\n{}".format(r))