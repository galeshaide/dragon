import requests

paths = [
    ("Gateway", "http://127.0.0.1:9001/predict/"),
    ("Model 8002", "http://127.0.0.1:8002/predict/"),
]
img = r"c:\Users\gales\Desktop\github zip\Dragon-main\frontend\ui\public\images\lung.jpg"
for name, url in paths:
    try:
        with open(img, "rb") as f:
            files = {"file": ("lung.jpg", f, "image/jpeg")}
            data = {"model_name": "lung"}
            print(f"POSTing to {name}: {url}")
            r = requests.post(url, files=files, data=data, timeout=10)
            print(name, r.status_code, r.headers.get('content-type'))
            try:
                print(r.json())
            except Exception:
                print(r.text[:1000])
    except Exception as e:
        print(name, "error:", e)
