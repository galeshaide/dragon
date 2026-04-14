import requests
import os

IMG = r"c:\Users\gales\Desktop\github zip\Dragon-main\frontend\ui\public\images\brain.jpg"
ENDPOINTS = [
    ("Gateway", "http://127.0.0.1:9001"),
    ("Model 8002", "http://127.0.0.1:8002"),
    ("Model 8003", "http://127.0.0.1:8003"),
    ("Model 8004", "http://127.0.0.1:8004"),
    ("Model 8005", "http://127.0.0.1:8005"),
]
MODELS = ["brain","lung","lung_size","skin","pancreas"]

for name, base in ENDPOINTS:
    for m in MODELS:
        for ep in ("/predict/","/gradcam/"):
            url = base + ep
            try:
                with open(IMG, "rb") as f:
                    files = {"file": (os.path.basename(IMG), f, "image/jpeg")}
                    data = {"model_name": m}
                    print(f"POST -> {name} {url} model={m}")
                    r = requests.post(url, files=files, data=data, timeout=10)
                    print("  status:", r.status_code, "content-type:", r.headers.get('content-type'))
                    try:
                        print("  json:", r.json())
                    except Exception:
                        print("  text:", r.text[:400])
            except Exception as e:
                print("  error:", e)
    print()
