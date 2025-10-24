import json
import requests

if __name__ == "__main__":
    url = "http://localhost:8000/api/plot"
    payload = {"x": "region", "y": "revenue", "kind": "pie"}
    try:
        r = requests.post(url, json=payload, timeout=30)
        print("status:", r.status_code)
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text)
    except Exception as e:
        print("REQUEST ERROR:", e)
