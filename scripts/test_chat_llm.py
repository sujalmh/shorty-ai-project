import json
import requests

API = "http://localhost:8000/api/chat"

def post(msg: str, mode: str = "llm") -> None:
    try:
        r = requests.post(API, json={"message": msg, "mode": mode}, timeout=30)
        r.raise_for_status()
        print(">>>", msg)
        print(json.dumps(r.json(), indent=2))
        print("-" * 60)
    except Exception as e:
        print("ERROR posting message:", msg, "->", e)

if __name__ == "__main__":
    tests = [
        # edit a cell
        "set revenue of row 2 to 300",
        # add row with values
        "add row with region='North', product='Z', q1=10, q2=15, revenue=25, cost=5",
        # delete column
        "delete column cost",
        # plot
        "pie chart of revenue by region",
    ]
    for t in tests:
        post(t, "llm")
