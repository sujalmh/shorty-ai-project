import json
import sys
import traceback

# Allow importing backend package
sys.path.append("backend")

from app.services.data_service import data_service  # type: ignore

def try_plot(x: str, y: str, kind: str) -> None:
    print(f"=== plot(x='{x}', y='{y}', kind='{kind}') ===")
    try:
        fig = data_service.plot(x=x, y=y, kind=kind)
        print("OK. Keys:", list(fig.keys()))
        print("Traces:", len(fig.get("data", [])))
        print("Title:", (fig.get("layout") or {}).get("title"))
        print("Sample trace type:", fig.get("data", [{}])[0].get("type") if fig.get("data") else None)
    except Exception as e:
        print("ERROR:", type(e).__name__, str(e))
        traceback.print_exc()
    print()

if __name__ == "__main__":
    # Inspect current columns to ensure x/y exist
    try:
        from app.services.data_service import data_service as ds  # type: ignore
        cols = list(map(str, ds.df.columns))
        print("Current columns:", cols)
        print("Row count:", len(ds.df.index))
        print()
    except Exception:
        pass

    # Test common cases
    try_plot("region", "revenue", "pie")
    try_plot("region", "revenue", "bar")
    try_plot("region", "q1", "bar")
