"""ZeroLeakX launcher — `python server.py` then open http://127.0.0.1:8000"""
import uvicorn

if __name__ == "__main__":
    print("\n  ZeroLeakX  ·  http://127.0.0.1:8000\n")
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
