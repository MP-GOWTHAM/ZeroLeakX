"""ZeroLeakX launcher — `python server.py` then open http://127.0.0.1:8000"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n  ZeroLeakX  ·  http://{host}:{port}\n")
    uvicorn.run("backend.app:app", host=host, port=port, reload=False)
