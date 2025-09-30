import subprocess
import time
import sys
import os
import signal

print(f"Using Python from: {sys.executable}")

backend_cmd = [sys.executable, "backend.py"]
frontend_cmd = [sys.executable, "frontend.py"]

backend = subprocess.Popen(backend_cmd)
print("🚀 Backend starting...")

time.sleep(5)
frontend = subprocess.Popen(frontend_cmd)
print("🎨 Frontend starting...")

try:
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    print("⏹️ Stopping services...")
    if os.name == "nt":  # Windows
        backend.send_signal(signal.CTRL_BREAK_EVENT)
        frontend.send_signal(signal.CTRL_BREAK_EVENT)
    else:  # Linux / Mac
        backend.terminate()
        frontend.terminate()
