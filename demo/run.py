import subprocess
import time
import sys
import os
import signal

def kill_ngrok():
    try:
        subprocess.run("taskkill /IM ngrok.exe /F", shell=True, check=True)
        print("❌ Đã tắt toàn bộ session ngrok cũ")
    except subprocess.CalledProcessError:
        print("ℹ️ Không có tiến trình ngrok nào đang chạy")

if __name__ == "__main__":
    kill_ngrok()
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
