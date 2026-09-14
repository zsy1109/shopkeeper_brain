import sys, time, subprocess

script = """
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('MODELSCOPE_CACHE', r'D:\\PyCharm项目\\shopkeeper_brain_1\\models')
os.environ.setdefault('MINERU_MODEL_DIR', r'D:\\PyCharm项目\\shopkeeper_brain_1\\models')
import uvicorn
from mineru.cli.server import create_app
try:
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=52915, log_level="debug")
except Exception as e:
    import traceback
    traceback.print_exc()
"""

p = subprocess.Popen(
    [sys.executable, "-c", script],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    env={**__import__('os').environ, "HF_ENDPOINT": "https://hf-mirror.com"}
)
time.sleep(30)
if p.poll() is None:
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
out, _ = p.communicate()
print("=== MINERU API FULL OUTPUT ===")
print(out[-10000:])