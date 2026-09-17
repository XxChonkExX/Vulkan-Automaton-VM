import urllib.request
import json
import sys
import time

# Usage: serve_bench_pp.py <port> <prompt_repeats>
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8092
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

prompt = "Write a detailed 400 word story about a deep sea diver. " * reps
body = json.dumps({
    "prompt": prompt,
    "n_predict": 32,
    "temperature": 0.7,
}).encode()

url = f"http://127.0.0.1:{port}/completion"
t0 = time.time()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as r:
    data = json.loads(r.read())
t = data.get("timings", {})
print(f"prompt: {t.get('prompt_n')} tokens in {t.get('prompt_ms', 0)/1000:.1f}s "
      f"({t.get('prompt_per_second', 0):.2f} t/s)")
print(f"decode: {t.get('predicted_n')} tokens "
      f"({t.get('predicted_per_second', 0):.2f} t/s)")
print(f"total wall: {time.time()-t0:.1f}s")
