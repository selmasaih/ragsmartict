import requests
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

t = time.time()
r = requests.post("http://127.0.0.1:8003/api/query", json={"question": "Qu est-ce que la transformee de Fourier?"})
data = r.json()
wall = int((time.time() - t) * 1000)

print(f"Status: {r.status_code}")
print(f"Server latency: {data.get('latency_ms')}ms")
print(f"Wall clock: {wall}ms")
print(f"Timings: {data.get('timings')}")
answer = data.get('answer', '')[:300]
print(f"Answer: {answer}")
