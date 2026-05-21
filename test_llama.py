import requests
import time

key = "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"
model = "meta-llama/llama-3-8b-instruct:free"

start_time = time.time()
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 10
}
response = requests.post(url, json=payload, headers=headers, timeout=5)
elapsed = time.time() - start_time
print(f"Model {model} -> Status: {response.status_code}, Time: {elapsed:.2f}s")
if response.status_code != 200:
    print(f"  Error: {response.text}")
