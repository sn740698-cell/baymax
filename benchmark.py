import requests
import time

openrouter_engines = [
    {"model": "google/gemma-7b-it", "key": "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"},
    {"model": "deepseek/deepseek-chat", "key": "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"},
    {"model": "openai/gpt-4o-mini", "key": "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"},
    {"model": "qwen/qwen-2.5-72b-instruct", "key": "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"}
]

for engine in openrouter_engines:
    start_time = time.time()
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {engine['key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": engine["model"],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        elapsed = time.time() - start_time
        print(f"Model {engine['model']} -> Status: {response.status_code}, Time: {elapsed:.2f}s")
        if response.status_code != 200:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"Model {engine['model']} -> Error: {str(e)}")
