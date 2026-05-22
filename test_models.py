import requests
import json

openrouter_engines = [
    {"model": "perplexity/llama-3.1-sonar-large-128k-online", "key": "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"},
    {"model": "deepseek/deepseek-chat", "key": "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"},
    {"model": "google/gemma-2-9b-it", "key": "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"},
    {"model": "openai/gpt-4o-mini", "key": "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"}
]

print("Testing OpenRouter Models...")
for i, engine in enumerate(openrouter_engines):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {engine['key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": engine["model"],
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"{i+1}. Model: {engine['model']} -> Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"   Error: {resp.text[:200]}")
    except Exception as e:
        print(f"{i+1}. Model: {engine['model']} -> Exception: {e}")

print("\nTesting Pollinations fallback...")
try:
    resp = requests.get("https://text.pollinations.ai/Hi?model=openai", timeout=10)
    print(f"Pollinations -> Status: {resp.status_code}")
except Exception as e:
    print(f"Pollinations -> Exception: {e}")
