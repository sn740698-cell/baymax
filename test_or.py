import requests
import json

keys_models = [
    ("google/gemma-2-9b-it:free", "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"),
    ("deepseek/deepseek-chat:free", "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"),
    ("openai/gpt-4o-mini", "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"),
    ("qwen/qwen-2-7b-instruct:free", "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"),
    ("qwen/qwen2.5-7b-instruct:free", "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")
]

for model, key in keys_models:
    print(f"Testing model {model} with key ending in {key[-4:]}...")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=5)
        print(response.status_code)
        if response.status_code == 200:
            print("Success")
        else:
            print(response.text)
    except Exception as e:
        print("Error:", e)
    print("-" * 30)
