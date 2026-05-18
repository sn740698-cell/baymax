import requests
import time
import json

def get_bot_response(user_text, mode):
    instructions = {
        "Health": "You are Baymax, a personal healthcare companion..."
    }
    
    models_to_try = ["openai", "gemini"]
    last_error = ""
    
    headers = {
        "Authorization": "Bearer sk_PqexpJokRElrFvPFxt5uy5vx8DWWS0ni",
        "Content-Type": "application/json"
    }
    
    for fallback_model in models_to_try:
        for attempt in range(2):
            try:
                url = "https://gen.pollinations.ai/v1/chat/completions"
                payload = {
                    "messages": [
                        {"role": "system", "content": instructions.get(mode, "")},
                        {"role": "user", "content": user_text}
                    ],
                    "model": fallback_model,
                    "stream": True
                }
                
                with requests.post(url, json=payload, headers=headers, stream=True, timeout=10.0) as response:
                    if response.status_code != 200:
                        last_error = f"HTTP {response.status_code}: {response.text[:100]}"
                        if response.status_code in [429, 503] or "queue full" in response.text.lower():
                            time.sleep(2.0)
                            continue
                        else:
                            break
                    
                    got_content = False
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    content = data_json["choices"][0]["delta"].get("content", "")
                                    if content:
                                        got_content = True
                                        yield content
                                except Exception:
                                    continue
                    
                    if got_content:
                        return
                    else:
                        last_error = f"Model {fallback_model} returned empty response."
                        break
            except Exception as e:
                last_error = str(e)
                time.sleep(2.0)
                continue
                
    yield f"Error: All models failed. Last error: {last_error}"

# Run a test
print("Starting stream test with gen.pollinations.ai POST and real API key...")
for token in get_bot_response("tell me a 2-word joke", "Health"):
    print(token, end="", flush=True)
print()
