import requests

key = "sk_PqexpJokRElrFvPFxt5uy5vx8DWWS0ni"

# Test OpenRouter
print("Testing OpenRouter...")
response_or = requests.get(
    "https://openrouter.ai/api/v1/auth/key",
    headers={"Authorization": f"Bearer {key}"}
)
print("OpenRouter Status:", response_or.status_code)
print("OpenRouter Response:", response_or.text)

# Test Pollinations
print("\nTesting Pollinations...")
response_poll = requests.get(
    "https://gen.pollinations.ai/models",
)
# Pollinations doesn't have an auth check endpoint easily available, but we know OpenRouter has one.
