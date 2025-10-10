import os
import requests

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    def generate_content(self, prompt: str, max_tokens: int = 700) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2}
        }
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        # Gemini returns candidates[0].content.parts[0].text
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            raise RuntimeError(f"Unexpected Gemini response: {data}")

def call_gemini(prompt: str, api_key: str, max_tokens: int = 700) -> str:
    client = GeminiClient(api_key)
    return client.generate_content(prompt, max_tokens)
