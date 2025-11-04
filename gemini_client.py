import os
import requests
from typing import List, Dict, Any

# Define the correct model alias for stability
MODEL_ALIAS = "gemini-2.5-flash"


class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # FIX: Using v1 and the currently recommended stable model alias
        self.base_url = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_ALIAS}:generateContent"

    def generate_content(self, prompt: str, max_tokens: int = 4096) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,  # Use API key from class instance
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2}
        }

        # NOTE: Using the fixed base_url
        # You may need to add 'verify=False' to requests.post if you are behind a corporate proxy
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()  # This raises the HTTPError on 4xx/5xx responses
        data = response.json()

        try:
            # Gemini returns candidates[0].content.parts[0].text
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            # Handle cases where the response is blocked, empty, or unexpected
            if 'error' in data:
                raise RuntimeError(f"Gemini API Error: {data['error'].get('message', 'Unknown Error')}")
            raise RuntimeError(f"Unexpected or blocked Gemini response: {data}")


def _messages_to_prompt_native(messages: List[Dict]) -> str:
    """Converts the list of messages (system/user roles) into a single prompt string."""
    sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_parts = [m["content"] for m in messages if m.get("role") == "user"]

    prompt = ""
    if sys_parts:
        prompt += f"System Instructions:\n{sys_parts[-1]}\n\n---\n\n"

    prompt += f"USER REQUEST:\n{user_parts[-1]}"

    return prompt


# --- MODIFIED call_gemini to match arguments from Summarizer.py ---

def call_gemini(
        base_url: str,
        model: str,
        messages: List[Dict],
        headers: Dict[str, str],
        timeout_seconds: int,
        verify_param: Any,
        max_output_tokens: int = 700,
) -> str:
    """
    Adapts the arguments from Summarizer.py's call signature to the native GeminiClient.
    """

    # 1. Extract API Key from headers (expected to be 'Authorization: Bearer <KEY>')
    auth_header = headers.get("Authorization", "")
    api_key = auth_header.replace("Bearer ", "").strip()

    if not api_key:
        api_key = headers.get("x-goog-api-key", "")

    if not api_key:
        raise ValueError("API Key not found in headers. Please check your authentication setup in Summarizer.py.")

    # 2. Convert messages to a single prompt string
    prompt = _messages_to_prompt_native(messages)

    # 3. Call the native Gemini client using the extracted key and prompt
    client = GeminiClient(api_key)
    return client.generate_content(prompt, max_tokens=max_output_tokens)