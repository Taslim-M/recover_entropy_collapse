"""
LLM client for OpenRouter API calls.
Handles retries, JSON parsing, and error handling.
"""

import json
import time
import requests
from typing import Optional

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_DELAY,
)


def call_llm(
    prompt: str,
    model: str,
    system_prompt: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
) -> str:
    """
    Call an LLM via OpenRouter API.

    Args:
        prompt: The user message
        model: OpenRouter model identifier
        system_prompt: Optional system message
        temperature: Sampling temperature (default from config)
        max_tokens: Max response tokens (default from config)
        json_mode: If True, hint the model to return JSON

    Returns:
        The model's text response

    Raises:
        RuntimeError: If all retry attempts fail
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Export it as an environment variable or set it in config.py"
        )

    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            elif "error" in data:
                raise RuntimeError(f"API error: {data['error']}")
            else:
                raise RuntimeError(f"Unexpected response format: {data}")

        except (requests.RequestException, RuntimeError) as e:
            if attempt < LLM_RETRY_ATTEMPTS - 1:
                print(f"  LLM call attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(LLM_RETRY_DELAY * (attempt + 1))
            else:
                raise RuntimeError(
                    f"All {LLM_RETRY_ATTEMPTS} LLM call attempts failed. Last error: {e}"
                )


def call_llm_json(
    prompt: str,
    model: str,
    system_prompt: str = "",
    temperature: Optional[float] = None,
) -> dict:
    """
    Call LLM and parse the response as JSON.
    Strips markdown code fences if present.

    Returns:
        Parsed JSON as a dict
    """
    raw = call_llm(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        json_mode=True,
    )

    # Strip markdown JSON fences
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Try to find JSON within the response
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        # Try array
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        raise RuntimeError(
            f"Failed to parse LLM response as JSON: {e}\nResponse was:\n{raw[:500]}"
        )
