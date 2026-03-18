"""
LLM client supporting both OpenRouter (model name) and cloud GPU (URL).
Handles retries, JSON parsing, and error handling.

Routing logic in call_llm / call_llm_json:
  url=None  → OpenRouter, authenticated with OPENROUTER_API_KEY,
              routed to OPENROUTER_BASE_URL using the `model` name.
  url=<str> → Cloud GPU endpoint at that URL, authenticated with
              CLOUD_GPU_API_KEY (optional). `model` is still sent in
              the payload (set it to whatever the server expects).
"""

import json
import re
import time
import requests
from typing import Optional


def _strip_think_blocks(text: str) -> str:
    """Remove <think>…</think> reasoning blocks produced by thinking models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    CLOUD_GPU_API_KEY,
    CLOUD_GPU_USE_CHAT_FORMAT,
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
    url: Optional[str] = None,
) -> str:
    """
    Call an LLM via OpenRouter or a cloud GPU endpoint.

    Args:
        prompt: The user message
        model: Model identifier – used as the OpenRouter model name when
               url=None, and sent in the payload for cloud GPU calls.
        system_prompt: Optional system message
        temperature: Sampling temperature (default from config)
        max_tokens: Max response tokens (default from config)
        json_mode: If True, hint the model to return JSON
        url: If provided, POST to this URL using CLOUD_GPU_API_KEY.
             If None, POST to OPENROUTER_BASE_URL using OPENROUTER_API_KEY.

    Returns:
        The model's text response

    Raises:
        RuntimeError: If all retry attempts fail
    """
    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

    if url:
        # Cloud GPU path
        headers: dict = {"Content-Type": "application/json"}
        if CLOUD_GPU_API_KEY:
            headers["Authorization"] = f"Bearer {CLOUD_GPU_API_KEY}"

        if CLOUD_GPU_USE_CHAT_FORMAT:
            # Chat completions format – requires model to have a chat template
            target_url = url
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload: dict = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response_key = ("choices", "message", "content")
        else:
            # Raw completions format – for base models with no chat template.
            # Auto-converts the URL: .../chat/completions → .../completions
            target_url = url.replace("/chat/completions", "/completions")
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            payload = {
                "model": model,
                "prompt": full_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response_key = ("choices", "text")
    else:
        # OpenRouter path – always uses chat completions format
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. "
                "Export it as an environment variable or set it in .env"
            )
        target_url = OPENROUTER_BASE_URL
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response_key = ("choices", "message", "content")

    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            response = requests.post(
                target_url,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if response_key == ("choices", "text"):
                    raw_text = choice["text"]
                else:
                    raw_text = choice["message"]["content"]
                return _strip_think_blocks(raw_text)
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
    url: Optional[str] = None,
) -> dict:
    """
    Call LLM and parse the response as JSON.
    Strips markdown code fences if present.

    Args:
        url: Forwarded to call_llm – see its docstring for routing behaviour.

    Returns:
        Parsed JSON as a dict
    """
    raw = call_llm(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        json_mode=True,
        url=url,
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
