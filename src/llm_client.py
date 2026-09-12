"""
llm_client.py
-------------
Thin wrapper so the rest of the codebase doesn't care which LLM provider is
configured. Default: OpenAI. Reads OPENAI_API_KEY from the environment
(load it from a .env file with python-dotenv -- see README).

To swap to Anthropic: implement AnthropicClient below (same .complete()
signature) and change PROVIDER in .env. Left as a documented stub rather than
built out, since the assignment only needs one working provider end-to-end
(see decision_log.md).
"""
import os
import json
import time
import threading
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Free-tier LLM APIs (esp. Gemini) enforce tight requests-per-minute caps.
# Rather than only reacting to 429s after the fact, we proactively space out
# calls so we rarely trigger them in the first place. Configurable because
# the exact free-tier RPM varies by model/account -- check your provider's
# dashboard and adjust GEMINI_MIN_INTERVAL_SECONDS in .env if you still see
# rate-limit errors (higher number = slower but safer).
_MIN_INTERVAL = float(os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "13"))
_rate_lock = threading.Lock()
_last_call_time = [0.0]


def _throttle():
    with _rate_lock:
        elapsed = time.time() - _last_call_time[0]
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_call_time[0] = time.time()


class OpenAIClient:
    def __init__(self, model: str = MODEL):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, json_mode: bool = False,
                 temperature: float = 0.3, max_tokens: int = 500) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return resp.choices[0].message.content


class AnthropicClient:
    """Stub -- see module docstring. Implement if you'd rather use Claude."""
    def __init__(self, model: str = "claude-sonnet-4-6"):
        raise NotImplementedError("Swap PROVIDER=openai -> implement this if needed.")

    def complete(self, system: str, user: str, json_mode: bool = False,
                 temperature: float = 0.3, max_tokens: int = 500) -> str:
        raise NotImplementedError


class GeminiClient:
    """Google Gemini via the free tier (AI Studio API key, no card required).
    Get a key at https://aistudio.google.com/apikey -- generous free rate
    limits, good enough for this assignment's subsample-sized runs.
    Uses the current `google-genai` SDK (the older `google-generativeai`
    package is deprecated as of 2025)."""
    def __init__(self, model: str = None):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv("LLM_MODEL", "gemini-2.0-flash")

    def complete(self, system: str, user: str, json_mode: bool = False,
                 temperature: float = 0.3, max_tokens: int = 500) -> str:
        import re as _re
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        # Free tier is rate-limited (e.g. 5 requests/minute on some models).
        # Our agent makes several calls per message (classify + escalate +
        # draft + judge). We proactively throttle (_throttle()) to avoid
        # most 429s, and still retry with the server-suggested delay as a
        # fallback if one slips through.
        max_retries = 6
        for attempt in range(max_retries):
            _throttle()
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=user, config=config,
                )
                return resp.text
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    match = _re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)", msg)
                    wait_s = float(match.group(1)) + 2 if match else 20
                    print(f"    [rate limited, waiting {wait_s:.0f}s before retry "
                          f"{attempt + 1}/{max_retries}]")
                    time.sleep(wait_s)
                    continue
                raise
        raise RuntimeError("Gemini API: exceeded max retries on rate limiting.")


def get_client():
    if PROVIDER == "openai":
        return OpenAIClient()
    elif PROVIDER == "anthropic":
        return AnthropicClient()
    elif PROVIDER == "gemini":
        return GeminiClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")


def safe_json_parse(text: str) -> dict:
    """LLMs sometimes wrap JSON in ```json fences despite instructions. Strip them."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw": text}
