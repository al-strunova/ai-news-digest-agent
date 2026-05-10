"""Gemini adapter for the curator's tool-use loop.

Uses the google-genai SDK. Function calling auto-execution is disabled
so we drive the loop manually, mirroring the Anthropic adapter shape.

Gemini's finish_reason is STOP whether the model is done or about to
call a tool; we decide whether to loop based on whether the response
contains function_call parts, not on finish_reason.

Cache tokens are reported as 0; the surface for them in google-genai
differs from Anthropic's, and our daily cadence won't hit caches anyway.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from google import genai
from google.genai import types
from google.genai.errors import ServerError

from curator import ProviderResult

# Retry transient 5xx (e.g. "503 UNAVAILABLE — high demand") with
# exponential backoff. Worst case adds ~14s of waiting before the
# adapter raises and curator falls back to the broken-agent email.
_MAX_RETRIES = 3
_BACKOFF_SECONDS = (2, 4, 8)


def _generate_with_retry(client, *, model, contents, config):
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except ServerError:
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_BACKOFF_SECONDS[attempt])


def run_tool_use_loop(
    *,
    model: str,
    system_prompt: str,
    user_message: str,
    tool_name: str,
    tool_description: str,
    tool_input_schema: dict,
    tool_handler: Callable[[str], str],
    max_output_tokens: int,
    api_key: str | None = None,
) -> ProviderResult:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=key)

    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=tool_name,
                description=tool_description,
                parameters=tool_input_schema,
            )
        ]
    )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[tool],
        max_output_tokens=max_output_tokens,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents: list = [
        types.Content(role="user", parts=[types.Part(text=user_message)]),
    ]

    tool_calls = 0
    input_tokens = 0
    output_tokens = 0
    finish_reason = None

    while True:
        response = _generate_with_retry(client, model=model, contents=contents, config=config)

        usage = response.usage_metadata
        if usage:
            input_tokens += usage.prompt_token_count or 0
            output_tokens += usage.candidates_token_count or 0

        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason

        function_calls: list = []
        text_parts: list[str] = []
        for part in candidate.content.parts or []:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                function_calls.append(fc)
            txt = getattr(part, "text", None)
            if txt:
                text_parts.append(txt)

        if not function_calls:
            break

        # Append model's tool-calling turn so the API has the full conversation
        contents.append(candidate.content)

        response_parts = []
        for fc in function_calls:
            args = dict(fc.args or {})
            url = args.get("url", "")
            result_text = tool_handler(url)
            response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result_text},
                    )
                )
            )
            tool_calls += 1
        contents.append(types.Content(role="user", parts=response_parts))

    text = "".join(text_parts).strip()
    hit_max = finish_reason == types.FinishReason.MAX_TOKENS

    return ProviderResult(
        text=text,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        hit_max_output_tokens=hit_max,
    )