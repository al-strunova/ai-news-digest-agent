"""Anthropic adapter for the curator's tool-use loop.

Implements run_tool_use_loop with the contract documented in curator.py.
Reads ANTHROPIC_API_KEY from env if no api_key is passed.
"""

from __future__ import annotations

import os
from typing import Callable

from anthropic import Anthropic

from curator import ProviderResult


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
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    client = Anthropic(api_key=key)

    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "input_schema": tool_input_schema,
    }

    messages: list[dict] = [{"role": "user", "content": user_message}]
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    cache_creation_tokens = 0
    tool_calls = 0

    while True:
        response = client.messages.create(
            model=model,
            system=system_prompt,
            tools=[tool_def],
            messages=messages,
            max_tokens=max_output_tokens,
        )

        u = response.usage
        input_tokens += u.input_tokens or 0
        output_tokens += u.output_tokens or 0
        cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                url_arg = (block.input or {}).get("url", "")
                result_text = tool_handler(url_arg)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
                tool_calls += 1

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    text = "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()

    return ProviderResult(
        text=text,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        hit_max_output_tokens=response.stop_reason == "max_tokens",
    )