"""Resend email wrapper.

One function: send_digest(markdown_content, subject). Renders the
markdown to HTML for rich display and includes the raw markdown as the
plain-text fallback.

Sender is hardcoded to Resend's free shared domain (onboarding@resend.dev)
so we don't need to verify a custom domain to start. Switch to a
verified domain by changing FROM_ADDRESS once that's set up.

Errors propagate. main.py decides what to do on failure.
"""

from __future__ import annotations

import os

import markdown as md_lib
import resend

FROM_ADDRESS = "onboarding@resend.dev"


def send_digest(markdown_content: str, subject: str) -> str:
    api_key = os.environ.get("RESEND_API_KEY")
    recipient = os.environ.get("RECIPIENT_EMAIL")
    if not api_key:
        raise RuntimeError("RESEND_API_KEY is not set")
    if not recipient:
        raise RuntimeError("RECIPIENT_EMAIL is not set")

    resend.api_key = api_key
    html = md_lib.markdown(
        markdown_content,
        extensions=["extra"],   # tables, fenced code, footnotes
    )

    response = resend.Emails.send({
        "from": FROM_ADDRESS,
        "to": [recipient],
        "subject": subject,
        "html": html,
        "text": markdown_content,
    })
    return response["id"]
