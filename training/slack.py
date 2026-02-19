"""Shared Slack notification helper for training scripts.

Usage:
    from slack import slack_notify

    slack_notify("Training started!")           # Uses SLACK_WEBHOOK_URL env var
    slack_notify("Done!", webhook_url="https://...")  # Explicit URL
"""

import json
import os
import urllib.request


def slack_notify(message: str, webhook_url: str = ""):
    """Send a Slack notification. Fails silently if no webhook configured.

    Args:
        message: Slack message text (supports mrkdwn formatting)
        webhook_url: Explicit webhook URL. If empty, reads SLACK_WEBHOOK_URL env var.
    """
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
    if not url:
        return

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": message}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[slack] failed: {e}")
