#!/usr/bin/env python3
"""One-time script: attach Notion connector tools to the reqs agent in Xpander.

Uses an existing Notion connection from your Xpander org (XPANDER_NOTION_CONNECTION_ID).
Requires XPANDER_API_KEY and REQ_AGENT in .env.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

from connectors.notion import NotionConnectorError, ensure_notion_connector

load_dotenv(_ROOT / ".env")


def _agent_id() -> str:
    agent_id = os.environ.get("REQ_AGENT", "").strip()
    if not agent_id:
        raise RuntimeError("Set REQ_AGENT in .env")
    return agent_id


async def main() -> None:
    agent_id = _agent_id()
    print(f"Attaching Notion connector to reqs agent {agent_id}...")
    result = await ensure_notion_connector(agent_id)
    print(json.dumps(result, indent=2))
    print(
        f"\nConnection ID (optional — save to XPANDER_NOTION_CONNECTION_ID): "
        f"{result['connection_id']}"
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (NotionConnectorError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
