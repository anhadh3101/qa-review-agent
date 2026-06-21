#!/usr/bin/env python3
"""One-time script: attach GitHub PR connector tools to the code agent in Xpander.

Uses an existing GitHub connection from your Xpander org (no connection ID required).
Requires XPANDER_API_KEY and CODE_AGENT in .env.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from connectors.git_pr_connector import GitHubConnectorError, ensure_github_connector

load_dotenv()

AGENT_ID = os.getenv("CODE_AGENT")


async def main() -> None:
    agent_id = AGENT_ID
    print(f"Attaching GitHub PR connector to code agent {agent_id}...")
    result = await ensure_github_connector(agent_id)
    print(json.dumps(result, indent=2))
    print(f"\nConnection ID (optional — save to XPANDER_GITHUB_CONNECTION_ID): {result['connection_id']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (GitHubConnectorError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
