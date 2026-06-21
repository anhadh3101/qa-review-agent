#!/usr/bin/env python3
"""Attach specialist agents to the QA coordinator via the xpander REST API.

Usage (from qa-backend/):
    ./venv/bin/python scripts/attach_coordinator_specialists.py

Requires in .env:
    XPANDER_API_KEY
    XPANDER_QA_COORDINATOR
    XPANDER_CODE_ANALYSIS_AGENT
    XPANDER_REQUIREMENT_ANALYSIS_AGENT
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from connectors.coordinator import CoordinatorSetupError, ensure_coordinator_specialists


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach specialist sub-agents to the QA coordinator agent."
    )
    parser.add_argument(
        "--coordinator-id",
        help="Override XPANDER_QA_COORDINATOR from .env",
    )
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Pass deploy=false so the agent is not redeployed after attaching",
    )
    args = parser.parse_args()

    try:
        result = await ensure_coordinator_specialists(
            coordinator_id=args.coordinator_id,
            deploy=not args.no_deploy,
        )
    except CoordinatorSetupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))

    if result["missing_specialists"]:
        print(
            f"\nWarning: still missing: {', '.join(result['missing_specialists'])}",
            file=sys.stderr,
        )
        return 1

    if result["newly_attached"]:
        print(f"\nAttached: {', '.join(result['newly_attached'])}")
    else:
        print("\nAll specialists were already attached.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
