import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.xpander.ai/v1"
NOTION_CONNECTOR_NAME = "Notion"

# Read-focused operations for requirement analysis.
DEFAULT_OPERATION_NAMES = (
    "Search for Content",
    "Retrieve Page Properties",
    "Retrieve Block Children",
    "Retrieve a Block",
    "Query a Database",
    "Retrieve a Database",
)


class NotionConnectorError(RuntimeError):
    """Raised when Notion connector setup against xpander fails."""


def _api_key() -> str:
    api_key = os.environ.get("XPANDER_API_KEY")
    if not api_key:
        raise NotionConnectorError("Set XPANDER_API_KEY in .env")
    return api_key


def _headers() -> dict[str, str]:
    return {"x-api-key": _api_key(), "Content-Type": "application/json"}


async def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
) -> Any:
    url = f"{API_BASE}{path}"
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method,
            url,
            headers=_headers(),
            params=params,
            json=json,
            timeout=60.0,
        )
        if not response.is_success:
            raise NotionConnectorError(
                f"{method} {path} failed ({response.status_code}): {response.text}"
            )
        if not response.content:
            return None
        return response.json()


async def find_notion_connector() -> dict[str, Any]:
    connector_id = os.environ.get("XPANDER_NOTION_CONNECTOR_ID")
    if connector_id:
        return {"id": connector_id, "name": NOTION_CONNECTOR_NAME}

    data = await _request(
        "GET",
        "/tools",
        params={"type": "connector", "query": "notion", "per_page": 50},
    )
    for item in data.get("items", []):
        if item.get("name") == NOTION_CONNECTOR_NAME:
            return item

    raise NotionConnectorError(
        f"Could not find '{NOTION_CONNECTOR_NAME}' in the xpander connector catalog"
    )


async def get_or_create_connection(connector_id: str) -> str:
    connection_id = os.environ.get("XPANDER_NOTION_CONNECTION_ID")
    if connection_id:
        return connection_id

    connector = await _request(
        "GET",
        "/tools",
        params={"type": "connector", "query": "notion", "per_page": 50},
    )
    for item in connector.get("items", []):
        if item.get("id") != connector_id:
            continue
        connections = item.get("connections") or []
        if connections:
            return connections[0]["id"]

    notion_token = os.environ.get("NOTION_TOKEN")
    if notion_token:
        result = await _request(
            "POST",
            f"/tools/connectors/{connector_id}/connect",
            json={
                "name": "QA Backend Notion",
                "access_scope": "organizational",
                "security": {
                    "authMethod": "apiKey",
                    "apiKey": notion_token,
                    "headerName": "Authorization",
                },
            },
        )
        if isinstance(result, dict) and result.get("id"):
            return result["id"]
        if isinstance(result, dict) and result.get("connection_id"):
            return result["connection_id"]
        raise NotionConnectorError(
            "Notion connection created but no connection id was returned"
        )

    raise NotionConnectorError(
        "No Notion connection found. Set XPANDER_NOTION_CONNECTION_ID, add NOTION_TOKEN, "
        "or create a connection in the xpander dashboard"
    )


async def list_connector_operations(
    connector_id: str, connection_id: str
) -> list[dict[str, Any]]:
    return await _request(
        "GET",
        f"/tools/connectors/{connector_id}/operations",
        params={"connection_id": connection_id},
    )


async def list_agent_tools(agent_id: str) -> list[dict[str, Any]]:
    return await _request("GET", f"/agents/{agent_id}/tools")


def _resolve_operation_ids(
    operations: list[dict[str, Any]],
    *,
    operation_names: tuple[str, ...] = DEFAULT_OPERATION_NAMES,
) -> list[str]:
    by_name = {
        (op.get("pretty_name") or op.get("operationId") or ""): op["id"]
        for op in operations
    }
    resolved: list[str] = []
    for name in operation_names:
        operation_id = by_name.get(name)
        if operation_id:
            resolved.append(operation_id)
    if not resolved:
        raise NotionConnectorError(
            "None of the default Notion operations were found for this connection"
        )
    return resolved


async def attach_operations_to_agent(
    agent_id: str,
    connection_id: str,
    operation_ids: list[str],
    *,
    deploy: bool = True,
) -> list[dict[str, Any]]:
    if not operation_ids:
        return await list_agent_tools(agent_id)

    return await _request(
        "POST",
        f"/agents/{agent_id}/tools",
        params={"deploy": str(deploy).lower()},
        json={
            "type": "action",
            "connection_id": connection_id,
            "operation_ids": operation_ids,
        },
    )


async def ensure_notion_connector(agent_id: str) -> dict[str, Any]:
    """Ensure the requirement analysis agent has Notion tools attached."""
    connector = await find_notion_connector()
    connector_id = connector["id"]
    connection_id = await get_or_create_connection(connector_id)

    operations = await list_connector_operations(connector_id, connection_id)
    desired_operation_ids = _resolve_operation_ids(operations)

    existing_tools = await list_agent_tools(agent_id)
    attached_operation_ids = {
        tool["operation_id"]
        for tool in existing_tools
        if tool.get("type") == "action" and tool.get("connection_id") == connection_id
    }
    missing_operation_ids = [
        op_id for op_id in desired_operation_ids if op_id not in attached_operation_ids
    ]

    if missing_operation_ids:
        await attach_operations_to_agent(
            agent_id,
            connection_id,
            missing_operation_ids,
        )

    tools = await list_agent_tools(agent_id)
    notion_tools = [
        tool
        for tool in tools
        if tool.get("connection_id") == connection_id
        and tool.get("operation_id") in desired_operation_ids
    ]

    return {
        "connector_id": connector_id,
        "connector_name": connector.get("name", NOTION_CONNECTOR_NAME),
        "connection_id": connection_id,
        "attached_tools": [
            {
                "id": tool.get("id"),
                "name": tool.get("name"),
                "operation_id": tool.get("operation_id"),
            }
            for tool in notion_tools
        ],
        "newly_attached": len(missing_operation_ids),
    }
