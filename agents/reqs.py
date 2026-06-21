import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from xpander_sdk import Agents
from xpander_sdk.exceptions.module_exception import ModuleException

from connectors.notion import ensure_notion_connector, list_agent_tools

load_dotenv()

AGENT_ID = os.getenv("REQ_AGENT") or os.getenv("XPANDER_REQUIREMENT_ANALYSIS_AGENT")


def create_system_prompt(instructions) -> str:
    parts = []
    if getattr(instructions, "general", None):
        parts.append(f"System: {instructions.general}")
    if getattr(instructions, "goal_str", None):
        parts.append(f"Goals:\n{instructions.goal_str}")
    if getattr(instructions, "instructions", None):
        instr_list = "\n".join(f"- {i}" for i in instructions.instructions)
        parts.append(f"Instructions:\n{instr_list}")
    return "\n\n".join(parts) or "You are a requirement analysis specialist."


async def load_xpander_agent():
    agents = Agents()
    return await agents.aget(AGENT_ID)


def build_react_agent(xpander_agent):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = create_system_prompt(xpander_agent.instructions)
    graph = create_react_agent(llm, [])
    return graph, system_prompt


def _extract_final_content(result: dict) -> str:
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", None)
        if content:
            return content
    return ""


async def _print_agent_connectors() -> None:
    tools = await list_agent_tools(AGENT_ID)
    connectors = [
        {
            "name": tool.get("name"),
            "id": tool.get("id"),
            "connection_id": tool.get("connection_id"),
            "operation_id": tool.get("operation_id"),
        }
        for tool in tools
        if tool.get("type") == "action" and tool.get("connection_id")
    ]
    print("Agent connectors:", connectors)


async def run_orchestrator(user_prompt: str) -> dict:
    xpander_agent = await load_xpander_agent()
    await _print_agent_connectors()
    graph, system_prompt = build_react_agent(xpander_agent)
    result = await graph.ainvoke({
        "messages": [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    })
    return {
        "connected": True,
        "agent_id": AGENT_ID,
        "agent_name": xpander_agent.name,
        "status": "completed",
        "result": _extract_final_content(result),
        "agent_tools": [tool.name for tool in xpander_agent.tools.list],
    }


async def setup_notion_connector() -> dict:
    """Attach Notion connector tools to the requirement analysis agent."""
    return await ensure_notion_connector(AGENT_ID)


async def test_connection() -> dict:
    return await run_orchestrator("Reply with exactly: connection ok")


async def run_agent(prompt: str) -> dict:
    return await run_orchestrator(prompt)


def build_requirement_review_prompt(record: dict) -> str:
    diff = record.get("diff", "")
    max_diff_chars = 80_000
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + "\n\n[diff truncated]"
    return (
        f"Perform a requirement analysis for this pull request.\n\n"
        f"Title: {record['title']}\n"
        f"Repository: {record['owner']}/{record['repo']}\n"
        f"PR number: {record['number']}\n"
        f"Base branch: {record['base_ref']}\n"
        f"Head branch: {record['head_ref']}\n"
        f"URL: {record['html_url']}\n\n"
        f"Focus on: requirement coverage, acceptance criteria, spec alignment, and gaps.\n"
        f"Use Notion tools to look up relevant specs where needed.\n\n"
        f"Diff:\n```diff\n{diff}\n```"
    )


async def run_requirement_review(record: dict) -> dict:
    result = await run_orchestrator(build_requirement_review_prompt(record))
    return {
        **result,
        "pr": {
            "owner": record["owner"],
            "repo": record["repo"],
            "number": record["number"],
            "html_url": record["html_url"],
        },
    }


async def _main() -> None:
    try:
        result = await test_connection()
        print("Connection test passed:", result)
    except ModuleException as exc:
        print(f"xpander API error {exc.status_code}: {exc.description}")
    except Exception as exc:
        print(f"Connection test failed: {exc}")


if __name__ == "__main__":
    asyncio.run(_main())
