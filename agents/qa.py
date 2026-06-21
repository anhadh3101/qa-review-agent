import asyncio, importlib.util, os, sys, uuid
from operator import add
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from xpander_sdk import Agents
from xpander_sdk.exceptions.module_exception import ModuleException


load_dotenv()

AGENT_ID = os.getenv("QA_AGENT")
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 120

SPECIALIST_TOOL_NAMES = frozenset(
    {"run_code_analysis", "run_requirement_analysis"}
)

_AGENTS_DIR = Path(__file__).parent

def create_system_prompt(instructions) -> str:
    parts = []
    if getattr(instructions, "general", None):
        parts.append(f"System: {instructions.general}")
    if getattr(instructions, "goal_str", None):
        parts.append(f"Goals:\n{instructions.goal_str}")
    if getattr(instructions, "instructions", None):
        instr_list = "\n".join(f"- {i}" for i in instructions.instructions)
        parts.append(f"Instructions:\n{instr_list}")
    return "\n\n".join(parts) or "You are a QA coordinator."

async def load_xpander_agent():
    agents = Agents()
    return await agents.aget(AGENT_ID)

def build_react_agent(xpander_agent):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = create_system_prompt(xpander_agent.instructions)
    # No tools for now — testing orchestrator only
    graph = create_react_agent(llm, [])
    return graph, system_prompt

def _extract_final_content(result: dict) -> str:
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", None)
        if content:
            return content
    return ""

async def run_orchestrator(user_prompt: str) -> dict:
    xpander_agent = await load_xpander_agent()
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
    
async def setup_specialist_tools() -> dict:
    """Register and sync specialist delegation tools onto the QA coordinator agent."""
    agents = Agents()
    agent = await agents.aget(AGENT_ID)
    local_tools = agent.tools.get_local_tools_for_sync()
    if local_tools:
        await agent.sync_local_tools(tools=local_tools)
        agent = await agents.aget(AGENT_ID)
    tool_names = {tool.name for tool in agent.tools.list}
    return {
        "agent_id": AGENT_ID,
        "agent_name": agent.name,
        "synced_tools": [tool.name for tool in local_tools],
        "agent_tools": sorted(tool_names),
        "missing_tools": sorted(SPECIALIST_TOOL_NAMES - tool_names),
    }
    
async def test_connection() -> dict:
    return await run_orchestrator("Reply with exactly: connection ok")

async def run_agent(prompt: str) -> dict:
    return await run_orchestrator(prompt)

def build_pr_review_prompt(record: dict) -> str:
    diff = record.get("diff", "")
    max_diff_chars = 80_000
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + "\n\n[diff truncated]"
    return (
        f"Perform a QA review for this pull request.\n\n"
        f"Title: {record['title']}\n"
        f"Repository: {record['owner']}/{record['repo']}\n"
        f"PR number: {record['number']}\n"
        f"Base branch: {record['base_ref']}\n"
        f"Head branch: {record['head_ref']}\n"
        f"URL: {record['html_url']}\n\n"
        f"Diff:\n```diff\n{diff}\n```"
    )
    
# ---------------------------------------------------------------------------
# HITL supervisor graph
#
# The QA coordinator is built as an explicit StateGraph: a supervisor node
# routes to specialist subgraphs (code / requirements) and pauses via
# interrupt() for human approval before each delegation. Compiled with a
# checkpointer so runs can be paused and resumed by thread_id.
# ---------------------------------------------------------------------------

_SPECIALISTS = {"code": "code.py", "requirements": "reqs.py"}

_specialist_modules: dict[str, object] = {}
_specialist_cache: dict[str, tuple] = {}
_coordinator_cache: dict[str, str] = {}
_qa_graph = None
_checkpointer = InMemorySaver()


class QAState(TypedDict):
    messages: Annotated[list, add_messages]
    next: Optional[str]
    completed: Annotated[list[str], add]


class Route(BaseModel):
    next: Literal["code", "requirements", "end"]
    reason: str


def _load_sibling(module_name: str, filename: str):
    # Load by file path so the local agents/ folder does not shadow the
    # OpenAI "agents" package required by xpander_sdk.
    path = _AGENTS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_specialist_module(name: str):
    if name not in _specialist_modules:
        _specialist_modules[name] = _load_sibling(
            f"qa_specialist_{name}", _SPECIALISTS[name]
        )
    return _specialist_modules[name]


async def _get_specialist(name: str) -> tuple:
    if name not in _specialist_cache:
        module = _get_specialist_module(name)
        xpander_agent = await module.load_xpander_agent()
        graph, system_prompt = module.build_react_agent(xpander_agent)
        _specialist_cache[name] = (graph, system_prompt, xpander_agent.name)
    return _specialist_cache[name]


async def _get_coordinator_prompt() -> str:
    if "prompt" not in _coordinator_cache:
        xpander_agent = await load_xpander_agent()
        _coordinator_cache["prompt"] = create_system_prompt(
            xpander_agent.instructions
        )
        _coordinator_cache["name"] = xpander_agent.name
    return _coordinator_cache["prompt"]


def _original_request(state: QAState) -> str:
    for message in state["messages"]:
        if getattr(message, "type", None) == "human":
            return message.content
    return _extract_final_content(state)


async def _run_specialist_node(name: str, state: QAState) -> dict:
    graph, system_prompt, _ = await _get_specialist(name)
    task = _original_request(state)
    result = await graph.ainvoke(
        {"messages": [("system", system_prompt), ("user", task)]}
    )
    answer = _extract_final_content(result)
    return {
        "messages": [("ai", f"[{name} specialist]\n{answer}")],
        "completed": [name],
    }


async def code_node(state: QAState) -> dict:
    return await _run_specialist_node("code", state)


async def requirements_node(state: QAState) -> dict:
    return await _run_specialist_node("requirements", state)


def _routing_context(coordinator_prompt: str, completed: list[str]) -> str:
    return (
        f"{coordinator_prompt}\n\n"
        "You coordinate two specialists:\n"
        "- code: analyzes code, diffs, bugs, security, and code quality.\n"
        "- requirements: checks specs, acceptance criteria, and requirement "
        "coverage.\n\n"
        f"Specialists already consulted: {completed or 'none'}.\n"
        "Decide which specialist to delegate to next, or 'end' when the review "
        "is complete. Avoid repeating a specialist that already ran unless it "
        "is clearly necessary."
    )


async def supervisor_node(state: QAState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    coordinator_prompt = await _get_coordinator_prompt()
    completed = state.get("completed", [])
    context = _routing_context(coordinator_prompt, completed)

    decision = await llm.with_structured_output(Route).ainvoke(
        [("system", context), *state["messages"]]
    )

    if decision.next == "end":
        final = await llm.ainvoke(
            [
                (
                    "system",
                    context
                    + "\n\nProduce a final, consolidated QA review that "
                    "summarizes the specialists' findings for the user.",
                ),
                *state["messages"],
            ]
        )
        return {"next": "end", "messages": [final]}

    # HITL: pause for human approval before delegating to a specialist.
    approval = interrupt(
        {
            "type": "approve_delegation",
            "proposed_next": decision.next,
            "reason": decision.reason,
            "completed": completed,
        }
    )

    action = (approval or {}).get("action", "approve")
    if action == "reject":
        return {"next": "end"}
    if action == "override" and (approval or {}).get("next"):
        return {"next": approval["next"]}
    return {"next": decision.next}


def _route(state: QAState) -> str:
    return state["next"]


def build_supervisor_graph(checkpointer):
    builder = StateGraph(QAState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("code", code_node)
    builder.add_node("requirements", requirements_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route,
        {"code": "code", "requirements": "requirements", "end": END},
    )
    builder.add_edge("code", "supervisor")
    builder.add_edge("requirements", "supervisor")

    return builder.compile(checkpointer=checkpointer)


def get_qa_graph():
    global _qa_graph
    if _qa_graph is None:
        _qa_graph = build_supervisor_graph(_checkpointer)
    return _qa_graph


def _interrupt_payload(result: dict):
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


def _format_result(result: dict, thread_id: str) -> dict:
    payload = _interrupt_payload(result)
    if payload is not None:
        return {
            "status": "paused",
            "thread_id": thread_id,
            "request": payload,
        }
    return {
        "status": "completed",
        "thread_id": thread_id,
        "result": _extract_final_content(result),
    }


async def run_qa(prompt: str, thread_id: Optional[str] = None) -> dict:
    """Start a QA coordinator run; may pause for human approval (HITL)."""
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = await get_qa_graph().ainvoke(
        {"messages": [("user", prompt)], "completed": []}, config=config
    )
    return _format_result(result, thread_id)


async def resume_qa(thread_id: str, value) -> dict:
    """Resume a paused QA coordinator run with the human's response."""
    config = {"configurable": {"thread_id": thread_id}}
    result = await get_qa_graph().ainvoke(Command(resume=value), config=config)
    return _format_result(result, thread_id)


async def run_qa_review(record: dict, thread_id: Optional[str] = None) -> dict:
    result = await run_qa(build_pr_review_prompt(record), thread_id)
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