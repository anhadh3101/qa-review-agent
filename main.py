import importlib.util
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from store import pr_store
from utils.git import fetch_pr_diff

import tools  # noqa: F401 — registers coordinator specialist tools on import

load_dotenv(Path(__file__).parent / ".env")

_code_analysis = None
_requirement_analysis = None
_qa_coordinator = None


def _load_agent_module(module_name: str, filename: str):
    path = Path(__file__).parent / "agents" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_code_analysis():
    global _code_analysis
    if _code_analysis is not None:
        return _code_analysis
    # Load by file path so the local agents/ folder does not shadow the
    # OpenAI "agents" package required by xpander_sdk.
    _code_analysis = _load_agent_module("code_analysis", "code.py")
    return _code_analysis


def _get_requirement_analysis():
    global _requirement_analysis
    if _requirement_analysis is not None:
        return _requirement_analysis
    _requirement_analysis = _load_agent_module(
        "requirement_analysis", "reqs.py"
    )
    return _requirement_analysis


def _get_qa_coordinator():
    global _qa_coordinator
    if _qa_coordinator is not None:
        return _qa_coordinator
    _qa_coordinator = _load_agent_module("qa_coordinator", "qa.py")
    return _qa_coordinator


logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PR_DIFF_ACTIONS = {"opened", "synchronize", "reopened"}


class CoordinatorChatRequest(BaseModel):
    prompt: str


def _handle_agent_error(exc: Exception) -> HTTPException:
    if type(exc).__name__ == "ModuleException":
        return HTTPException(
            status_code=502,
            detail=f"xpander API error {exc.status_code}: {exc.description}",
        )
    return HTTPException(status_code=502, detail=str(exc))


@app.get("/api/prs")
async def list_prs():
    return pr_store.list_prs()


@app.get("/api/prs/{owner}/{repo}/{pr_number}")
async def get_pr(owner: str, repo: str, pr_number: int):
    record = pr_store.get_pr(owner, repo, pr_number)
    if record is None:
        raise HTTPException(status_code=404, detail="PR not found")
    return record


@app.post("/api/agent/notion-connector")
async def agent_notion_connector():
    try:
        requirement_analysis = _get_requirement_analysis()
        return await requirement_analysis.setup_notion_connector()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/agent/requirement-test")
async def agent_requirement_test():
    try:
        requirement_analysis = _get_requirement_analysis()
        return await requirement_analysis.test_connection()
    except Exception as exc:
        raise _handle_agent_error(exc) from exc


@app.post("/api/agent/github-connector")
async def agent_github_connector():
    try:
        code_analysis = _get_code_analysis()
        return await code_analysis.setup_github_connector()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/agent/test")
async def agent_test():
    try:
        code_analysis = _get_code_analysis()
        return await code_analysis.test_connection()
    except Exception as exc:
        raise _handle_agent_error(exc) from exc


@app.post("/api/agent/coordinator/chat")
async def coordinator_chat(body: CoordinatorChatRequest):
    """Chat with the QA coordinator agent."""
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        qa_coordinator = _get_qa_coordinator()
        return await qa_coordinator.run_agent(prompt)
    except Exception as exc:
        raise _handle_agent_error(exc) from exc


@app.post("/api/agent/code/chat")
async def code_agent_chat(body: CoordinatorChatRequest):
    """Chat with the code analysis specialist agent."""
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        code_analysis = _get_code_analysis()
        return await code_analysis.run_agent(prompt)
    except Exception as exc:
        raise _handle_agent_error(exc) from exc


@app.post("/api/agent/requirement/chat")
async def requirement_agent_chat(body: CoordinatorChatRequest):
    """Chat with the requirement analysis specialist agent."""
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        requirement_analysis = _get_requirement_analysis()
        return await requirement_analysis.run_agent(prompt)
    except Exception as exc:
        raise _handle_agent_error(exc) from exc


@app.get("/api/agent/coordinator-test")
async def agent_coordinator_test():
    try:
        qa_coordinator = _get_qa_coordinator()
        return await qa_coordinator.test_connection()
    except Exception as exc:
        raise _handle_agent_error(exc) from exc


@app.post("/api/agent/specialist-tools")
async def agent_specialist_tools():
    try:
        qa_coordinator = _get_qa_coordinator()
        return await qa_coordinator.setup_specialist_tools()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/prs/{owner}/{repo}/{pr_number}/analyze")
async def analyze_pr(owner: str, repo: str, pr_number: int):
    record = pr_store.get_pr(owner, repo, pr_number)
    if record is None:
        raise HTTPException(status_code=404, detail="PR not found")
    try:
        qa_coordinator = _get_qa_coordinator()
        return await qa_coordinator.run_qa_review(record)
    except Exception as exc:
        raise _handle_agent_error(exc) from exc


@app.post("/github/webhook")
async def github_webhook(request: Request):
    event_type = request.headers.get("X-GitHub-Event", "")
    payload = await request.json()

    if event_type == "pull_request" and payload.get("action") in PR_DIFF_ACTIONS:
        pr = payload["pull_request"]
        repo = payload["repository"]
        owner = repo["owner"]["login"]
        repo_name = repo["name"]

        diff = await fetch_pr_diff(pr["diff_url"])
        record = pr_store.save_pr(
            owner=owner,
            repo=repo_name,
            number=pr["number"],
            title=pr["title"],
            base_ref=pr["base"]["ref"],
            head_ref=pr["head"]["ref"],
            html_url=pr["html_url"],
            diff=diff,
        )

        logger.info(
            "Stored diff for PR #%s (%s/%s), %d bytes",
            pr["number"],
            owner,
            repo_name,
            len(diff),
        )

        return {
            "status": "ok",
            "event": event_type,
            "action": payload["action"],
            "pr_number": record["number"],
            "base_ref": record["base_ref"],
            "head_ref": record["head_ref"],
            "diff_bytes": len(diff),
        }

    return {"status": "ok", "event": event_type, "action": payload.get("action")}
