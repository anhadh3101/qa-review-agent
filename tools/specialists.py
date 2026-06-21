from xpander_sdk import register_tool

from utils.agent_invoke import invoke_and_wait, require_env


@register_tool(add_to_graph=True)
async def run_code_analysis(prompt: str) -> str:
    """Analyze code, pull requests, and implementation details using the code analysis specialist.

    Use for: PR diffs, GitHub review, code quality, bugs, security issues in changed files.
    Pass a self-contained prompt with repo/PR context and what to analyze.
    """
    agent_id = require_env("XPANDER_CODE_ANALYSIS_AGENT")
    return await invoke_and_wait(
        agent_id,
        prompt,
        source="qa-coordinator-code-analysis",
    )


@register_tool(add_to_graph=True)
async def run_requirement_analysis(prompt: str) -> str:
    """Check requirements, specs, and acceptance criteria using the requirement analysis specialist.

    Use for: Notion specs, user stories, acceptance criteria, requirement coverage vs implementation.
    Pass a self-contained prompt describing what requirements to validate.
    """
    agent_id = require_env("XPANDER_REQUIREMENT_ANALYSIS_AGENT")
    return await invoke_and_wait(
        agent_id,
        prompt,
        source="qa-coordinator-requirement-analysis",
    )
