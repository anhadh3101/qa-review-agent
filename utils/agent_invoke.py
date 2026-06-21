import asyncio
import os

from xpander_sdk import AgentExecutionStatus, Backend, Tasks, User

POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 120

_TERMINAL_STATUSES = {
    AgentExecutionStatus.Completed,
    AgentExecutionStatus.Failed,
    AgentExecutionStatus.Error,
    AgentExecutionStatus.Stopped,
}


async def invoke_and_wait(
    agent_id: str,
    prompt: str,
    *,
    source: str = "qa-backend",
    timeout: float = POLL_TIMEOUT_SECONDS,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> str:
    """Invoke an Xpander agent and poll until the task reaches a terminal state."""
    backend = Backend()
    tasks = Tasks()
    task = await backend.ainvoke_agent(
        agent_id=agent_id,
        prompt=prompt,
        source=source,
        user_details=User(id="qa-backend", email="qa-backend@localhost"),
    )

    elapsed = 0.0
    while task.status not in _TERMINAL_STATUSES:
        if elapsed >= timeout:
            raise TimeoutError(
                f"Agent task {task.id} did not finish within {timeout}s"
            )
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        task = await tasks.aget(task_id=task.id)

    if task.status != AgentExecutionStatus.Completed:
        raise RuntimeError(
            f"Agent {agent_id} finished with status {task.status.value}: {task.result}"
        )
    return task.result or ""


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in .env")
    return value
