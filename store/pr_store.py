from datetime import datetime, timezone

_prs: dict[str, dict] = {}


def _key(owner: str, repo: str, number: int) -> str:
    return f"{owner}/{repo}/{number}"


def save_pr(
    *,
    owner: str,
    repo: str,
    number: int,
    title: str,
    base_ref: str,
    head_ref: str,
    html_url: str,
    diff: str,
) -> dict:
    record = {
        "owner": owner,
        "repo": repo,
        "number": number,
        "title": title,
        "base_ref": base_ref,
        "head_ref": head_ref,
        "html_url": html_url,
        "diff": diff,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _prs[_key(owner, repo, number)] = record
    return record


def list_prs() -> list[dict]:
    items = list(_prs.values())
    items.sort(key=lambda pr: pr["updated_at"], reverse=True)
    return [
        {k: v for k, v in pr.items() if k != "diff"}
        for pr in items
    ]


def get_pr(owner: str, repo: str, number: int) -> dict | None:
    return _prs.get(_key(owner, repo, number))
