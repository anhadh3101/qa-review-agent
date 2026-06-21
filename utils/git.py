import os, httpx

TOKEN = os.getenv("GITHUB_TOKEN")

"""
Get the PR diff to display in the frontend.
"""
async def fetch_pr_diff(diff_url: str) -> str:
    headers = {}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    async with httpx.AsyncClient() as client:
        response = await client.get(diff_url, headers=headers, follow_redirects=True)
        response.raise_for_status()
        return response.text
