"""
GitHub Sync - Persistenza dei dati su GitHub per Streamlit Cloud.
Ogni modifica ai file dati viene sincronizzata come commit nel repo.
Quando l'app si riavvia, i dati sono sempre aggiornati dal repo.

Setup: aggiungi in .streamlit/secrets.toml:
  [github]
  token = "ghp_xxxxxxxxxxxx"
  repo = "Sbirrondi/sfc-portfolio-tracker"
"""

import base64
import json
import time
from pathlib import Path
from typing import Optional

# Files to sync (relative to repo root)
SYNC_FILES = [
    "data/fund_transactions.csv",
    "data/fund_positions.csv",
    "data/fund_nav_history.csv",
    "data/fund_info.json",
    "data/fund_cash.json",
    "data/isin_map.json",
    "data/overrides.json",
]

_github_config = None
_last_sync = 0
_MIN_SYNC_INTERVAL = 2  # seconds between syncs to avoid rate limits


def _get_config() -> Optional[dict]:
    """Load GitHub config from Streamlit secrets."""
    global _github_config
    if _github_config is not None:
        return _github_config

    try:
        import streamlit as st
        if hasattr(st, "secrets") and "github" in st.secrets:
            _github_config = {
                "token": st.secrets["github"]["token"],
                "repo": st.secrets["github"]["repo"],
            }
            return _github_config
    except Exception:
        pass

    _github_config = {}  # Empty = disabled
    return _github_config


def is_enabled() -> bool:
    """Check if GitHub sync is configured."""
    config = _get_config()
    return bool(config and config.get("token") and config.get("repo"))


def _api_request(method: str, endpoint: str, data: dict = None) -> dict:
    """Make a GitHub API request."""
    import urllib.request
    import urllib.error

    config = _get_config()
    if not config:
        return {}

    url = f"https://api.github.com/repos/{config['repo']}/{endpoint}"
    headers = {
        "Authorization": f"token {config['token']}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SFC-Portfolio-Tracker",
    }

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"GitHub API error {e.code}: {error_body[:200]}")
        return {"error": e.code, "message": error_body}
    except Exception as e:
        print(f"GitHub API request failed: {e}")
        return {"error": str(e)}


def _get_file_sha(file_path: str) -> Optional[str]:
    """Get the SHA of a file in the repo (needed for updates)."""
    result = _api_request("GET", f"contents/{file_path}")
    if "sha" in result:
        return result["sha"]
    return None


def push_file(local_path: Path, repo_path: str, message: str = "Auto-sync data",
              content_bytes: bytes = None) -> bool:
    """Push a single file to GitHub.

    If content_bytes is provided, push that content instead of reading from disk.
    This avoids race conditions when Streamlit Cloud pulls overwrite local files.
    """
    if not is_enabled():
        return False
    if content_bytes is None and not local_path.exists():
        return False

    global _last_sync
    now = time.time()
    if now - _last_sync < _MIN_SYNC_INTERVAL:
        time.sleep(_MIN_SYNC_INTERVAL - (now - _last_sync))

    try:
        content = content_bytes if content_bytes is not None else local_path.read_bytes()
        encoded = base64.b64encode(content).decode()

        # Get current SHA (needed for update, None for create)
        sha = _get_file_sha(repo_path)

        data = {
            "message": message,
            "content": encoded,
            "branch": "main",
        }
        if sha:
            data["sha"] = sha

        result = _api_request("PUT", f"contents/{repo_path}", data)
        _last_sync = time.time()

        if "content" in result:
            return True
        else:
            print(f"Push failed for {repo_path}: {result}")
            return False
    except Exception as e:
        print(f"Push error for {repo_path}: {e}")
        return False


def sync_file(local_path: Path, repo_path: str, message: str = None):
    """Sync a local file to GitHub (fire-and-forget)."""
    if not is_enabled():
        return

    if message is None:
        message = f"Update {repo_path}"

    try:
        push_file(local_path, repo_path, message)
    except Exception as e:
        # Never let sync errors break the app
        print(f"Sync error (non-blocking): {e}")


def sync_data_file(filename: str, message: str = None):
    """Convenience: sync a file from the data/ directory."""
    from fund_manager import DATA_DIR
    local_path = DATA_DIR / filename
    repo_path = f"data/{filename}"
    sync_file(local_path, repo_path, message or f"Update {filename}")


def sync_all_data(message: str = "Sync all data"):
    """Push all data files to GitHub in a SINGLE commit.

    Uses the Git Data API (blobs → tree → commit → update ref) so only one
    commit is created. This avoids triggering multiple Streamlit Cloud
    redeploys that would kill the script mid-sync and cause race conditions
    (KeyError on module reload, stale NAV, etc.).
    """
    if not is_enabled():
        return

    from fund_manager import DATA_DIR

    # Pre-read all files into memory
    files_to_push = []
    for repo_path in SYNC_FILES:
        filename = repo_path.split("/")[-1]
        local_path = DATA_DIR / filename
        if local_path.exists():
            files_to_push.append((repo_path, local_path.read_bytes()))

    if not files_to_push:
        return

    global _last_sync
    now = time.time()
    if now - _last_sync < _MIN_SYNC_INTERVAL:
        time.sleep(_MIN_SYNC_INTERVAL - (now - _last_sync))

    try:
        # 1. Get current branch ref
        ref_data = _api_request("GET", "git/ref/heads/main")
        if "object" not in ref_data:
            print(f"sync_all_data: failed to get ref: {ref_data}")
            return
        current_commit_sha = ref_data["object"]["sha"]

        # 2. Get current commit to find its tree
        commit_data = _api_request("GET", f"git/commits/{current_commit_sha}")
        if "tree" not in commit_data:
            print(f"sync_all_data: failed to get commit: {commit_data}")
            return
        base_tree_sha = commit_data["tree"]["sha"]

        # 3. Create a blob for each file
        tree_entries = []
        for repo_path, content in files_to_push:
            blob_data = _api_request("POST", "git/blobs", {
                "content": base64.b64encode(content).decode(),
                "encoding": "base64",
            })
            if "sha" not in blob_data:
                print(f"sync_all_data: failed to create blob for {repo_path}: {blob_data}")
                return
            tree_entries.append({
                "path": repo_path,
                "mode": "100644",
                "type": "blob",
                "sha": blob_data["sha"],
            })

        # 4. Create a new tree based on the current one
        tree_data = _api_request("POST", "git/trees", {
            "base_tree": base_tree_sha,
            "tree": tree_entries,
        })
        if "sha" not in tree_data:
            print(f"sync_all_data: failed to create tree: {tree_data}")
            return
        new_tree_sha = tree_data["sha"]

        # 5. Create a new commit
        new_commit_data = _api_request("POST", "git/commits", {
            "message": message,
            "tree": new_tree_sha,
            "parents": [current_commit_sha],
        })
        if "sha" not in new_commit_data:
            print(f"sync_all_data: failed to create commit: {new_commit_data}")
            return
        new_commit_sha = new_commit_data["sha"]

        # 6. Update branch ref to point to the new commit
        update_data = _api_request("PATCH", "git/refs/heads/main", {
            "sha": new_commit_sha,
        })
        if "error" in update_data:
            print(f"sync_all_data: failed to update ref: {update_data}")
            return

        _last_sync = time.time()
    except Exception as e:
        print(f"sync_all_data error (non-blocking): {e}")


def get_sync_status() -> dict:
    """Get the current sync status for display."""
    if not is_enabled():
        return {"enabled": False, "message": "GitHub sync non configurato"}

    config = _get_config()
    return {
        "enabled": True,
        "repo": config.get("repo", ""),
        "message": f"Sincronizzato con {config.get('repo', '')}",
    }
