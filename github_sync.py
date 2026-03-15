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


def push_file(local_path: Path, repo_path: str, message: str = "Auto-sync data") -> bool:
    """Push a single file to GitHub."""
    if not is_enabled() or not local_path.exists():
        return False

    global _last_sync
    now = time.time()
    if now - _last_sync < _MIN_SYNC_INTERVAL:
        time.sleep(_MIN_SYNC_INTERVAL - (now - _last_sync))

    try:
        content = local_path.read_bytes()
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
    """Push all data files to GitHub."""
    if not is_enabled():
        return

    from fund_manager import DATA_DIR
    for repo_path in SYNC_FILES:
        filename = repo_path.split("/")[-1]
        local_path = DATA_DIR / filename
        if local_path.exists():
            sync_file(local_path, repo_path, message)


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
