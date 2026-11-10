"""Shared paths and read-only connections. No writes to source DBs.

Paths are read from a JSON file outside the repository, so no local path or account name is committed:
  $TOPIC_MOBILITY_PATHS  or  ~/.config/korea-topic-mobility/paths.json
Keys: web_db, koad_db, openalex_cache, release_work_dir, private_root.
All outputs (raw professor_uid / work ids) go to <private_root>/outputs, never into this repository.
"""
import json
import os
import sqlite3
from pathlib import Path

_cfg_path = Path(os.environ.get("TOPIC_MOBILITY_PATHS",
                                Path.home() / ".config/korea-topic-mobility/paths.json"))
if not _cfg_path.exists():
    raise SystemExit(f"path config not found: {_cfg_path} (see analysis/README.md)")
_cfg = json.loads(_cfg_path.read_text())

WEB = Path(_cfg["web_db"])
KOAD = Path(_cfg["koad_db"])
CACHE = Path(_cfg["openalex_cache"])
RELEASE_WORK = Path(_cfg["release_work_dir"])
ROOT = Path(_cfg["private_root"])
OUT = ROOT / "outputs"
REFS_DB = ROOT / "refs/work_references.sqlite"
JOURNAL_TYPES = ("article", "review", "letter", "editorial")
MAX_YEAR = 2026


def ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def short(w):
    return (w or "").rstrip("/").rsplit("/", 1)[-1]
