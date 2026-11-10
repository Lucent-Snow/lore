"""WorkBuddy AI coding assistant session extractor.

Extracts chat history from WorkBuddy:
- ~/.workbuddy/projects/<project-slug>/<session_id>.jsonl
- Enriches metadata (titles, cwd) from ~/.workbuddy/workbuddy.db
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..core.models import Role, TitleSource, Tool, UnifiedMessage, UnifiedSession
from ..utils.datetime import make_naive, parse_timestamp
from ..utils.paths import make_thread_id
from .base import BaseExtractor

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_LINE_CHARS = 8 * 1024 * 1024
_MAX_PATH_CHARS = 4096
_MAX_TOOL_ARG_CHARS = 500
_MAX_TOOL_OUTPUT_CHARS = 20_000
_MAX_REASONING_CHARS = 20_000


def _safe_ts(value: Any, fallback: datetime) -> datetime:
    """Parse a timestamp safely from untrusted input."""
    if value is None or value == "" or value == 0:
        return fallback
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > 1e11:  # milliseconds timestamp
                return make_naive(datetime.fromtimestamp(value / 1000.0, tz=timezone.utc))
            elif value > 0:
                return make_naive(datetime.fromtimestamp(float(value), tz=timezone.utc))
        parsed = make_naive(parse_timestamp(value))
        if parsed.year < 2000:
            return fallback
        return parsed
    except (OverflowError, OSError, ValueError, TypeError, RecursionError):
        return fallback


def _clean_path(value: Any) -> Optional[str]:
    """Return value if it is a plain absolute path without nulls or control characters."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > _MAX_PATH_CHARS or any(c in value for c in "\x00\r\n"):
        return None
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        return None
    return value


def _clean_str(value: Any) -> Optional[str]:
    """Return non-empty string or None."""
    if isinstance(value, str) and value:
        return value
    return None


def _extract_user_text(raw_text: str) -> tuple[str, str]:
    """Extract clean user text from WorkBuddy user prompt.

    WorkBuddy often wraps system instructions and identities into:
    <system-reminder data-role="user-context">...</system-reminder>
    followed by:
    <user_query>actual user prompt</user_query>

    Returns (clean_text, kind) where kind is 'text' or 'injected_context'.
    """
    if not raw_text:
        return "", "text"

    # Check for background task notifications
    if raw_text.strip().startswith("<task-notification>"):
        return raw_text.strip(), "injected_context"

    # Try extracting <user_query>...</user_query>
    query_match = re.search(r"<user_query>(.*?)</user_query>", raw_text, re.DOTALL)
    if query_match:
        extracted = query_match.group(1).strip()
        if extracted:
            return extracted, "text"

    # Strip system reminders if present
    stripped = re.sub(
        r"<system-reminder[^>]*>.*?</system-reminder>", "", raw_text, flags=re.DOTALL
    ).strip()
    if stripped:
        return stripped, "text"

    return raw_text.strip(), "text"


class WorkBuddyExtractor(BaseExtractor):
    """Extracts chat history from WorkBuddy JSONL transcripts and SQLite DB."""

    tool = Tool.WORKBUDDY
    PARSER_VERSION = 1

    def __init__(
        self,
        projects_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        if projects_dir is not None:
            self.projects_dir = projects_dir
        else:
            self.projects_dir = Path.home() / ".workbuddy" / "projects"

        if db_path is not None:
            self.db_path = db_path
        else:
            self.db_path = Path.home() / ".workbuddy" / "workbuddy.db"

        self._db_meta_cache: Optional[Dict[str, Dict[str, Any]]] = None

    def source_description(self) -> str:
        return "~/.workbuddy/projects/*/*.jsonl"

    def is_available(self) -> bool:
        """Check if WorkBuddy data directory or SQLite DB exists."""
        return self.projects_dir.is_dir() or self.db_path.is_file()

    def _load_db_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load session metadata from workbuddy.db safely."""
        if self._db_meta_cache is not None:
            return self._db_meta_cache

        meta: Dict[str, Dict[str, Any]] = {}
        if not self.db_path.is_file():
            self._db_meta_cache = meta
            return meta

        try:
            # Use SQLite in read-only mode with timeout
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=2.0,
            )
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, cwd, title, custom_title, created_at, updated_at, model
                FROM sessions
                WHERE deleted_at IS NULL OR deleted_at = -1
                """
            )
            for row in cursor.fetchall():
                sid = row["id"]
                if not sid:
                    continue
                meta[sid] = {
                    "cwd": row["cwd"],
                    "title": row["title"],
                    "custom_title": row["custom_title"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "model": row["model"],
                }
            conn.close()
        except Exception as e:
            logger.debug("Failed to read workbuddy.db: %s", e)

        self._db_meta_cache = meta
        return meta

    def _discover_jsonl_files(self) -> List[Path]:
        """Find all session .jsonl files under projects/."""
        if not self.projects_dir.is_dir():
            return []
        files = []
        try:
            for p in self.projects_dir.iterdir():
                if p.is_dir():
                    for jf in p.glob("*.jsonl"):
                        if jf.is_file():
                            files.append(jf)
        except OSError as e:
            logger.warning("Error scanning WorkBuddy projects directory: %s", e)
        return files

    def extract_sessions(self) -> Iterator[UnifiedSession]:
        """Extract all conversation sessions from WorkBuddy."""
        db_meta = self._load_db_metadata()
        jsonl_files = self._discover_jsonl_files()
        self._count("files_matched", len(jsonl_files))

        for file_path in jsonl_files:
            if self._source_unchanged(file_path):
                continue
            session = self._parse_jsonl_file(file_path, db_meta)
            if session:
                self._count("files_parsed", 1)
                yield session

    def find_session_by_id(self, session_id: str) -> Optional[UnifiedSession]:
        """Find a specific session by its session_id."""
        db_meta = self._load_db_metadata()
        target_name = f"{session_id}.jsonl"
        if self.projects_dir.is_dir():
            for p in self.projects_dir.iterdir():
                if p.is_dir():
                    candidate = p / target_name
                    if candidate.is_file():
                        return self._parse_jsonl_file(candidate, db_meta)
        return None

    def _parse_jsonl_file(
        self,
        file_path: Path,
        db_meta: Dict[str, Dict[str, Any]],
    ) -> Optional[UnifiedSession]:
        """Parse one WorkBuddy session .jsonl file."""
        try:
            stat = file_path.stat()
            if stat.st_size > _MAX_FILE_BYTES:
                return None
        except OSError:
            return None

        file_mtime = datetime.fromtimestamp(stat.st_mtime)
        session_id = file_path.stem
        cwd: Optional[str] = None
        ai_title: Optional[str] = None

        db_entry = db_meta.get(session_id, {})
        if db_entry:
            cwd = db_entry.get("cwd")

        created_at: Optional[datetime] = None
        last_updated: Optional[datetime] = None

        def touch_ts(ts: datetime) -> None:
            nonlocal created_at, last_updated
            if created_at is None or ts < created_at:
                created_at = ts
            if last_updated is None or ts > last_updated:
                last_updated = ts

        messages: List[UnifiedMessage] = []
        # Track pending function calls by callId or id to match results
        pending_tool_calls: Dict[str, Dict[str, Any]] = {}

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if len(line) > _MAX_LINE_CHARS:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except (ValueError, RecursionError):
                        self._count("records_unknown")
                        continue
                    if not isinstance(obj, dict):
                        continue

                    rec_type = obj.get("type")
                    rec_ts = _safe_ts(obj.get("timestamp"), file_mtime)
                    touch_ts(rec_ts)

                    if rec_type == "session-meta":
                        sid = obj.get("sessionId")
                        if sid and isinstance(sid, str):
                            session_id = sid
                        if not cwd and obj.get("cwd"):
                            cwd = _clean_path(obj.get("cwd"))

                    elif rec_type == "ai-title":
                        title_val = obj.get("aiTitle")
                        if title_val and isinstance(title_val, str):
                            ai_title = title_val.strip()
                        if not cwd and obj.get("cwd"):
                            cwd = _clean_path(obj.get("cwd"))

                    elif rec_type == "message":
                        role_str = obj.get("role")
                        content_val = obj.get("content", [])
                        raw_text = ""
                        if isinstance(content_val, list):
                            for b in content_val:
                                if isinstance(b, dict):
                                    raw_text += b.get("text", "")
                        elif isinstance(content_val, str):
                            raw_text = content_val

                        msg_id = obj.get("id") or f"{session_id}_m{len(messages)}"

                        if role_str == "user":
                            clean_text, kind = _extract_user_text(raw_text)
                            role = Role.USER if kind == "text" else Role.TOOL
                            messages.append(
                                UnifiedMessage(
                                    role=role,
                                    content=clean_text,
                                    timestamp=rec_ts,
                                    message_id=msg_id,
                                    kind=kind,
                                )
                            )
                        elif role_str == "assistant":
                            # Flush any accumulated tool calls into this assistant message
                            t_calls = list(pending_tool_calls.values())
                            pending_tool_calls.clear()

                            messages.append(
                                UnifiedMessage(
                                    role=Role.ASSISTANT,
                                    content=raw_text,
                                    timestamp=rec_ts,
                                    message_id=msg_id,
                                    tool_calls=t_calls,
                                )
                            )

                    elif rec_type == "function_call":
                        call_id = (
                            obj.get("callId") or obj.get("id") or f"call_{len(pending_tool_calls)}"
                        )
                        name = obj.get("name") or "tool"
                        raw_args = obj.get("arguments") or {}
                        if isinstance(raw_args, str):
                            try:
                                args_dict = json.loads(raw_args)
                            except Exception:
                                args_dict = {"raw": raw_args[:_MAX_TOOL_ARG_CHARS]}
                        elif isinstance(raw_args, dict):
                            args_dict = raw_args
                        else:
                            args_dict = {}

                        prov = obj.get("providerData", {})
                        model_name = prov.get("model") or prov.get("requestModelId")
                        reasoning_val = prov.get("reasoning")
                        raw_usage = prov.get("rawUsage") or {}

                        tokens = None
                        if isinstance(raw_usage, dict):
                            tokens = {
                                "input": raw_usage.get("prompt_tokens", 0),
                                "output": raw_usage.get("completion_tokens", 0),
                                "total": raw_usage.get("total_tokens", 0),
                            }

                        call_record = {
                            "id": call_id,
                            "tool": name,
                            "input": args_dict,
                            "output": None,
                            "status": "completed",
                        }
                        pending_tool_calls[call_id] = call_record

                        # If there is reasoning, attach it to a preceding or new assistant turn
                        if reasoning_val:
                            msg_id = obj.get("id") or f"{session_id}_r{len(messages)}"
                            messages.append(
                                UnifiedMessage(
                                    role=Role.ASSISTANT,
                                    content="",
                                    timestamp=rec_ts,
                                    message_id=msg_id,
                                    reasoning=str(reasoning_val)[:_MAX_REASONING_CHARS],
                                    model=model_name,
                                    tokens=tokens,
                                    tool_calls=[call_record],
                                )
                            )

                    elif rec_type == "function_call_result":
                        call_id = obj.get("callId")
                        out_val = obj.get("output")
                        result_text = ""
                        if isinstance(out_val, dict):
                            result_text = out_val.get("text", "")
                        elif isinstance(out_val, str):
                            result_text = out_val
                        if len(result_text) > _MAX_TOOL_OUTPUT_CHARS:
                            result_text = result_text[:_MAX_TOOL_OUTPUT_CHARS] + "..."

                        if call_id and call_id in pending_tool_calls:
                            pending_tool_calls[call_id]["output"] = result_text

                    elif rec_type == "reasoning":
                        prov = obj.get("providerData", {})
                        reasoning_val = prov.get("reasoning")
                        model_name = prov.get("model") or prov.get("requestModelId")
                        if reasoning_val:
                            messages.append(
                                UnifiedMessage(
                                    role=Role.ASSISTANT,
                                    content="",
                                    timestamp=rec_ts,
                                    message_id=obj.get("id") or f"{session_id}_rea{len(messages)}",
                                    reasoning=str(reasoning_val)[:_MAX_REASONING_CHARS],
                                    model=model_name,
                                )
                            )

        except Exception as e:
            logger.warning("Error reading WorkBuddy transcript %s: %s", file_path, e)
            return None

        # Flush any trailing pending tool calls
        if pending_tool_calls:
            messages.append(
                UnifiedMessage(
                    role=Role.ASSISTANT,
                    content="",
                    timestamp=last_updated or file_mtime,
                    tool_calls=list(pending_tool_calls.values()),
                )
            )

        if not messages:
            return None

        # Resolve title: custom_title > aiTitle > title in DB > first user message
        title = None
        title_source = TitleSource.FALLBACK
        if db_entry.get("custom_title"):
            title = db_entry["custom_title"]
            title_source = TitleSource.NATIVE
        elif ai_title:
            title = ai_title
            title_source = TitleSource.SUMMARY
        elif db_entry.get("title"):
            title = db_entry["title"]
            title_source = TitleSource.SUMMARY
        else:
            first_user = next(
                (m.content for m in messages if m.role == Role.USER and m.content.strip()), None
            )
            if first_user:
                first_line = first_user.strip().split("\n")[0][:100]
                title = first_line
                title_source = TitleSource.FIRST_MESSAGE

        created_dt = created_at or file_mtime
        updated_dt = last_updated or file_mtime

        return UnifiedSession(
            tool=self.tool,
            session_id=session_id,
            created_at=created_dt,
            last_updated=updated_dt,
            messages=messages,
            project_path=cwd,
            title=title,
            title_source=title_source,
            source_path=str(file_path),
            thread_id=make_thread_id(session_id),
        )
