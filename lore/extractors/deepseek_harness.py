"""DeepSeek Harness (dsh) AI session extractor.

Extracts chat history from DeepSeek Harness:
- ~/.dsh/sessions/--<slug>--/<session_id>/session.v4.jsonl.zstd
- ~/.dsh/sessions/--<slug>--/<session_id>/session.jsonl.zstd
- Also supports uncompressed session.jsonl
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    import zstandard  # type: ignore[import-untyped]
except ImportError:
    zstandard = None

from ..core.models import Role, TitleSource, Tool, UnifiedMessage, UnifiedSession
from ..utils.datetime import make_naive, parse_timestamp
from ..utils.paths import make_thread_id
from .base import BaseExtractor

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_LINE_CHARS = 8 * 1024 * 1024
_MAX_PATH_CHARS = 4096
_MAX_TOOL_OUTPUT_CHARS = 20_000


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


def _decompress_zstd_file(path: Path) -> Optional[str]:
    """Decompress a .zstd file into string using python-zstandard or system zstd binary."""
    if zstandard is not None:
        try:
            dctx = zstandard.ZstdDecompressor()
            with open(path, "rb") as fh:
                return dctx.stream_reader(fh).read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug("python-zstandard failed on %s: %s", path, e)

    # Fallback to system zstd CLI if installed
    zstd_bin = shutil.which("zstd")
    if zstd_bin:
        try:
            res = subprocess.run(
                [zstd_bin, "-dc", str(path)],
                capture_output=True,
                check=False,
                timeout=10,
            )
            if res.returncode == 0:
                return res.stdout.decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug("system zstd failed on %s: %s", path, e)

    return None


class DeepSeekHarnessExtractor(BaseExtractor):
    """Extracts chat history from DeepSeek Harness session archives."""

    tool = Tool.DEEPSEEK_HARNESS
    PARSER_VERSION = 1

    def __init__(self, base_paths: Optional[List[Path]] = None) -> None:
        if base_paths is not None:
            self.base_paths = base_paths
        else:
            self.base_paths = [
                Path.home() / ".dsh" / "sessions",
                Path.home() / ".dsh-cc" / "sessions",
            ]

    def source_description(self) -> str:
        return "~/.dsh/sessions/*/*/session*.zstd"

    def is_available(self) -> bool:
        """Check if any DeepSeek Harness sessions directory exists."""
        return any(p.is_dir() for p in self.base_paths)

    def _discover_session_files(self) -> List[Path]:
        """Find candidate session files (.jsonl and .zstd)."""
        files: List[Path] = []
        for root in self.base_paths:
            if not root.is_dir():
                continue
            try:
                for slug_dir in root.iterdir():
                    if not slug_dir.is_dir():
                        continue
                    for session_dir in slug_dir.iterdir():
                        if not session_dir.is_dir():
                            continue
                        # Look for session.v4.jsonl.zstd, session.jsonl.zstd, session.jsonl
                        for candidate in session_dir.glob("session*.zstd"):
                            if candidate.is_file():
                                files.append(candidate)
                        for candidate in session_dir.glob("session*.jsonl"):
                            if candidate.is_file():
                                files.append(candidate)
            except OSError as e:
                logger.warning("Error discovering DeepSeek Harness sessions: %s", e)
        return files

    def extract_sessions(self) -> Iterator[UnifiedSession]:
        """Extract all conversation sessions from DeepSeek Harness."""
        files = self._discover_session_files()
        self._count("files_matched", len(files))

        for file_path in files:
            if self._source_unchanged(file_path):
                continue
            session = self._parse_session_file(file_path)
            if session:
                self._count("files_parsed", 1)
                yield session

    def find_session_by_id(self, session_id: str) -> Optional[UnifiedSession]:
        """Find a specific session by ID."""
        for root in self.base_paths:
            if not root.is_dir():
                continue
            for slug_dir in root.iterdir():
                target_dir = slug_dir / session_id
                if target_dir.is_dir():
                    for cand in target_dir.glob("session*.zstd"):
                        return self._parse_session_file(cand)
                    for cand in target_dir.glob("session*.jsonl"):
                        return self._parse_session_file(cand)
        return None

    def _parse_session_file(self, file_path: Path) -> Optional[UnifiedSession]:
        """Parse one DeepSeek Harness session file."""
        try:
            stat = file_path.stat()
            if stat.st_size > _MAX_FILE_BYTES:
                return None
        except OSError:
            return None

        file_mtime = datetime.fromtimestamp(stat.st_mtime)

        # Decompress or read text
        if file_path.suffix == ".zstd":
            content_text = _decompress_zstd_file(file_path)
            if content_text is None:
                self._count("records_unknown")
                return None
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                    content_text = fh.read()
            except OSError:
                return None

        session_id = file_path.parent.name
        cwd: Optional[str] = None
        title: Optional[str] = None
        created_at: Optional[datetime] = None
        last_updated: Optional[datetime] = None

        def touch_ts(ts: datetime) -> None:
            nonlocal created_at, last_updated
            if created_at is None or ts < created_at:
                created_at = ts
            if last_updated is None or ts > last_updated:
                last_updated = ts

        messages: List[UnifiedMessage] = []
        # Keep track of active tool calls to correlate with tool results
        active_tool_calls: Dict[str, Dict[str, Any]] = {}

        for line in content_text.splitlines():
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
            raw_time = obj.get("time") or obj.get("createdAt")
            if raw_time is not None:
                ts = _safe_ts(raw_time, file_mtime)
                if ts.year >= 2000:
                    touch_ts(ts)
            else:
                ts = file_mtime

            if rec_type == "session":
                sid = obj.get("id")
                if sid and isinstance(sid, str):
                    session_id = sid
                if obj.get("cwd"):
                    cwd = _clean_path(obj.get("cwd"))
                if obj.get("createdAt"):
                    created_at = _safe_ts(obj.get("createdAt"), file_mtime)

            elif rec_type == "session/title":
                data = obj.get("data", {})
                t_val = data.get("title")
                if t_val and isinstance(t_val, str):
                    title = t_val.strip()

            elif rec_type == "user/message":
                data = obj.get("data", {})
                content_list = data.get("content", [])
                text_parts: List[str] = []
                for b in content_list:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                    elif isinstance(b, str):
                        text_parts.append(b)
                raw_text = "".join(text_parts).strip()

                source = data.get("source", {})
                is_system = (
                    source.get("kind") == "system"
                    or raw_text.startswith("<system-reminder>")
                    or raw_text.startswith("Current runtime context")
                    or raw_text.startswith("Time sampled")
                    or raw_text.startswith("The approval policy changed")
                )

                if is_system:
                    role = Role.INFO
                    kind = "injected_context"
                else:
                    role = Role.USER
                    kind = "text"

                msg_id = data.get("id") or f"{session_id}_u{len(messages)}"
                messages.append(
                    UnifiedMessage(
                        role=role,
                        content=raw_text,
                        timestamp=ts,
                        message_id=msg_id,
                        kind=kind,
                    )
                )

            elif rec_type == "assistant/message":
                data = obj.get("data", {})
                msg_obj = data.get("message", {})
                model = msg_obj.get("source", {}).get("model")
                reasoning = msg_obj.get("reasoning")

                content_list = msg_obj.get("content", [])
                text_parts = []
                t_calls: List[Dict[str, Any]] = []

                for b in content_list:
                    if not isinstance(b, dict):
                        continue
                    b_type = b.get("type")
                    if b_type == "text":
                        text_parts.append(b.get("text", ""))
                    elif b_type == "tool-call":
                        cid = b.get("id") or b.get("callId")
                        name = b.get("name") or "tool"
                        raw_args = b.get("arguments") or {}
                        if isinstance(raw_args, str):
                            try:
                                args_dict = json.loads(raw_args)
                            except Exception:
                                args_dict = {"raw": raw_args}
                        elif isinstance(raw_args, dict):
                            args_dict = raw_args
                        else:
                            args_dict = {}

                        call_record = {
                            "id": cid,
                            "tool": name,
                            "input": args_dict,
                            "output": None,
                            "status": "completed",
                        }
                        if cid:
                            active_tool_calls[cid] = call_record
                        t_calls.append(call_record)

                text = "".join(text_parts).strip()
                # Only append assistant message if it has text or tool calls
                if text or t_calls or reasoning:
                    msg_id = msg_obj.get("id") or f"{session_id}_a{len(messages)}"
                    messages.append(
                        UnifiedMessage(
                            role=Role.ASSISTANT,
                            content=text,
                            timestamp=ts,
                            message_id=msg_id,
                            model=model,
                            reasoning=reasoning if isinstance(reasoning, str) else None,
                            tool_calls=t_calls,
                        )
                    )

            elif rec_type == "tool/result":
                data = obj.get("data", {})
                call_id = data.get("toolCallId")
                msg_obj = data.get("message", {})
                content_list = msg_obj.get("content", [])
                res_parts = []
                for b in content_list:
                    if isinstance(b, dict) and b.get("type") == "text":
                        res_parts.append(b.get("text", ""))
                    elif isinstance(b, str):
                        res_parts.append(b)
                res_text = "".join(res_parts)
                if len(res_text) > _MAX_TOOL_OUTPUT_CHARS:
                    res_text = res_text[:_MAX_TOOL_OUTPUT_CHARS] + "..."

                if call_id and call_id in active_tool_calls:
                    active_tool_calls[call_id]["output"] = res_text

        if not messages:
            return None

        # Title resolution
        title_source = TitleSource.FALLBACK
        if title:
            title_source = TitleSource.NATIVE
        else:
            first_user = next(
                (m.content for m in messages if m.role == Role.USER and m.content.strip()), None
            )
            if first_user:
                first_line = first_user.strip().split("\n")[0][:100]
                title = first_line
                title_source = TitleSource.FIRST_MESSAGE

        if not created_at or created_at.year < 2000:
            created_dt = (
                messages[0].timestamp
                if messages and messages[0].timestamp.year >= 2000
                else file_mtime
            )
        else:
            created_dt = created_at

        if not last_updated or last_updated.year < 2000:
            updated_dt = (
                messages[-1].timestamp
                if messages and messages[-1].timestamp.year >= 2000
                else file_mtime
            )
        else:
            updated_dt = last_updated

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
