import json

from lore.core.models import Role, Tool
from lore.extractors.claude import ClaudeCodeExtractor
from lore.extractors.codex import CodexExtractor
from lore.extractors.deepseek_harness import DeepSeekHarnessExtractor
from lore.extractors.gemini import GeminiCLIExtractor
from lore.extractors.workbuddy import WorkBuddyExtractor


def test_claude_parse_session_contract(tmp_path):
    session_file = tmp_path / "session.jsonl"
    records = [
        {"type": "summary", "summary": "Refactor parser"},
        {
            "type": "user",
            "timestamp": "2026-03-01T10:00:00",
            "message": {"content": "Please refactor this"},
            "sessionId": "ses_claude_1",
            "uuid": "u1",
        },
        {
            "type": "assistant",
            "timestamp": "2026-03-01T10:01:00",
            "message": {"content": [{"type": "text", "text": "Done."}]},
            "sessionId": "ses_claude_1",
            "uuid": "a1",
        },
    ]
    session_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    extractor = ClaudeCodeExtractor()
    session = extractor._parse_session(session_file, "/repo/project")

    assert session.tool == Tool.CLAUDE_CODE
    assert session.session_id == "ses_claude_1"
    assert session.title == "Refactor parser"
    assert len(session.messages) == 2
    assert session.messages[0].role == Role.USER
    assert session.messages[1].role == Role.ASSISTANT


def test_claude_sanitizes_kryptic_summary_title(tmp_path):
    session_file = tmp_path / "session-kryptic.jsonl"
    records = [
        {
            "type": "summary",
            "summary": "<command-name>login</command-name> <command-message>login</command-message>",
        },
        {
            "type": "user",
            "timestamp": "2026-03-01T10:00:00",
            "message": {"content": "Please debug this issue"},
            "sessionId": "ses_claude_2",
            "uuid": "u1",
        },
    ]
    session_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    extractor = ClaudeCodeExtractor()
    session = extractor._parse_session(session_file, "/repo/project")

    assert session.title is None
    assert session.summary is None


def test_gemini_parse_session_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(GeminiCLIExtractor, "_build_hash_map", lambda self: None)
    session_file = tmp_path / "session-1.json"
    payload = {
        "sessionId": "ses_gemini_1",
        "startTime": "2026-03-01T09:00:00",
        "lastUpdated": "2026-03-01T09:05:00",
        "messages": [
            {
                "id": "m1",
                "type": "user",
                "timestamp": "2026-03-01T09:00:00",
                "content": "show files",
            },
            {
                "id": "m2",
                "type": "gemini",
                "timestamp": "2026-03-01T09:01:00",
                "content": "",
                "toolCalls": [{"name": "run_terminal_cmd", "args": {"command": "ls -la"}}],
            },
        ],
    }
    session_file.write_text(json.dumps(payload), encoding="utf-8")

    extractor = GeminiCLIExtractor()
    session = extractor._parse_session_file(session_file, "/repo/project", "hash123")

    assert session.tool == Tool.GEMINI_CLI
    assert session.session_id == "ses_gemini_1"
    assert len(session.messages) == 2
    assert session.messages[1].role == Role.ASSISTANT
    assert "[Tool: Bash]" in session.messages[1].content


def test_codex_parse_session_contract_with_malformed_lines(tmp_path):
    session_file = tmp_path / "rollout-1.jsonl"
    lines = [
        json.dumps(
            {
                "type": "session_meta",
                "timestamp": "2026-03-02T08:00:00",
                "payload": {"id": "ses_codex_1", "cwd": "/repo/project"},
            }
        ),
        "not-json",
        json.dumps(
            {
                "type": "message",
                "timestamp": "2026-03-02T08:01:00",
                "payload": {"id": "m1", "role": "user", "content": "help"},
            }
        ),
    ]
    session_file.write_text("\n".join(lines), encoding="utf-8")

    extractor = CodexExtractor()
    session = extractor._parse_session_file(session_file)

    assert session.tool == Tool.CODEX
    assert session.session_id == "ses_codex_1"
    assert len(session.messages) == 1
    assert session.messages[0].role == Role.USER


def test_workbuddy_parse_session_contract(tmp_path):
    projects_dir = tmp_path / "projects"
    slug_dir = projects_dir / "Users-test-wb"
    slug_dir.mkdir(parents=True)
    session_file = slug_dir / "wb_ses_1.jsonl"
    records = [
        {"type": "session-meta", "sessionId": "wb_ses_1", "cwd": "/repo/project"},
        {"type": "ai-title", "aiTitle": "Fix WorkBuddy issue"},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "<user_query>Hello from user</user_query>"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello! How can I help?"}],
        },
    ]
    session_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    extractor = WorkBuddyExtractor(projects_dir=projects_dir)
    session = extractor._parse_jsonl_file(session_file, {})
    assert session.tool == Tool.WORKBUDDY
    assert session.session_id == "wb_ses_1"
    assert session.title == "Fix WorkBuddy issue"
    assert len(session.messages) == 2
    assert session.messages[0].role == Role.USER
    assert session.messages[0].content == "Hello from user"
    assert session.messages[1].role == Role.ASSISTANT


def test_deepseek_harness_parse_session_contract(tmp_path):
    sessions_root = tmp_path / "sessions"
    session_dir = sessions_root / "--repo-project--" / "dsh_ses_1"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "session.jsonl"
    records = [
        {"type": "session", "id": "dsh_ses_1", "cwd": "/repo/project"},
        {"type": "session/title", "data": {"title": "DSH test title"}},
        {
            "type": "user/message",
            "data": {"content": [{"type": "text", "text": "Run DSH task"}]},
        },
        {
            "type": "assistant/message",
            "data": {"message": {"content": [{"type": "text", "text": "DSH task completed."}]}},
        },
    ]
    session_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    extractor = DeepSeekHarnessExtractor(base_paths=[sessions_root])
    session = extractor._parse_session_file(session_file)
    assert session.tool == Tool.DEEPSEEK_HARNESS
    assert session.session_id == "dsh_ses_1"
    assert session.title == "DSH test title"
    assert len(session.messages) == 2
    assert session.messages[0].role == Role.USER
    assert session.messages[0].content == "Run DSH task"
    assert session.messages[1].role == Role.ASSISTANT
