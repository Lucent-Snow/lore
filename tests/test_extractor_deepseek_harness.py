"""Tests for the DeepSeek Harness (dsh) extractor."""

from __future__ import annotations

import json

from lore.core.models import Role, Tool
from lore.extractors.deepseek_harness import DeepSeekHarnessExtractor


def test_deepseek_harness_parse_uncompressed_session(tmp_path):
    sessions_root = tmp_path / "sessions"
    slug_dir = sessions_root / "--Users-test-myproject--"
    session_dir = slug_dir / "session-dsh-test-1"
    session_dir.mkdir(parents=True)

    session_file = session_dir / "session.jsonl"

    records = [
        {
            "type": "session",
            "id": "session-dsh-test-1",
            "createdAt": 1786788478000,
            "cwd": "/Users/test/myproject",
        },
        {
            "type": "session/title",
            "seq": 1,
            "time": 1786788479000,
            "data": {
                "title": "测试 DSH 会话标题",
            },
        },
        {
            "type": "user/message",
            "seq": 2,
            "time": 1786788480000,
            "data": {
                "id": "u1",
                "content": [{"type": "text", "text": "帮我看看这个模块的设计"}],
                "source": {"kind": "user"},
            },
        },
        {
            "type": "assistant/message",
            "seq": 3,
            "time": 1786788481000,
            "data": {
                "message": {
                    "id": "a1",
                    "content": [
                        {
                            "type": "tool-call",
                            "id": "call_123",
                            "name": "bash",
                            "arguments": '{"command": "ls -la"}',
                        }
                    ],
                    "source": {"model": "deepseek-chat"},
                }
            },
        },
        {
            "type": "tool/result",
            "seq": 4,
            "time": 1786788482000,
            "data": {
                "toolCallId": "call_123",
                "message": {
                    "content": [{"type": "text", "text": "file1.py\nfile2.py"}],
                },
            },
        },
        {
            "type": "assistant/message",
            "seq": 5,
            "time": 1786788483000,
            "data": {
                "message": {
                    "id": "a2",
                    "content": [{"type": "text", "text": "该模块设计良好，目录包含 file1.py。"}],
                    "source": {"model": "deepseek-chat"},
                }
            },
        },
    ]

    session_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    extractor = DeepSeekHarnessExtractor(base_paths=[sessions_root])
    assert extractor.is_available() is True

    sessions = list(extractor.extract_sessions())
    assert len(sessions) == 1
    session = sessions[0]

    assert session.tool == Tool.DEEPSEEK_HARNESS
    assert session.session_id == "session-dsh-test-1"
    assert session.title == "测试 DSH 会话标题"
    assert session.project_path == "/Users/test/myproject"

    user_msgs = [m for m in session.messages if m.role == Role.USER]
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "帮我看看这个模块的设计"

    asst_msgs = [m for m in session.messages if m.role == Role.ASSISTANT]
    assert len(asst_msgs) >= 1
    assert any("该模块设计良好" in m.content for m in asst_msgs)

    # Check find_session_by_id
    found = extractor.find_session_by_id("session-dsh-test-1")
    assert found is not None
    assert found.session_id == "session-dsh-test-1"
