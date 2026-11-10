"""Tests for the WorkBuddy extractor."""

from __future__ import annotations

import json

from lore.core.models import Role, Tool
from lore.extractors.workbuddy import WorkBuddyExtractor, _extract_user_text


def test_extract_user_text_with_user_query():
    raw = (
        '<system-reminder data-role="user-context">\n'
        "<user_info>OS: darwin</user_info>\n"
        "</system-reminder>\n"
        "<user_query>请帮我排查一下这个问题</user_query>"
    )
    clean, kind = _extract_user_text(raw)
    assert clean == "请帮我排查一下这个问题"
    assert kind == "text"


def test_extract_user_text_strips_system_reminder():
    raw = (
        '<system-reminder data-role="user-context">\n'
        "<user_info>OS: darwin</user_info>\n"
        "</system-reminder>\n"
        "直接看下项目的README"
    )
    clean, kind = _extract_user_text(raw)
    assert clean == "直接看下项目的README"
    assert kind == "text"


def test_extract_user_text_task_notification():
    raw = "<task-notification>\n<task-id>123</task-id>\n</task-notification>"
    clean, kind = _extract_user_text(raw)
    assert kind == "injected_context"


def test_workbuddy_extractor_parse_session(tmp_path):
    projects_dir = tmp_path / "projects"
    project_slug = projects_dir / "Users-test-myproject"
    project_slug.mkdir(parents=True)

    session_id = "test-session-uuid-1"
    session_file = project_slug / f"{session_id}.jsonl"

    records = [
        {
            "type": "session-meta",
            "sessionId": session_id,
            "timestamp": 1792498173000,
            "cwd": "/Users/test/myproject",
        },
        {
            "type": "ai-title",
            "sessionId": session_id,
            "aiTitle": "测试排查问题会话",
            "cwd": "/Users/test/myproject",
            "timestamp": 1792498174000,
        },
        {
            "type": "message",
            "id": "m1",
            "role": "user",
            "timestamp": 1792498175000,
            "content": [
                {
                    "type": "input_text",
                    "text": "<system-reminder>sys info</system-reminder>\n<user_query>帮我看下代码</user_query>",
                }
            ],
        },
        {
            "type": "reasoning",
            "id": "r1",
            "timestamp": 1792498176000,
            "providerData": {
                "model": "deepseek-v4.1-flash",
                "reasoning": "User wants to inspect code.",
            },
        },
        {
            "type": "function_call",
            "id": "fc1",
            "callId": "call_1",
            "name": "ReadFile",
            "arguments": '{"path": "README.md"}',
            "timestamp": 1792498177000,
            "providerData": {
                "model": "deepseek-v4.1-flash",
            },
        },
        {
            "type": "function_call_result",
            "callId": "call_1",
            "name": "ReadFile",
            "output": {"text": "# Project README"},
            "timestamp": 1792498178000,
        },
        {
            "type": "message",
            "id": "m2",
            "role": "assistant",
            "timestamp": 1792498179000,
            "content": [
                {
                    "type": "output_text",
                    "text": "这是你的项目说明文档。",
                }
            ],
        },
    ]

    session_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    extractor = WorkBuddyExtractor(projects_dir=projects_dir, db_path=tmp_path / "workbuddy.db")
    assert extractor.is_available() is True

    sessions = list(extractor.extract_sessions())
    assert len(sessions) == 1
    session = sessions[0]

    assert session.tool == Tool.WORKBUDDY
    assert session.session_id == session_id
    assert session.title == "测试排查问题会话"
    assert session.project_path == "/Users/test/myproject"

    user_msgs = [m for m in session.messages if m.role == Role.USER]
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "帮我看下代码"

    asst_msgs = [m for m in session.messages if m.role == Role.ASSISTANT]
    assert len(asst_msgs) >= 1
    last_asst = asst_msgs[-1]
    assert "这是你的项目说明文档。" in last_asst.content

    # Check find_session_by_id
    found = extractor.find_session_by_id(session_id)
    assert found is not None
    assert found.session_id == session_id
