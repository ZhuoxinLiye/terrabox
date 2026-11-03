"""Synthetic handlers verify execution/manifest consistency, not model behavior."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from terrabox.core.schemas import ExecuteRequestIn
from terrabox.core.services import tool_service
from terrabox.core.services.connection_service import ConnectionService
from terrabox.core.services.tool_service import ToolService, explicit_result_error


@pytest.mark.parametrize("payload", [
    {"success": False, "error": {"code": "invalid_scope", "message": "scope missing"}},
    {"ok": False, "reason_code": "invalid_scope"},
    {"execution_blocked": True},
    {"passed": False},
    {"success": True, "outputs": {"success": False, "error": "nested failure"}},
])
def test_failed_handler_keeps_outputs_and_logs_error(monkeypatch, payload):
    tool = SimpleNamespace(parameters={}, requires_connection=False)
    monkeypatch.setattr(tool_service, "get_tool", lambda _: tool)
    monkeypatch.setattr(tool_service, "get_handler", lambda _: lambda args, ctx: payload)
    monkeypatch.setattr(tool_service, "list_toolkits", lambda: [])
    monkeypatch.setattr(ConnectionService, "select_connection", lambda *_: None)
    monkeypatch.setattr(tool_service, "prepare_runtime_context", lambda *a, **kw: {"execution_id": "synthetic"})
    monkeypatch.setattr(tool_service, "apply_default_output_paths", lambda slug, args, *rest: args)
    log = Mock()
    manifest = Mock(return_value="/synthetic/manifest.json")
    monkeypatch.setattr(ToolService, "_log_tool_execution", log)
    monkeypatch.setattr(tool_service, "write_manifest", manifest)
    result = asyncio.run(ToolService.execute_tool(Mock(), "synthetic", "synthetic.tool", ExecuteRequestIn(inputs={})))
    assert result.success is False
    assert result.outputs == payload
    assert result.error
    assert log.call_args.args[7] is False
    assert log.call_args.args[8] == result.error
    assert manifest.call_args.kwargs["status"] == "error"
    assert manifest.call_args.kwargs["outputs"] == payload
    assert manifest.call_args.kwargs["error"] == result.error


@pytest.mark.parametrize("payload", [
    {"count": 0, "nodes": [], "success": True},
    {"result": 0}, {}, {"observation": {"ok": False}}, {"passed": 0},
])
def test_zero_empty_and_observation_flags_are_not_execution_errors(payload):
    assert explicit_result_error(payload) is None


def test_zero_result_is_successful_in_response_log_and_manifest(monkeypatch):
    payload = {"success": True, "ok": True, "count": 0, "nodes": []}
    monkeypatch.setattr(tool_service, "get_tool", lambda _: SimpleNamespace(parameters={}, requires_connection=False))
    monkeypatch.setattr(tool_service, "get_handler", lambda _: lambda args, ctx: payload)
    monkeypatch.setattr(tool_service, "list_toolkits", lambda: [])
    monkeypatch.setattr(tool_service, "prepare_runtime_context", lambda *a, **kw: {"execution_id": "synthetic"})
    monkeypatch.setattr(tool_service, "apply_default_output_paths", lambda slug, args, *rest: args)
    log, manifest = Mock(), Mock(return_value="/synthetic/manifest.json")
    monkeypatch.setattr(ToolService, "_log_tool_execution", log)
    monkeypatch.setattr(tool_service, "write_manifest", manifest)
    result = asyncio.run(ToolService.execute_tool(Mock(), "synthetic", "synthetic.tool", ExecuteRequestIn(inputs={})))
    assert result.success is True and result.error is None
    assert result.outputs["count"] == 0
    assert log.call_args.args[7] is True
    assert manifest.call_args.kwargs["status"] == "success"
