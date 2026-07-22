"""
Tests for the optional LLM refinement step (llm_refine.py).

These use a mocked Anthropic client — no network, no API key required. Run:

    pytest test_llm_refine.py            # if pytest is installed
    python  test_llm_refine.py           # standalone fallback runner

Coverage:
  (a) no API key set               -> no-op, output unchanged ("not_configured")
  (b) heuristic parse complete     -> needs_refinement() False (API not called)
  (c) key set + parse incomplete   -> API called, response merged correctly
  (d) API raises                   -> caught, heuristic result unchanged ("failed")
"""
import json
import types

import config
import llm_refine


# --- test doubles ----------------------------------------------------------
class _Block:
    """Mimics an Anthropic text content block."""
    type = "text"

    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


def _fake_client_returning(json_text):
    """A stand-in Anthropic() client whose messages.create returns json_text."""
    client = types.SimpleNamespace()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _Resp(json_text)

    client.messages = types.SimpleNamespace(create=create)
    client.calls = calls
    return client


def _fake_client_raising():
    client = types.SimpleNamespace()

    def create(**kwargs):
        raise RuntimeError("boom — no network / expired key")

    client.messages = types.SimpleNamespace(create=create)
    return client


# --- (a) no API key set -> no-op -------------------------------------------
def test_no_api_key_is_noop(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    parsed = {"name": None, "id_number": None, "extra_fields": {"line_0": "FOO"}}

    result, status = llm_refine.maybe_refine(["FOO"], parsed)

    assert status == "not_configured"
    assert result == parsed  # completely unchanged


# --- (b) complete parse -> API not needed ----------------------------------
def test_complete_parse_does_not_need_refinement():
    complete = {
        "name": "JOHN A SMITH",
        "id_number": "123456789",
        "extra_fields": {},
    }
    assert llm_refine.needs_refinement(complete) is False


def test_incomplete_parses_need_refinement():
    assert llm_refine.needs_refinement(
        {"name": None, "id_number": "1", "extra_fields": {}}
    ) is True
    assert llm_refine.needs_refinement(
        {"name": "X", "id_number": None, "extra_fields": {}}
    ) is True
    assert llm_refine.needs_refinement(
        {"name": "X", "id_number": "1", "extra_fields": {"a": "1", "b": "2", "c": "3"}}
    ) is True


# --- (c) key set + incomplete -> API called, merged correctly --------------
def test_refinement_applied_and_merged(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-test")

    llm_json = json.dumps(
        {
            "name": "RODNEY G MULLINS",
            "id_number": "66J9-V66-FY35",
            "dob": None,
            "expiry": None,
            "issue_date": None,
            "sex": None,
            "address": None,
            "organization": "MEDICARE HEALTH INSURANCE",
            "group_number": None,
            "dates": [],
            "extra_fields": {},
        }
    )
    client = _fake_client_returning(llm_json)
    monkeypatch.setattr(llm_refine, "Anthropic", lambda: client)

    # Heuristic got id + a payer match, but missed the name.
    parsed = {
        "name": None,
        "id_number": "66J9-V66-FY35",
        "organization": "MEDICARE HEALTH INSURANCE",
        "extra_fields": {"line_1": "RODNEY G MULLINS"},
        "payer_match": None,  # must survive the merge untouched
    }
    raw_lines = ["MEDICARE HEALTH INSURANCE", "RODNEY G MULLINS", "66J9-V66-FY35"]

    result, status = llm_refine.maybe_refine(raw_lines, parsed)

    assert status == "applied"
    assert result["name"] == "RODNEY G MULLINS"          # LLM filled the gap
    assert result["id_number"] == "66J9-V66-FY35"        # preserved
    assert "payer_match" in result and result["payer_match"] is None  # untouched
    # exactly one call, and the image / payer DB were never sent
    assert len(client.calls) == 1
    sent = client.calls[0]
    assert set(json.loads(sent["messages"][0]["content"]).keys()) == {
        "raw_lines",
        "fields",
    }


def test_merge_does_not_blank_out_heuristic_fields(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-test")
    # Sparse LLM response: name null, everything empty.
    llm_json = json.dumps({"name": None, "id_number": None, "extra_fields": {}})
    monkeypatch.setattr(
        llm_refine, "Anthropic", lambda: _fake_client_returning(llm_json)
    )

    parsed = {"name": "GOOD NAME", "id_number": "123", "extra_fields": {}}
    result, status = llm_refine.maybe_refine(["GOOD NAME", "123"], parsed)

    assert status == "applied"
    assert result["name"] == "GOOD NAME"   # NOT blanked by the null LLM value
    assert result["id_number"] == "123"


# --- (d) API raises -> clean fallback --------------------------------------
def test_api_exception_falls_back(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(llm_refine, "Anthropic", lambda: _fake_client_raising())

    parsed = {"name": None, "id_number": None, "extra_fields": {"line_0": "X"}}
    result, status = llm_refine.maybe_refine(["X"], parsed)

    assert status == "failed"
    assert result == parsed  # unchanged


def test_malformed_json_falls_back(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(
        llm_refine, "Anthropic", lambda: _fake_client_returning("not json at all {")
    )

    parsed = {"name": None, "id_number": "1", "extra_fields": {}}
    result, status = llm_refine.maybe_refine(["1"], parsed)

    assert status == "failed"
    assert result == parsed


def test_strips_markdown_fences(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-test")
    fenced = "```json\n" + json.dumps({"name": "FENCED NAME"}) + "\n```"
    monkeypatch.setattr(
        llm_refine, "Anthropic", lambda: _fake_client_returning(fenced)
    )

    parsed = {"name": None, "id_number": "1", "extra_fields": {}}
    result, status = llm_refine.maybe_refine(["FENCED NAME"], parsed)

    assert status == "applied"
    assert result["name"] == "FENCED NAME"


# --- standalone runner (no pytest required) --------------------------------
if __name__ == "__main__":
    import inspect

    class _MP:
        """Minimal monkeypatch shim so the tests run without pytest."""

        def __init__(self):
            self._undo = []

        def setattr(self, target, name, value):
            old = getattr(target, name)
            self._undo.append((target, name, old))
            setattr(target, name, value)

        def undo(self):
            for target, name, old in reversed(self._undo):
                setattr(target, name, old)
            self._undo.clear()

    tests = [
        (n, f)
        for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    passed = 0
    for name, fn in tests:
        mp = _MP()
        try:
            if "monkeypatch" in inspect.signature(fn).parameters:
                fn(mp)
            else:
                fn()
            print(f"PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {name}: {e!r}")
        finally:
            mp.undo()
    print(f"\n{passed}/{len(tests)} passed")
