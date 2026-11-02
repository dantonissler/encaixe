import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "bakeoff", Path(__file__).resolve().parents[2] / "scripts" / "bakeoff.py"
)
bakeoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bakeoff)


def test_summarize_and_table():
    data = {"results": {"results": [
        {"success": True, "latencyMs": 1000, "testCase": {"metadata": {"category": "rag"}}},
        {"success": False, "latencyMs": 3000, "testCase": {"metadata": {"category": "rag"}}},
        {"success": True, "latencyMs": 2000, "testCase": {"metadata": {"category": "acl"}}},
    ]}}  # fmt: skip
    data["extra"] = {"multiturn": {"passed": 1, "total": 2}}
    s = bakeoff.summarize_promptfoo(data)
    assert s["categories"] == {"acl": 1.0, "escrita": 0.5, "rag": 0.5}
    assert round(s["pass_rate"], 3) == 0.6 and s["latency_ms"] == 2000
    table = bakeoff.build_table({"m": s}, {"gemma4:e4b": "não baixado"})
    assert "| m | 60% | 2.0 | 100% | 50% | 50% |" in table and "gemma4:e4b" in table
