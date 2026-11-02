import json

import compare_baseline as cb


def test_detects_drop_over_threshold():
    baseline = {"pass_rate": 0.9, "recall@5": 0.9, "mrr": 0.8}
    current = {"pass_rate": 0.80, "recall@5": 0.88, "mrr": 0.8}
    problems = cb.compare(current, baseline)
    assert len(problems) == 1 and "pass_rate" in problems[0]
    assert cb.compare({"pass_rate": 0.86, "recall@5": 0.9, "mrr": 0.8}, baseline) == []


def test_no_baseline_passes_and_update_writes(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "app.json").write_text(
        json.dumps(
            {
                "results": {"stats": {"successes": 9, "failures": 1}},
                "extra": {"multiturn": {"passed": 1, "total": 2}},
            }
        )
    )
    (results / "app-retrieval.json").write_text(
        json.dumps({"recall@5": 0.8, "mrr": 0.6, "acl_leaks": 0})
    )
    baselines = tmp_path / "baselines"

    assert (
        cb.main(["--app", "app"], results_dir=results, baselines_dir=baselines, min_cases=10) == 0
    )
    assert not (baselines / "app.json").exists()

    assert (
        cb.main(
            ["--app", "app", "--update"], results_dir=results, baselines_dir=baselines, min_cases=10
        )
        == 0
    )
    saved = json.loads((baselines / "app.json").read_text())
    assert saved == {"pass_rate": 10 / 12, "recall@5": 0.8, "mrr": 0.6, "acl_leaks": 0}

    (results / "app-retrieval.json").write_text(
        json.dumps({"recall@5": 0.5, "mrr": 0.6, "acl_leaks": 0})
    )
    assert (
        cb.main(["--app", "app"], results_dir=results, baselines_dir=baselines, min_cases=10) == 1
    )


def _write(results, promptfoo, retrieval):
    results.mkdir(exist_ok=True)
    (results / "app.json").write_text(json.dumps(promptfoo))
    (results / "app-retrieval.json").write_text(json.dumps(retrieval))


def test_missing_promptfoo_results_fail_even_if_multiturn_passes(tmp_path):
    results = tmp_path / "r"
    only_multiturn = {"extra": {"multiturn": {"passed": 2, "total": 2}}}
    _write(results, only_multiturn, {"recall@5": 1, "mrr": 1, "acl_leaks": 0})
    args = {"results_dir": results, "baselines_dir": tmp_path / "b", "min_cases": 28}
    assert cb.main(["--app", "app"], **args) == 1
    partial = {"results": {"stats": {"successes": 5, "failures": 0}}, **only_multiturn}
    _write(results, partial, {"recall@5": 1, "mrr": 1, "acl_leaks": 0})
    assert cb.main(["--app", "app"], **args) == 1  # menos que os 28 casos
    full = {"results": {"stats": {"successes": 26, "failures": 1, "errors": 1}}, **only_multiturn}
    _write(results, full, {"recall@5": 1, "mrr": 1, "acl_leaks": 0})
    assert cb.main(["--app", "app"], **args) == 0


def test_acl_leak_fails_regardless_of_baseline(tmp_path):
    assert cb.compare({"pass_rate": 1, "recall@5": 1, "mrr": 1, "acl_leaks": 1}, {})
    results = tmp_path / "r"
    ok = {"results": {"stats": {"successes": 10, "failures": 0}}}
    _write(results, ok, {"recall@5": 1, "mrr": 1, "acl_leaks": 2})
    args = {"results_dir": results, "baselines_dir": tmp_path / "b", "min_cases": 10}
    assert cb.main(["--app", "app"], **args) == 1  # sem baseline
    assert cb.main(["--app", "app", "--update"], **args) == 1  # nem grava
    assert not (tmp_path / "b" / "app.json").exists()
