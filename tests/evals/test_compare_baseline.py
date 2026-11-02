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
    (results / "app-retrieval.json").write_text(json.dumps({"recall@5": 0.8, "mrr": 0.6}))
    baselines = tmp_path / "baselines"

    assert cb.main(["--app", "app"], results_dir=results, baselines_dir=baselines) == 0
    assert not (baselines / "app.json").exists()

    assert cb.main(["--app", "app", "--update"], results_dir=results, baselines_dir=baselines) == 0
    saved = json.loads((baselines / "app.json").read_text())
    assert saved == {"pass_rate": 10 / 12, "recall@5": 0.8, "mrr": 0.6}

    (results / "app-retrieval.json").write_text(json.dumps({"recall@5": 0.5, "mrr": 0.6}))
    assert cb.main(["--app", "app"], results_dir=results, baselines_dir=baselines) == 1
