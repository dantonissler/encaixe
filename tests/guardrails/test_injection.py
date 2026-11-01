from encaixe.guardrails.injection import InjectionGuard


class Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids):
        return " ".join("bad" if i == 1500 else "w" for i in ids)


class FakePipe:
    tokenizer = Tok()

    def __init__(self):
        self.calls = []

    def __call__(self, windows, top_k=None):
        self.calls.append(windows)
        return [
            [{"label": "INJECTION" if "bad" in w else "SAFE", "score": 0.99},
             {"label": "SAFE" if "bad" in w else "INJECTION", "score": 0.01}]
            for w in windows
        ]


async def test_windowing_blocks_if_any_window_malicious():
    pipe = FakePipe()
    g = InjectionGuard("m", 0.9, pipeline_factory=lambda _m: pipe, backend="model")
    r = await g.check_text(" ".join(["w"] * 3000))
    assert r.verdict == "block"
    assert len(pipe.calls[0]) > 1


class FixedPipe:
    tokenizer = None

    def __init__(self, scores):
        self.scores = scores

    def __call__(self, windows, top_k=None):
        return [list(self.scores) for _ in windows]


def _guard(scores):
    return InjectionGuard("m", 0.9, pipeline_factory=lambda _m: FixedPipe(scores), backend="model")


async def test_safe_label_high_score_passes():
    g = _guard([{"label": "SAFE", "score": 0.99}, {"label": "INJECTION", "score": 0.01}])
    assert (await g.check_text("oi")).verdict == "pass"


async def test_malicious_below_threshold_passes_and_above_blocks():
    assert (await _guard([{"label": "LABEL_1", "score": 0.5}]).check_text("oi")).verdict == "pass"
    assert (await _guard([{"label": "LABEL_1", "score": 0.95}]).check_text("oi")).verdict == "block"


async def test_unknown_label_raises():
    import pytest

    with pytest.raises(ValueError):
        await _guard([{"label": "WEIRD", "score": 0.99}]).check_text("oi")


async def test_last_window_is_covered():
    pipe = FakePipe()
    g = InjectionGuard("m", 0.9, pipeline_factory=lambda _m: pipe, backend="model")
    await g.check_text(" ".join(["w"] * 1200))
    last_ids = pipe.tokenizer(" ".join(["w"] * 1200))["input_ids"][-1]
    assert last_ids == 1199 and len(pipe.calls[0]) >= 3
