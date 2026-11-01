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

    def __call__(self, windows):
        self.calls.append(windows)
        return [
            {"label": "INJECTION" if "bad" in w else "SAFE", "score": 0.99} for w in windows
        ]


async def test_windowing_blocks_if_any_window_malicious():
    pipe = FakePipe()
    g = InjectionGuard("m", 0.9, pipeline_factory=lambda _m: pipe)
    r = await g.check_text(" ".join(["w"] * 3000))
    assert r.verdict == "block"
    assert len(pipe.calls[0]) > 1


async def test_safe_and_threshold():
    pipe = FakePipe()
    pipe.__class__ = type("P", (FakePipe,), {"__call__": lambda s, w: [{"label": "LABEL_1", "score": 0.5} for _ in w]})
    g = InjectionGuard("m", 0.9, pipeline_factory=lambda _m: pipe)
    assert (await g.check_text("oi")).verdict == "pass"
