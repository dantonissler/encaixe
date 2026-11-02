from __future__ import annotations

import math


class HeuristicTokenCounter:
    """Contagem heurística: ceil(len/3)."""

    def count(self, text: str) -> int:
        return math.ceil(len(text) / 3)
