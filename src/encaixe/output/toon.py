from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from toon_format import decode, encode


def to_toon(data: dict | list | BaseModel) -> str:
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    return encode(data)


def from_toon(text: str) -> Any:
    return decode(text)
