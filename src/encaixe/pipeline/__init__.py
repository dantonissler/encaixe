"""Pipeline de chat: eventos, vaga de geração e orquestração por requisição."""

from encaixe.pipeline.chat import ChatPipeline
from encaixe.pipeline.events import ChatEvent, Final, Retraction, Status, TextDelta
from encaixe.pipeline.gate import GenerationGate, QueueFull, QueueTimeout

__all__ = [
    "ChatEvent",
    "ChatPipeline",
    "Final",
    "GenerationGate",
    "QueueFull",
    "QueueTimeout",
    "Retraction",
    "Status",
    "TextDelta",
]
