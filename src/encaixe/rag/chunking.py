from __future__ import annotations

from pathlib import Path

from encaixe.core.types import ChunkDraft


class DoclingChunker:
    """Docling (conversão) + HybridChunker (por títulos e limite de tokens do embedder)."""

    def __init__(self, tokenizer_id: str = "BAAI/bge-m3", max_tokens: int = 512) -> None:
        self.tokenizer_id = tokenizer_id
        self.max_tokens = max_tokens
        self._converter = None
        self._chunker = None

    def _load(self) -> None:
        if self._chunker is not None:
            return
        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_id),
            max_tokens=self.max_tokens,
        )
        self._converter = DocumentConverter()
        self._chunker = HybridChunker(tokenizer=tokenizer)

    def chunk(self, path: Path, title: str) -> list[ChunkDraft]:
        self._load()
        path = Path(path)
        if path.suffix.lower() == ".txt":
            # Docling não lê .txt: o conteúdo é tratado como markdown (via stream em memória).
            from io import BytesIO

            from docling.datamodel.base_models import DocumentStream

            source = DocumentStream(name=path.stem + ".md", stream=BytesIO(path.read_bytes()))
        else:
            source = path
        doc = self._converter.convert(source).document
        drafts: list[ChunkDraft] = []
        for ch in self._chunker.chunk(dl_doc=doc):
            body = ch.text.strip()
            if not body:
                continue
            headings = tuple(ch.meta.headings or ())
            prefix = " > ".join([title, *headings])
            page = None
            for item in ch.meta.doc_items or []:
                if item.prov:
                    page = item.prov[0].page_no
                    break
            drafts.append(ChunkDraft(f"{prefix}\n\n{body}", headings, page))
        return drafts
