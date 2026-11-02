from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from encaixe.core.types import ChunkDraft

if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import PdfPipelineOptions

log = structlog.get_logger(__name__)


class DoclingChunker:
    """Docling (conversão) + HybridChunker (por títulos e limite de tokens do embedder).

    `artifacts_path`: modelos do Docling já baixados (`docling-tools models download`), para
    converter PDF sem rede; se o diretório não existir, vale o padrão do Docling (baixa no 1º
    uso). `do_ocr`: OCR de PDFs escaneados (desligado: exige modelos de OCR).
    """

    def __init__(
        self,
        tokenizer_id: str = "BAAI/bge-m3",
        max_tokens: int = 512,
        *,
        artifacts_path: Path | str | None = None,
        do_ocr: bool = False,
    ) -> None:
        self.tokenizer_id = tokenizer_id
        self.max_tokens = max_tokens
        self.artifacts_path = artifacts_path
        self.do_ocr = do_ocr
        self._converter = None
        self._chunker = None

    def pdf_pipeline_options(self) -> PdfPipelineOptions:
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        options = PdfPipelineOptions(do_ocr=self.do_ocr)
        if self.artifacts_path is not None:
            path = Path(self.artifacts_path).expanduser()
            if path.is_dir():
                options.artifacts_path = path
            else:
                log.warning("docling_artifacts_missing", path=str(path))
        return options

    def _load(self) -> None:
        if self._chunker is not None:
            return
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_id),
            max_tokens=self.max_tokens,
        )
        pdf = PdfFormatOption(pipeline_options=self.pdf_pipeline_options())
        self._converter = DocumentConverter(format_options={InputFormat.PDF: pdf})
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
