from pathlib import Path

import pytest

from encaixe.rag.chunking import DoclingChunker


@pytest.fixture(scope="module")
def chunker() -> DoclingChunker:
    return DoclingChunker()


def test_markdown_keeps_heading_path(chunker, tmp_path: Path):
    f = tmp_path / "politica.md"
    f.write_text(
        "# Férias\n\nTexto geral sobre férias.\n\n## Prazo\n\nO prazo é de 30 dias.\n",
        encoding="utf-8",
    )
    drafts = chunker.chunk(f, "Política")
    assert any(d.heading_path == ("Férias", "Prazo") for d in drafts)
    prazo = next(d for d in drafts if d.heading_path == ("Férias", "Prazo"))
    assert prazo.text.startswith("Política > Férias > Prazo")
    assert "30 dias" in prazo.text


def test_txt_supported(chunker, tmp_path: Path):
    f = tmp_path / "nota.txt"
    f.write_text("Reembolso de despesas.\n\nO reembolso sai em 5 dias.\n", encoding="utf-8")
    drafts = chunker.chunk(f, "Nota")
    assert drafts
    assert any("5 dias" in d.text for d in drafts)
    assert all(d.text.startswith("Nota") for d in drafts)


def test_pdf_options_use_artifacts_path_and_disable_ocr(tmp_path: Path):
    opts = DoclingChunker(artifacts_path=tmp_path).pdf_pipeline_options()
    assert Path(opts.artifacts_path) == tmp_path
    assert opts.do_ocr is False


def test_missing_artifacts_dir_falls_back_to_docling_default(tmp_path: Path):
    opts = DoclingChunker(artifacts_path=tmp_path / "nao-existe").pdf_pipeline_options()
    assert opts.artifacts_path is None


def test_ocr_can_be_enabled():
    assert DoclingChunker(do_ocr=True).pdf_pipeline_options().do_ocr is True


def test_converter_uses_pdf_options(tmp_path: Path):
    from docling.datamodel.base_models import InputFormat

    chunker = DoclingChunker(artifacts_path=tmp_path)
    chunker._load()
    opts = chunker._converter.format_to_options[InputFormat.PDF].pipeline_options
    assert Path(opts.artifacts_path) == tmp_path and opts.do_ocr is False


def minimal_pdf(lines: list[str]) -> bytes:
    """PDF de uma página com texto (Helvetica), escrito à mão: não há gerador de PDF instalado."""
    body = " ".join(f"({line}) Tj T*" for line in lines)
    content = f"BT /F1 14 Tf 72 720 Td 20 TL {body} ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return out


_CONVERT_OFFLINE = """
import json, sys
from pathlib import Path
from encaixe.rag.chunking import DoclingChunker
drafts = DoclingChunker(artifacts_path=sys.argv[1]).chunk(Path(sys.argv[2]), "Reembolso")
print(json.dumps([d.text for d in drafts]))
"""


@pytest.mark.integration
def test_text_pdf_converts_offline(tmp_path: Path):
    """PDF de texto convertido sem rede: modelos do Docling vêm de `docling_artifacts_path`
    (baixados com `docling-tools models download`) e o tokenizer do cache do HF."""
    import json
    import os
    import subprocess
    import sys

    from encaixe.settings import Settings

    artifacts = Settings().docling_artifacts_path
    if artifacts is None or not artifacts.is_dir():
        pytest.skip("modelos do Docling não baixados (docling-tools models download)")
    pdf = tmp_path / "reembolso.pdf"
    pdf.write_bytes(minimal_pdf(["Politica de reembolso", "O reembolso sai em 5 dias uteis."]))
    dead = "http://127.0.0.1:9"  # qualquer tentativa de rede falha na hora
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "HTTP_PROXY": dead,
        "HTTPS_PROXY": dead,
        "http_proxy": dead,
        "https_proxy": dead,
    }
    env.pop("DOCLING_ARTIFACTS_PATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CONVERT_OFFLINE, str(artifacts), str(pdf)],
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    texts = json.loads(proc.stdout.strip().splitlines()[-1])
    assert any("5 dias" in t for t in texts), texts
