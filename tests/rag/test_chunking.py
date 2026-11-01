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
