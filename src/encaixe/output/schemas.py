from pydantic import BaseModel, Field


class RespostaBase(BaseModel):
    resposta: str = Field(description="Resposta ao usuário, em português do Brasil.")
    citacoes: list[str] = Field(
        default_factory=list,
        description=(
            "Ids dos trechos recuperados usados na resposta. "
            "Use apenas ids que apareceram nos trechos fornecidos; lista vazia se nenhum foi usado."
        ),
    )
