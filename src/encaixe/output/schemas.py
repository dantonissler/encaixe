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


# A resposta final é entregue por uma ferramenta de saída (ToolOutput), não pela gramática do
# Ollama: com `response_format` json_schema o modelo nunca consegue chamar ferramentas.
OUTPUT_TOOL = "responder"
OUTPUT_TOOL_DESCRIPTION = (
    "Entrega a resposta final ao usuário. Toda resposta termina chamando esta ferramenta, "
    "inclusive para pedir confirmação, dizer que não encontrou a informação ou recusar."
)
# Devolvido ao modelo quando ele responde em texto puro num turno que pode precisar de citações
# (no lugar do "Please include your response in a tool call." do Pydantic AI).
TEXT_ANSWER_RETRY = (
    f"Entregue a resposta chamando a ferramenta {OUTPUT_TOOL}, com os ids dos trechos usados "
    "em `citacoes` (lista vazia se nenhum foi usado)."
)
