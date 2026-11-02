REFUSAL_INPUT = "Não posso ajudar com esse pedido."
NOT_FOUND = "Não encontrei essa informação nos documentos disponíveis."
ALL_FAILED = "Desculpe, não consegui responder agora. Tente novamente em instantes."
LIMIT_REACHED = "Não consegui concluir a tarefa dentro do limite de passos."
QUEUE_WAIT = "Aguardando na fila…"
RETRACTION = "A resposta anterior foi removida por violar as políticas de segurança."
APPROVAL_PROMPT = (
    'Para continuar, preciso da sua confirmação: {acao}. '
    'Responda "sim" para confirmar ou "não" para cancelar.'
)
# Resultado que o MODELO recebe quando uma ferramenta de escrita fica aguardando confirmação.
# Sem isso o modelo respondia em texto (fora da ferramenta de saída) ou repetia a chamada.
APPROVAL_PENDING_RESULT = (
    "AGUARDANDO CONFIRMAÇÃO DO USUÁRIO: a ação {acao} ainda NÃO foi executada; ela só roda "
    'quando o usuário responder "sim" na próxima mensagem. Não chame {ferramenta} de novo '
    "neste turno e não diga que a ação foi feita. Termine agora chamando a ferramenta {saida} "
    "com `citacoes` vazia e `resposta` igual a este texto:\n" + APPROVAL_PROMPT
)
APPROVAL_CANCELLED = "Ação cancelada."
GUARD_UNAVAILABLE = (
    "O serviço está temporariamente indisponível por segurança. Tente novamente em instantes."
)
