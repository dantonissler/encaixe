-- Dedup de documentos por conteúdo dentro da app.
CREATE UNIQUE INDEX IF NOT EXISTS documents_app_hash_uniq
    ON documents (app_id, content_hash) WHERE content_hash IS NOT NULL;
