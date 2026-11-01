-- Schema inicial do encaixe. Executado como superusuário (postgres); o app conecta como encaixe_app.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TEXT SEARCH CONFIGURATION pt_unaccent (COPY = portuguese);
ALTER TEXT SEARCH CONFIGURATION pt_unaccent
    ALTER MAPPING FOR hword, hword_part, word WITH unaccent, portuguese_stem;

CREATE TABLE api_keys (
    id          bigserial PRIMARY KEY,
    app_id      text NOT NULL,
    key_hash    text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    revoked_at  timestamptz
);

-- Mantém o corpus_version por app; não tem RLS (consultada pelo runtime fora de app_scope).
CREATE TABLE app_runtime (
    app_id         text PRIMARY KEY,
    corpus_version integer NOT NULL DEFAULT 0
);

CREATE TABLE conversations (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    app_id      text NOT NULL,
    user_ref    text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX conversations_app_idx ON conversations (app_id);

CREATE TABLE messages (
    id              bigserial PRIMARY KEY,
    app_id          text NOT NULL,
    conversation_id uuid NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    seq             integer NOT NULL,
    role            text NOT NULL,
    content         text NOT NULL,
    tool_calls      jsonb,
    tokens          integer,
    trace_id        text,
    prompt_version  text,
    model           text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, seq)
);

CREATE TABLE conversation_summaries (
    conversation_id  uuid PRIMARY KEY REFERENCES conversations (id) ON DELETE CASCADE,
    app_id           text NOT NULL,
    summary          text NOT NULL,
    covers_until_seq integer NOT NULL
);

CREATE TABLE documents (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    app_id       text NOT NULL,
    title        text NOT NULL,
    source_path  text,
    content_hash text,
    acl          text[] NOT NULL DEFAULT ARRAY['app:*'],
    status       text NOT NULL DEFAULT 'pending',
    quarantined  boolean NOT NULL DEFAULT false,
    error        text,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX documents_app_idx ON documents (app_id);

CREATE TABLE chunks (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    app_id          text NOT NULL,
    document_id     uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    text            text NOT NULL,
    heading_path    text[] NOT NULL DEFAULT '{}',
    acl_principals  text[] NOT NULL DEFAULT ARRAY['app:*'],
    metadata        jsonb NOT NULL DEFAULT '{}',
    embedding       vector(1024),
    embedding_model text,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('pt_unaccent'::regconfig, text)) STORED
);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_acl_idx ON chunks USING gin (acl_principals);
CREATE INDEX chunks_app_idx ON chunks (app_id);
CREATE INDEX chunks_document_idx ON chunks (document_id);

CREATE TABLE tool_approvals (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    app_id          text NOT NULL,
    conversation_id uuid NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    user_ref        text,
    server          text NOT NULL,
    tool            text NOT NULL,
    args            jsonb NOT NULL DEFAULT '{}',
    status          text NOT NULL DEFAULT 'pending',
    expires_at      timestamptz NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- RLS: current_setting(..., true) devolve NULL se o setting não existe => nenhuma linha, nunca erro.
ALTER TABLE conversations         ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages              ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents             ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks                ENABLE ROW LEVEL SECURITY;
ALTER TABLE tool_approvals        ENABLE ROW LEVEL SECURITY;

CREATE POLICY app_isolation ON conversations
    USING (app_id = current_setting('encaixe.app_id', true))
    WITH CHECK (app_id = current_setting('encaixe.app_id', true));
CREATE POLICY app_isolation ON messages
    USING (app_id = current_setting('encaixe.app_id', true))
    WITH CHECK (app_id = current_setting('encaixe.app_id', true));
CREATE POLICY app_isolation ON conversation_summaries
    USING (app_id = current_setting('encaixe.app_id', true))
    WITH CHECK (app_id = current_setting('encaixe.app_id', true));
CREATE POLICY app_isolation ON documents
    USING (app_id = current_setting('encaixe.app_id', true))
    WITH CHECK (app_id = current_setting('encaixe.app_id', true));
CREATE POLICY app_isolation ON tool_approvals
    USING (app_id = current_setting('encaixe.app_id', true))
    WITH CHECK (app_id = current_setting('encaixe.app_id', true));
CREATE POLICY app_isolation ON chunks
    USING (app_id = current_setting('encaixe.app_id', true))
    WITH CHECK (app_id = current_setting('encaixe.app_id', true));
-- Política restritiva: leitura/alteração de chunks exige interseção de ACL com os principais.
-- A inserção (ingestão) só exige o app, para o worker poder gravar chunks restritos.
CREATE POLICY acl_visibility ON chunks AS RESTRICTIVE FOR SELECT
    USING (acl_principals && string_to_array(current_setting('encaixe.principals', true), ','));
CREATE POLICY acl_visibility_upd ON chunks AS RESTRICTIVE FOR UPDATE
    USING (acl_principals && string_to_array(current_setting('encaixe.principals', true), ','));
CREATE POLICY acl_visibility_del ON chunks AS RESTRICTIVE FOR DELETE
    USING (acl_principals && string_to_array(current_setting('encaixe.principals', true), ','));

-- Privilégios do role da aplicação (tabelas pertencem ao postgres).
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO encaixe_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO encaixe_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO encaixe_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO encaixe_app;
REVOKE ALL ON schema_migrations FROM encaixe_app;
GRANT SELECT ON schema_migrations TO encaixe_app;
