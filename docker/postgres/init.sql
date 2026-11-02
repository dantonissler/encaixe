-- Executado uma vez na criação do volume (banco "encaixe" já criado por POSTGRES_DB).
-- Senha apenas de desenvolvimento; a mesma consta em .env.example.
CREATE ROLE encaixe_app LOGIN PASSWORD 'encaixe_app_dev'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;

CREATE DATABASE phoenix;
CREATE DATABASE encaixe_test;

GRANT ALL PRIVILEGES ON DATABASE encaixe TO encaixe_app;
GRANT ALL PRIVILEGES ON DATABASE encaixe_test TO encaixe_app;

\connect encaixe
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
GRANT ALL ON SCHEMA public TO encaixe_app;

\connect encaixe_test
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
GRANT ALL ON SCHEMA public TO encaixe_app;
