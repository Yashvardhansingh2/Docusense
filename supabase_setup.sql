-- ============================================================
-- DocuSense AI — Supabase SQL Setup
-- Run this entire file in: Supabase → SQL Editor → New Query
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id             UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename       TEXT        NOT NULL,
    storage_path   TEXT        NOT NULL DEFAULT '',
    file_size_bytes INTEGER,
    chunk_count    INTEGER     DEFAULT 0,
    status         TEXT        DEFAULT 'processing',
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table with vector embeddings (768 dims = Gemini gemini-embedding-001 with output_dimensionality=768)
CREATE TABLE IF NOT EXISTS chunks (
    id          UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID    REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT    NOT NULL,
    embedding   vector(768),
    chunk_index INTEGER NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

-- Query logs table
CREATE TABLE IF NOT EXISTS query_logs (
    id              UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID    REFERENCES documents(id) ON DELETE SET NULL,
    question        TEXT    NOT NULL,
    answer          TEXT,
    sources_used    INTEGER,
    response_time_ms INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
