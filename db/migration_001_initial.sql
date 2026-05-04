-- ============================================================
-- TOS Summarizer: Supabase Database Migration
-- Run this entire script in: Supabase Dashboard -> SQL Editor -> New Query
-- ============================================================

-- 1. Documents table: tracks every uploaded file per user
CREATE TABLE IF NOT EXISTS public.documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    filename      TEXT NOT NULL,
    s3_key        TEXT NOT NULL,          -- e.g. "uploads/{user_id}/{doc_id}.pdf"
    service_name  TEXT DEFAULT 'Unknown Service',
    doc_type      TEXT DEFAULT 'Terms of Service',
    pinecone_ns   TEXT NOT NULL,          -- "{user_id}_{doc_id}" namespace in Pinecone
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Chat sessions: groups a conversation for a specific document
CREATE TABLE IF NOT EXISTS public.chat_sessions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    document_id  UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    title        TEXT DEFAULT 'New Chat',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Chat messages: individual Q&A turns within a session
CREATE TABLE IF NOT EXISTS public.chat_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    cited_sources   JSONB DEFAULT '[]',   -- stores the list of citation objects
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Row-Level Security: users can only see their own data
ALTER TABLE public.documents     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

-- Documents: owner-only access
CREATE POLICY "Users can manage own documents"
    ON public.documents FOR ALL
    USING (auth.uid() = user_id);

-- Chat sessions: owner-only access
CREATE POLICY "Users can manage own chat sessions"
    ON public.chat_sessions FOR ALL
    USING (auth.uid() = user_id);

-- Chat messages: access via session ownership
CREATE POLICY "Users can manage own chat messages"
    ON public.chat_messages FOR ALL
    USING (
        session_id IN (
            SELECT id FROM public.chat_sessions WHERE user_id = auth.uid()
        )
    );

-- 5. Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_documents_user     ON public.documents(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_document  ON public.chat_sessions(document_id);
CREATE INDEX IF NOT EXISTS idx_messages_session   ON public.chat_messages(session_id);
