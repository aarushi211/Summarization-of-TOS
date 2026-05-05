-- ── MIGRATION 003: Row Level Security (RLS) ─────────────────────────────────
-- This script secures your database so that users can only see and modify 
-- their own documents and chat messages.

-- 1. Enable RLS on all tables
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- 2. Documents Policies
CREATE POLICY "Users can insert their own documents" ON documents 
    FOR INSERT TO authenticated 
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own documents" ON documents 
    FOR SELECT TO authenticated 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own documents" ON documents 
    FOR UPDATE TO authenticated 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own documents" ON documents 
    FOR DELETE TO authenticated 
    USING (auth.uid() = user_id);

-- 3. Chat Sessions Policies
CREATE POLICY "Users can insert their own chat sessions" ON chat_sessions 
    FOR INSERT TO authenticated 
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own chat sessions" ON chat_sessions 
    FOR SELECT TO authenticated 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own chat sessions" ON chat_sessions 
    FOR UPDATE TO authenticated 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own chat sessions" ON chat_sessions 
    FOR DELETE TO authenticated 
    USING (auth.uid() = user_id);

-- 4. Chat Messages Policies (Derived from session_id)
CREATE POLICY "Users can insert messages in their own sessions" ON chat_messages 
    FOR INSERT TO authenticated 
    WITH CHECK (session_id IN (SELECT id FROM chat_sessions WHERE user_id = auth.uid()));

CREATE POLICY "Users can view messages in their own sessions" ON chat_messages 
    FOR SELECT TO authenticated 
    USING (session_id IN (SELECT id FROM chat_sessions WHERE user_id = auth.uid()));

CREATE POLICY "Users can update messages in their own sessions" ON chat_messages 
    FOR UPDATE TO authenticated 
    USING (session_id IN (SELECT id FROM chat_sessions WHERE user_id = auth.uid()));

CREATE POLICY "Users can delete messages in their own sessions" ON chat_messages 
    FOR DELETE TO authenticated 
    USING (session_id IN (SELECT id FROM chat_sessions WHERE user_id = auth.uid()));
