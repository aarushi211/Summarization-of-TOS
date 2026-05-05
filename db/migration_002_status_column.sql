-- Add status column to documents table (run in Supabase SQL Editor)
ALTER TABLE public.documents
    ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'ready';
