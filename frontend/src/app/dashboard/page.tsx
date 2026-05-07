"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  LogOut, FileText, Upload, Search,
  Send, Loader2, Info, Menu, Trash2,
} from "lucide-react";
import { getValidToken, clearTokens } from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────
interface Document {
  id: string;
  filename: string;
  service_name: string;
  doc_type: string;
  status: string;
  created_at: string;
}

interface Source {
  tag: string;
  citation: string;
  section: string;
  subsection: string;
  page: number;
  excerpt: string;
}

interface TopicResult {
  label: string;
  summary: string;
  sources?: Source[];
}

interface SummaryResponse {
  topics: TopicResult[];
}

interface Message {
  role: "user" | "assistant";
  content: string;
  cited_sources?: Source[];
  error?: string;
  isStreaming?: boolean;
}

// ── API base URL — trailing slash stripped to prevent double-slash URLs ───────
const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

// ── Authenticated fetch helper ────────────────────────────────────────────────
async function authFetch(
  path: string,
  options: RequestInit = {},
): Promise<Response> {
  const token = await getValidToken();
  if (!token) throw new Error("NOT_AUTHENTICATED");

  return fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options.headers ?? {}),
      Authorization: `Bearer ${token}`,
    },
  });
}

// ── SSE parser ────────────────────────────────────────────────────────────────
async function consumeSse(
  response: Response,
  onEvent: (event: Record<string, unknown>) => void,
): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data: ")) continue;
        try {
          onEvent(JSON.parse(line.slice(6)));
        } catch {
          // Malformed SSE line — skip silently
        }
      }
    }
  } finally {
    reader.cancel();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const router = useRouter();

  // ── Auth ──────────────────────────────────────────────────────────────────
  const [userEmail, setUserEmail] = useState("");
  const [isSidebarOpen, setSidebarOpen] = useState(true);

  // ── Data ──────────────────────────────────────────────────────────────────
  const [historyDocs, setHistoryDocs] = useState<Document[]>([]);
  const [activeDoc, setActiveDoc] = useState<Document | null>(null);

  // ── Upload ────────────────────────────────────────────────────────────────
  const [file, setFile] = useState<File | null>(null);
  const [serviceName, setServiceName] = useState("");
  const [docType, setDocType] = useState("Terms of Service");
  const [isUploading, setIsUploading] = useState(false);

  // ── Document status polling ───────────────────────────────────────────────
  const [docStatus, setDocStatus] = useState<string>("ready");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Model readiness ───────────────────────────────────────────────────────
  const [modelReady, setModelReady] = useState(false);
  const [modelChecking, setModelChecking] = useState(true);
  const modelPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Summary ───────────────────────────────────────────────────────────────
  const [summaryTopics, setSummaryTopics] = useState<TopicResult[]>([]);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  // ── Chat ──────────────────────────────────────────────────────────────────
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatting, setIsChatting] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // ── UI State ──────────────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<"summary" | "chat">("summary");
  const [isDeleting, setIsDeleting] = useState(false);

  // ── Scroll chat to bottom on new messages ─────────────────────────────────
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Model readiness poller ────────────────────────────────────────────────
  // Polls /health every 5s until model_ready=true, then stops.
  // Runs independently of auth so the banner shows immediately on load.
  useEffect(() => {
    const checkModel = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
          const data = await res.json();
          if (data.model_ready) {
            setModelReady(true);
            setModelChecking(false);
            if (modelPollRef.current) {
              clearInterval(modelPollRef.current);
              modelPollRef.current = null;
            }
          }
        }
      } catch {
        // Network error — will retry on next interval
      }
    };

    checkModel();
    modelPollRef.current = setInterval(checkModel, 5000);

    return () => {
      if (modelPollRef.current) clearInterval(modelPollRef.current);
    };
  }, []);

  // ── Auth check on mount ───────────────────────────────────────────────────
  useEffect(() => {
    const email = localStorage.getItem("user_email");
    getValidToken().then(token => {
      if (!token) {
        router.push("/");
        return;
      }
      setUserEmail(email && email !== "undefined" ? email : "User");
      fetchHistory();
    });
  }, [router]);

  // ── Fetch document history ────────────────────────────────────────────────
  const fetchHistory = useCallback(async () => {
    try {
      const res = await authFetch("/history/documents");
      if (res.status === 401) { handleLogout(); return; }
      if (res.ok) {
        const data = await res.json();
        setHistoryDocs(data.documents ?? []);
      }
    } catch (err) {
      if ((err as Error).message === "NOT_AUTHENTICATED") handleLogout();
    }
  }, []);

  // ── Logout ────────────────────────────────────────────────────────────────
  const handleLogout = () => {
    clearTokens();
    localStorage.removeItem("user_email");
    router.push("/");
  };

  // ── Upload PDF ────────────────────────────────────────────────────────────
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("service_name", serviceName || "Unknown");
      formData.append("doc_type", docType);

      const res = await authFetch("/ingest/upload", { method: "POST", body: formData });

      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail ?? "Upload failed");
      }

      const data = await res.json();
      const newDoc: Document = {
        id: data.document_id,
        filename: file.name,
        service_name: serviceName || "Unknown",
        doc_type: docType,
        status: "processing",
        created_at: new Date().toISOString(),
      };

      setActiveDoc(newDoc);
      setMessages([]);
      setSummaryTopics([]);
      setSummaryError(null);
      setChatError(null);
      setDocStatus("processing");
      fetchHistory();

      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const r = await authFetch(`/history/documents/${data.document_id}/status`);
          if (r.ok) {
            const s = await r.json();
            setDocStatus(s.status);
            if (s.status === "ready" || s.status === "error") {
              clearInterval(pollRef.current!);
              pollRef.current = null;
              fetchHistory();
              if (s.status === "error" && s.error_reason) {
                setSummaryError(`Ingestion failed: ${s.error_reason}`);
              }
            }
          }
        } catch { /* ignore — will retry next interval */ }
      }, 3000);

    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  // ── Load past document ────────────────────────────────────────────────────
  const loadPastDocument = async (doc: Document) => {
    setActiveDoc(doc);
    setSummaryTopics([]);
    setSummaryError(null);
    setChatError(null);
    setMessages([]);
    setDocStatus(doc.status ?? "ready");

    try {
      const [chatRes, summaryRes] = await Promise.all([
        authFetch(`/history/chats/${doc.id}`),
        authFetch(`/history/summaries/${doc.id}`)
      ]);

      if (chatRes.ok) {
        const data = await chatRes.json();
        const loaded: Message[] = [];
        (data.sessions ?? []).forEach((s: any) => {
          s.messages.forEach((m: any) => {
            loaded.push({ role: m.role, content: m.content, cited_sources: m.sources });
          });
        });
        setMessages(loaded);
      }

      if (summaryRes.ok) {
        const data = await summaryRes.json();
        setSummaryTopics(data);
      }
    } catch (err) {
      console.error("Failed to load document data:", err);
    }
  };

  // ── Generate summary ──────────────────────────────────────────────────────
  const handleGenerateSummary = async () => {
    if (!activeDoc || !modelReady) return;
    setIsSummarizing(true);
    setSummaryTopics([]);
    setSummaryError(null);

    try {
      const res = await authFetch(`/chat/summary/${activeDoc.id}`);
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail ?? "Failed to start summary stream.");
      }

      await consumeSse(res, event => {
        if (event.type === "topic_ready") {
          setSummaryTopics(prev => [...prev, event.data as TopicResult]);
        } else if (event.type === "error") {
          setSummaryError(event.data as string);
        }
      });

    } catch (err: unknown) {
      setSummaryError(err instanceof Error ? err.message : "Failed to generate summary.");
    } finally {
      setIsSummarizing(false);
    }
  };

  // ── Send chat message ─────────────────────────────────────────────────────
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = chatInput.trim();
    if (!query || !activeDoc || !modelReady) return;

    setChatInput("");
    setChatError(null);
    setIsChatting(true);

    setMessages(prev => [
      ...prev,
      { role: "user", content: query },
      { role: "assistant", content: "", isStreaming: true },
    ]);

    try {
      const res = await authFetch("/chat/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          document_id: activeDoc.id,
          service_name: activeDoc.service_name,
        }),
      });

      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail ?? `HTTP ${res.status}`);
      }

      await consumeSse(res, event => {
        if (event.type === "token") {
          setMessages(prev => {
            const next = [...prev];
            const last = { ...next[next.length - 1] };
            last.content += event.data as string;
            next[next.length - 1] = last;
            return next;
          });
        } else if (event.type === "sources") {
          setMessages(prev => {
            const next = [...prev];
            const last = { ...next[next.length - 1] };
            last.cited_sources = event.data as Source[];
            next[next.length - 1] = last;
            return next;
          });
        } else if (event.type === "done") {
          setMessages(prev => {
            const next = [...prev];
            const last = { ...next[next.length - 1] };
            last.isStreaming = false;
            next[next.length - 1] = last;
            return next;
          });
        } else if (event.type === "error") {
          const errMsg = event.data as string;
          setChatError(errMsg);
          setMessages(prev => {
            const next = [...prev];
            const last = { ...next[next.length - 1] };
            last.isStreaming = false;
            last.error = errMsg;
            next[next.length - 1] = last;
            return next;
          });
        }
      });

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Connection failed.";
      setChatError(msg);
      setMessages(prev => {
        const next = [...prev];
        const last = { ...next[next.length - 1] };
        last.isStreaming = false;
        last.error = msg;
        next[next.length - 1] = last;
        return next;
      });
    } finally {
      setIsChatting(false);
    }
  };

  // ── Delete document ───────────────────────────────────────────────────────
  const handleDeleteDocument = async () => {
    if (!activeDoc) return;
    if (!confirm(
      `Delete "${activeDoc.filename}"?\n\nThis removes all chat history and vectors. Cannot be undone.`
    )) return;

    setIsDeleting(true);
    try {
      const res = await authFetch(`/history/documents/${activeDoc.id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Delete failed.");

      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      setActiveDoc(null);
      setMessages([]);
      setSummaryTopics([]);
      setSummaryError(null);
      setChatError(null);
      fetchHistory();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete document.");
    } finally {
      setIsDeleting(false);
    }
  };

  // ── Inline citation renderer ──────────────────────────────────────────────
  const renderCitedText = (text: string, sources?: Source[]) => {
    if (!sources?.length || !text) return <span>{text}</span>;

    const tagMap: Record<string, number> = {};
    sources.forEach((s, i) => {
      tagMap[s.tag.toLowerCase().trim()] = i + 1;
    });

    const parts = text.split(/(\[source \d+\]|\[SOURCE \d+\])/gi);

    return (
      <>
        {parts.map((part, i) => {
          const key = part.toLowerCase().trim();
          const num = tagMap[key];
          if (!num) return <span key={i}>{part}</span>;
          const src = sources[num - 1];
          return (
            <span
              key={i}
              title={src.citation}
              style={{
                color: "var(--accent-primary)",
                fontWeight: 700,
                fontSize: "0.8em",
                verticalAlign: "super",
                cursor: "help",
                marginLeft: "2px",
                padding: "0 2px",
                borderBottom: "1px solid var(--accent-primary)",
              }}
            >
              [{num}]
            </span>
          );
        })}
      </>
    );
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="layout-container">

      {/* ── Sidebar ── */}
      <aside
        className="sidebar"
        style={{
          transform: isSidebarOpen ? "translateX(0)" : "translateX(-100%)",
          transition: "transform 0.3s ease",
          position: "absolute",
          height: "100%",
          zIndex: 50,
          left: 0,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem" }}>
          <h2 className="heading-gradient" style={{ fontSize: "1.25rem", fontWeight: 700 }}>TOS Summarizer</h2>
          <button onClick={() => setSidebarOpen(false)} style={{ color: "var(--text-muted)" }}>
            <Menu size={20} />
          </button>
        </div>

        <div style={{ flex: 1, overflowY: "auto" }}>
          <h3 style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em", marginBottom: "1rem" }}>
            My Documents
          </h3>

          <button
            className="btn btn-secondary"
            style={{ width: "100%", justifyContent: "flex-start", marginBottom: "1rem" }}
            onClick={() => { setActiveDoc(null); setSummaryTopics([]); setMessages([]); setSidebarOpen(false); }}
          >
            + New Document
          </button>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {historyDocs.map(doc => (
              <div
                key={doc.id}
                className="group"
                style={{ display: "flex", alignItems: "center", gap: "0.25rem", width: "100%" }}
              >
                <button
                  onClick={() => { loadPastDocument(doc); setSidebarOpen(false); }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.75rem",
                    padding: "0.75rem",
                    borderRadius: "0.5rem",
                    background: activeDoc?.id === doc.id ? "rgba(59, 130, 246, 0.1)" : "transparent",
                    border: activeDoc?.id === doc.id ? "1px solid rgba(59, 130, 246, 0.3)" : "1px solid transparent",
                    textAlign: "left",
                    color: activeDoc?.id === doc.id ? "white" : "var(--text-secondary)",
                    transition: "all 0.2s",
                    flex: 1,
                    minWidth: 0,
                  }}
                >
                  <FileText size={16} style={{ color: activeDoc?.id === doc.id ? "var(--accent-primary)" : "var(--text-muted)", flexShrink: 0 }} />
                  <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    <div style={{ fontSize: "0.875rem", fontWeight: 500 }}>{doc.filename}</div>
                    <div style={{ fontSize: "0.75rem", opacity: 0.7 }}>{doc.service_name}</div>
                  </div>
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); setActiveDoc(doc); handleDeleteDocument(); }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ padding: "0.5rem", color: "var(--text-muted)", background: "transparent", border: "none" }}
                  onMouseEnter={e => (e.currentTarget.style.color = "#ef4444")}
                  onMouseLeave={e => (e.currentTarget.style.color = "var(--text-muted)")}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div style={{ borderTop: "var(--glass-border)", paddingTop: "1.5rem", marginTop: "auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
            <div style={{
              width: "32px", height: "32px", borderRadius: "50%",
              background: "var(--gradient-brand)",
              display: "flex", alignItems: "center", justifyContent: "center", fontWeight: "bold",
            }}>
              {userEmail.charAt(0).toUpperCase()}
            </div>
            <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: "0.875rem" }}>
              {userEmail}
            </div>
          </div>
          <button className="btn btn-ghost" style={{ width: "100%", justifyContent: "flex-start" }} onClick={handleLogout}>
            <LogOut size={16} /> Logout
          </button>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main-content" style={{ marginLeft: isSidebarOpen ? "320px" : "0", transition: "margin-left 0.3s ease" }}>

        {/* Header */}
        <header style={{
          height: "64px",
          borderBottom: "var(--glass-border)",
          display: "flex",
          alignItems: "center",
          padding: "0 2rem",
          background: "rgba(10, 10, 15, 0.8)",
          backdropFilter: "blur(12px)",
          position: "sticky",
          top: 0,
          zIndex: 40,
        }}>
          {!isSidebarOpen && (
            <button onClick={() => setSidebarOpen(true)} style={{ marginRight: "1rem", color: "var(--text-muted)" }}>
              <Menu size={20} />
            </button>
          )}
          {activeDoc ? (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", width: "100%" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ color: "var(--text-muted)" }}>Analyzing:</span>
                <span style={{ fontWeight: 600, color: "white" }}>{activeDoc.filename}</span>
                <span style={{
                  fontSize: "0.75rem", padding: "0.25rem 0.5rem",
                  background: "rgba(59, 130, 246, 0.1)", color: "var(--accent-primary)",
                  borderRadius: "1rem", marginLeft: "0.5rem",
                }}>
                  {activeDoc.service_name}
                </span>
              </div>
              <button
                onClick={handleDeleteDocument}
                disabled={isDeleting}
                style={{
                  display: "flex", alignItems: "center", gap: "0.5rem",
                  color: "var(--text-muted)", fontSize: "0.875rem",
                  background: "transparent", border: "none", cursor: "pointer",
                  opacity: isDeleting ? 0.5 : 1, transition: "color 0.2s",
                }}
                onMouseEnter={e => (e.currentTarget.style.color = "#ef4444")}
                onMouseLeave={e => (e.currentTarget.style.color = "var(--text-muted)")}
              >
                {isDeleting ? <Loader2 className="spinner" size={16} /> : <Trash2 size={16} />}
                Delete
              </button>
            </div>
          ) : (
            <span style={{ fontWeight: 600, color: "white" }}>Upload Document</span>
          )}
        </header>

        {/* ── Model loading banner ── */}
        {!modelReady && (
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
            padding: "0.625rem 2rem",
            background: "rgba(251, 191, 36, 0.08)",
            borderBottom: "1px solid rgba(251, 191, 36, 0.2)",
          }}>
            <Loader2 className="spinner" size={15} color="#fbbf24" />
            <span style={{ fontSize: "0.8125rem", color: "#fbbf24" }}>
              {modelChecking
                ? "AI model is warming up — this takes ~60 seconds on a cold start…"
                : "Connecting to AI…"}
            </span>
          </div>
        )}

        {/* Toggle Bar */}
        {activeDoc && (
          <div style={{
            display: "flex",
            justifyContent: "center",
            padding: "1rem 0",
            background: "rgba(10, 10, 15, 0.4)",
            borderBottom: "1px solid rgba(255,255,255,0.05)",
          }}>
            <div style={{
              display: "flex",
              background: "rgba(0,0,0,0.3)",
              padding: "4px",
              borderRadius: "12px",
              border: "1px solid rgba(255,255,255,0.05)",
            }}>
              <button
                onClick={() => setActiveTab("summary")}
                style={{
                  padding: "8px 24px",
                  borderRadius: "8px",
                  fontSize: "0.875rem",
                  fontWeight: 600,
                  transition: "all 0.2s",
                  background: activeTab === "summary" ? "var(--gradient-brand)" : "transparent",
                  color: activeTab === "summary" ? "white" : "var(--text-muted)",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                Analysis View
              </button>
              <button
                onClick={() => setActiveTab("chat")}
                style={{
                  padding: "8px 24px",
                  borderRadius: "8px",
                  fontSize: "0.875rem",
                  fontWeight: 600,
                  transition: "all 0.2s",
                  background: activeTab === "chat" ? "var(--gradient-brand)" : "transparent",
                  color: activeTab === "chat" ? "white" : "var(--text-muted)",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                Interactive Chat
              </button>
            </div>
          </div>
        )}

        {/* Content */}
        <div style={{ flex: 1, overflowY: "auto", padding: "2rem" }}>

          {!activeDoc ? (

            /* ── Upload view ── */
            <div style={{ maxWidth: "600px", margin: "4rem auto 0" }}>
              <div className="glass-card" style={{ textAlign: "center", padding: "4rem 2rem" }}>
                <div style={{ display: "inline-flex", padding: "1rem", background: "rgba(59, 130, 246, 0.1)", borderRadius: "50%", marginBottom: "1.5rem" }}>
                  <Upload size={32} color="var(--accent-primary)" />
                </div>
                <h2 style={{ fontSize: "1.5rem", fontWeight: 600, marginBottom: "0.5rem" }}>Upload a Legal Document</h2>
                <p className="text-muted" style={{ marginBottom: "2rem" }}>
                  We'll index your document and prepare the AI for analysis.
                </p>

                <form onSubmit={handleUpload} style={{ textAlign: "left" }}>
                  <div className="input-group">
                    <label className="input-label">Service Name (e.g. Spotify, Apple)</label>
                    <input
                      type="text"
                      className="input-field"
                      value={serviceName}
                      onChange={e => setServiceName(e.target.value)}
                      required
                    />
                  </div>
                  <div className="input-group">
                    <label className="input-label">Document Type</label>
                    <select className="input-field" value={docType} onChange={e => setDocType(e.target.value)}>
                      <option>Terms of Service</option>
                      <option>Privacy Policy</option>
                      <option>Cookie Policy</option>
                      <option>EULA</option>
                    </select>
                  </div>
                  <div className="input-group">
                    <label className="input-label">PDF File</label>
                    <input
                      type="file"
                      accept=".pdf"
                      className="input-field"
                      onChange={e => setFile(e.target.files?.[0] ?? null)}
                      required
                      style={{ padding: "0.5rem", background: "rgba(255,255,255,0.02)" }}
                    />
                  </div>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    style={{ width: "100%", marginTop: "1rem" }}
                    disabled={isUploading}
                  >
                    {isUploading
                      ? <><Loader2 className="spinner" size={18} /> Uploading…</>
                      : "Ingest Document"
                    }
                  </button>
                </form>
              </div>
            </div>

          ) : (

            /* ── Active document view ── */
            <div style={{ height: "calc(100vh - 180px)", display: "flex", flexDirection: "column" }}>

              {activeTab === "summary" ? (

                /* ── Analysis View ── */
                <div style={{
                  display: "flex", flexDirection: "column", gap: "1.5rem",
                  overflowY: "auto", paddingRight: "1rem",
                  maxWidth: "1000px", margin: "0 auto", width: "100%",
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <h2 style={{ fontSize: "1.25rem", fontWeight: 600, display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <Search size={20} color="var(--accent-primary)" /> Key Clause Analysis
                    </h2>
                    <button
                      className="btn btn-secondary"
                      style={{
                        padding: "0.5rem 1rem",
                        fontSize: "0.75rem",
                        opacity: !modelReady ? 0.5 : 1,
                        cursor: !modelReady ? "not-allowed" : "pointer",
                      }}
                      onClick={handleGenerateSummary}
                      disabled={isSummarizing || docStatus === "processing" || !modelReady}
                      title={!modelReady ? "AI model is still loading…" : undefined}
                    >
                      {isSummarizing
                        ? <><Loader2 className="spinner" size={14} /> Analyzing…</>
                        : !modelReady
                          ? "AI Loading…"
                          : summaryTopics.length > 0 ? "Refresh Analysis" : "Generate Analysis"
                      }
                    </button>
                  </div>

                  {/* Processing banner */}
                  {docStatus === "processing" && (
                    <div style={{
                      padding: "1rem",
                      background: "rgba(251,191,36,0.08)",
                      border: "1px solid rgba(251,191,36,0.2)",
                      borderRadius: "0.75rem",
                      display: "flex", alignItems: "center", gap: "0.75rem",
                    }}>
                      <Loader2 className="spinner" size={18} color="#fbbf24" />
                      <span style={{ fontSize: "0.875rem", color: "#fbbf24" }}>
                        Indexing document in background… This may take a minute.
                      </span>
                    </div>
                  )}

                  {/* Summary error banner */}
                  {summaryError && (
                    <div style={{
                      padding: "1rem",
                      background: "rgba(239,68,68,0.08)",
                      border: "1px solid rgba(239,68,68,0.2)",
                      borderRadius: "0.75rem",
                      fontSize: "0.875rem",
                      color: "#f87171",
                    }}>
                      {summaryError}
                    </div>
                  )}

                  {summaryTopics.length === 0 && !isSummarizing && !summaryError && docStatus !== "processing" && (
                    <div className="glass-card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "4rem 2rem", opacity: 0.7 }}>
                      <Info size={32} color="var(--text-muted)" style={{ marginBottom: "1rem" }} />
                      <p style={{ textAlign: "center", color: "var(--text-secondary)" }}>
                        {modelReady
                          ? "Click generate to extract key clauses across 12 legal topics."
                          : "Waiting for AI model to finish loading…"}
                      </p>
                    </div>
                  )}

                  {summaryTopics.map((topic, idx) => (
                    <div key={idx} className="glass-card" style={{ padding: "1.5rem", animation: "fadeIn 0.4s ease-out" }}>
                      <h3 style={{
                        fontSize: "1rem", fontWeight: 600, marginBottom: "0.75rem",
                        color: topic.summary.includes("NOT_IN_DOCUMENT") ? "var(--text-muted)" : "white",
                      }}>
                        {topic.label}
                      </h3>
                      <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                        {topic.summary.includes("NOT_IN_DOCUMENT")
                          ? "This topic is not explicitly covered in the provided text."
                          : renderCitedText(topic.summary, topic.sources)
                        }
                      </p>

                      {topic.sources && topic.sources.length > 0 && (
                        <div style={{ marginTop: "1rem", display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                          <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.25rem" }}>
                            Source Passages
                          </p>
                          {topic.sources.map((src, sIdx) => {
                            const pageLabel = `Page ${src.page}`;
                            const sectionLabel = src.section && src.section !== "General" ? src.section : null;
                            const preview = src.excerpt ? src.excerpt.replace(/\s+/g, " ").trim().slice(0, 80) + "…" : "";
                            return (
                              <details key={sIdx} style={{ background: "rgba(59,130,246,0.05)", borderRadius: "0.5rem", border: "1px solid rgba(59,130,246,0.15)", overflow: "hidden" }}>
                                <summary style={{ cursor: "pointer", padding: "0.5rem 0.75rem", display: "flex", alignItems: "center", gap: "0.5rem", listStyle: "none" }}>
                                  <span style={{ background: "rgba(59,130,246,0.2)", color: "var(--accent-primary)", fontWeight: 700, fontSize: "0.7rem", padding: "1px 6px", borderRadius: "4px", flexShrink: 0 }}>
                                    {pageLabel}
                                  </span>
                                  {sectionLabel && (
                                    <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", background: "rgba(255,255,255,0.05)", padding: "1px 6px", borderRadius: "4px", flexShrink: 0 }}>
                                      {sectionLabel}
                                    </span>
                                  )}
                                  <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                    {preview}
                                  </span>
                                </summary>
                                <div style={{ padding: "0.75rem 1rem", borderTop: "1px solid rgba(59,130,246,0.1)", fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.6, fontStyle: "italic" }}>
                                  &ldquo;{src.excerpt}&rdquo;
                                </div>
                              </details>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  ))}

                  {isSummarizing && (
                    <div className="glass-card" style={{ padding: "1.5rem", opacity: 0.5 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                        <Loader2 className="spinner" size={16} color="var(--accent-primary)" />
                        <span style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>Analyzing next topic…</span>
                      </div>
                    </div>
                  )}
                </div>

              ) : (

                /* ── Chat View ── */
                <div className="glass-card" style={{
                  flex: 1, display: "flex", flexDirection: "column",
                  padding: "1rem", overflow: "hidden",
                  maxWidth: "900px", margin: "0 auto", width: "100%",
                }}>
                  <div style={{
                    flex: 1, overflowY: "auto", padding: "1.5rem",
                    display: "flex", flexDirection: "column", gap: "1.5rem",
                  }}>
                    {messages.length === 0 && (
                      <div style={{ margin: "auto", textAlign: "center", color: "var(--text-muted)" }}>
                        <p>
                          {modelReady
                            ? "No messages yet. Ask a question about this document!"
                            : "AI model is warming up, please wait…"}
                        </p>
                      </div>
                    )}

                    {/* Chat error banner */}
                    {chatError && (
                      <div style={{
                        padding: "0.75rem 1rem",
                        background: "rgba(239,68,68,0.08)",
                        border: "1px solid rgba(239,68,68,0.2)",
                        borderRadius: "0.75rem",
                        fontSize: "0.875rem",
                        color: "#f87171",
                      }}>
                        {chatError}
                      </div>
                    )}

                    {messages.map((msg, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: "flex", flexDirection: "column",
                          alignItems: msg.role === "user" ? "flex-end" : "flex-start",
                          animation: "fadeIn 0.3s ease-out",
                        }}
                      >
                        <div style={{
                          maxWidth: "85%",
                          padding: "1rem 1.25rem",
                          borderRadius: "1rem",
                          background: msg.role === "user" ? "var(--gradient-brand)" : "rgba(255,255,255,0.05)",
                          color: "white",
                          boxShadow: msg.role === "user" ? "0 4px 12px rgba(59, 130, 246, 0.2)" : "none",
                          fontSize: "0.9375rem",
                          lineHeight: 1.5,
                        }}>
                          {msg.isStreaming && !msg.content
                            ? <Loader2 className="spinner" size={16} color="var(--text-muted)" />
                            : renderCitedText(msg.content, msg.cited_sources)
                          }
                        </div>

                        {msg.role === "assistant" && msg.cited_sources && msg.cited_sources.length > 0 && (
                          <div style={{ marginTop: "0.5rem", width: "88%", display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                            <p style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.1rem" }}>
                              Sources
                            </p>
                            {msg.cited_sources.map((src, sIdx) => {
                              const pageLabel = `Page ${src.page}`;
                              const sectionLabel = src.section && src.section !== "General" ? src.section : null;
                              const preview = src.excerpt ? src.excerpt.replace(/\s+/g, " ").trim().slice(0, 80) + "…" : "";
                              return (
                                <details key={sIdx} style={{ background: "rgba(59,130,246,0.05)", borderRadius: "0.5rem", border: "1px solid rgba(59,130,246,0.15)", overflow: "hidden" }}>
                                  <summary style={{ cursor: "pointer", padding: "0.5rem 0.75rem", display: "flex", alignItems: "center", gap: "0.5rem", listStyle: "none" }}>
                                    <span style={{ background: "rgba(59,130,246,0.2)", color: "var(--accent-primary)", fontWeight: 700, fontSize: "0.7rem", padding: "1px 6px", borderRadius: "4px", flexShrink: 0 }}>
                                      {pageLabel}
                                    </span>
                                    {sectionLabel && (
                                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", background: "rgba(255,255,255,0.05)", padding: "1px 6px", borderRadius: "4px", flexShrink: 0 }}>
                                        {sectionLabel}
                                      </span>
                                    )}
                                    <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                      {preview}
                                    </span>
                                  </summary>
                                  <div style={{ padding: "0.75rem 1rem", borderTop: "1px solid rgba(59,130,246,0.1)", fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.6, fontStyle: "italic" }}>
                                    &ldquo;{src.excerpt}&rdquo;
                                  </div>
                                </details>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    ))}
                    <div ref={chatEndRef} />
                  </div>

                  {/* Chat Input */}
                  <form
                    onSubmit={handleSendMessage}
                    style={{
                      display: "flex", gap: "1rem", padding: "1.5rem",
                      borderTop: "1px solid rgba(255,255,255,0.05)",
                    }}
                  >
                    <input
                      type="text"
                      className="input-field"
                      placeholder={modelReady ? "Ask a specific question…" : "Waiting for AI model…"}
                      value={chatInput}
                      onChange={e => setChatInput(e.target.value)}
                      disabled={isChatting || !modelReady}
                    />
                    <button
                      type="submit"
                      className="btn btn-primary"
                      disabled={isChatting || !chatInput.trim() || !modelReady}
                      style={{ padding: "0.75rem" }}
                      title={!modelReady ? "AI model is still loading…" : undefined}
                    >
                      {isChatting ? <Loader2 className="spinner" size={18} /> : <Send size={18} />}
                    </button>
                  </form>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}