"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { LogOut, FileText, Upload, Link as LinkIcon, Search, Send, Loader2, Info, ChevronRight, Menu } from "lucide-react";

const API_URL = "http://localhost:8000";

interface Document {
  id: string;
  filename: string;
  service_name: string;
  doc_type: string;
  created_at: string;
}

interface Source {
  tag: string;
  citation: string;
  section: string;
  page: number;
  excerpt: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  cited_sources?: Source[];
}

export default function Dashboard() {
  const router = useRouter();
  
  // Auth state
  const [userEmail, setUserEmail] = useState("");
  const [token, setToken] = useState("");

  // UI state
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  
  // Data state
  const [historyDocs, setHistoryDocs] = useState<Document[]>([]);
  const [activeDoc, setActiveDoc] = useState<Document | null>(null);
  
  // Upload state
  const [uploadMode, setUploadMode] = useState<"pdf" | "url">("pdf");
  const [file, setFile] = useState<File | null>(null);
  const [urlInput, setUrlInput] = useState("");
  const [serviceName, setServiceName] = useState("");
  const [docType, setDocType] = useState("Terms of Service");
  const [isUploading, setIsUploading] = useState(false);

  // Chat & Summary state
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatting, setIsChatting] = useState(false);
  const [summaryTopics, setSummaryTopics] = useState<any[]>([]);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [docStatus, setDocStatus] = useState<string>("ready");
  const chatEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Initial load
  useEffect(() => {
    const access_token = localStorage.getItem("access_token");
    const email = localStorage.getItem("user_email");
    if (!access_token) {
      router.push("/");
      return;
    }
    setToken(access_token);
    setUserEmail(email || "User");
    fetchHistory(access_token);
  }, [router]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchHistory = async (auth_token: string) => {
    try {
      const res = await fetch(`${API_URL}/history/documents`, {
        headers: { Authorization: `Bearer ${auth_token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setHistoryDocs(data.documents);
      }
    } catch (err) {
      console.error("Failed to load history", err);
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    router.push("/");
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) { alert("Please select a file."); return; }
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("service_name", serviceName || "Unknown");
      formData.append("doc_type", docType);

      const res = await fetch(`${API_URL}/ingest/pdf`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Upload failed");
      }

      const data = await res.json();
      const newDoc: Document = {
        id: data.document_id,
        filename: file.name,
        service_name: serviceName || "Unknown",
        doc_type: docType,
        created_at: new Date().toISOString(),
      };

      setActiveDoc(newDoc);
      setMessages([]);
      setSummaryTopics([]);
      setDocStatus("processing");
      fetchHistory(token);

      // Poll every 3s until the document is ready
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const r = await fetch(`${API_URL}/documents/${data.document_id}/status`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (r.ok) {
            const s = await r.json();
            setDocStatus(s.status);
            if (s.status === "ready" || s.status === "error") {
              clearInterval(pollRef.current!);
              pollRef.current = null;
              fetchHistory(token);
            }
          }
        } catch { /* ignore */ }
      }, 3000);

    } catch (err: any) {
      alert(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const loadPastDocument = async (doc: Document) => {
    setActiveDoc(doc);
    setSummaryData(null);
    setMessages([]);
    
    try {
      const res = await fetch(`${API_URL}/history/chats/${doc.id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        const loadedMsgs: Message[] = [];
        data.sessions.forEach((s: any) => {
          s.messages.forEach((m: any) => {
            loadedMsgs.push({
              role: m.role,
              content: m.content,
              cited_sources: m.cited_sources
            });
          });
        });
        setMessages(loadedMsgs);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleGenerateSummary = async () => {
    if (!activeDoc) return;
    setIsSummarizing(true);
    setSummaryTopics([]);

    const params = new URLSearchParams({
      document_id:  activeDoc.id,
      service_name: activeDoc.service_name,
      doc_type:     activeDoc.doc_type,
    });

    try {
      const res = await fetch(`${API_URL}/summary/stream?${params}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.body) throw new Error("No stream body");

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const event = JSON.parse(line.slice(6));
          if (event.type === "topic_ready") {
            setSummaryTopics(prev => [...prev, event.data]);
          }
        }
      }
    } catch (err) {
      console.error(err);
      alert("Failed to generate summary.");
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || !activeDoc) return;

    const userMsg = chatInput;
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    // Placeholder for the streaming assistant message
    setMessages(prev => [...prev, { role: "assistant", content: "", cited_sources: [] }]);
    setChatInput("");
    setIsChatting(true);

    try {
      const res = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          query:        userMsg,
          document_id:  activeDoc.id,
          service_name: activeDoc.service_name,
        }),
      });
      if (!res.body) throw new Error("No stream");

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const event = JSON.parse(line.slice(6));
          if (event.type === "token") {
            // Append token to the last message in-place
            setMessages(prev => {
              const next = [...prev];
              const last = { ...next[next.length - 1] };
              last.content += event.data;
              next[next.length - 1] = last;
              return next;
            });
          } else if (event.type === "sources") {
            setMessages(prev => {
              const next = [...prev];
              const last = { ...next[next.length - 1] };
              last.cited_sources = event.data;
              next[next.length - 1] = last;
              return next;
            });
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", content: "Sorry, an error occurred." };
        return next;
      });
    } finally {
      setIsChatting(false);
    }
  };

  // Helper to render inline citations safely
  const renderCitedText = (text: string, sources?: Source[]) => {
    if (!sources || sources.length === 0) return text;
    
    const tagMap: Record<string, number> = {};
    sources.forEach((s, i) => { tagMap[s.tag] = i + 1; });

    // Simple replacement for [SOURCE X]
    const parts = text.split(/(\[SOURCE \d+\])/g);
    
    return parts.map((part, i) => {
      if (part.match(/\[SOURCE \d+\]/)) {
        const num = tagMap[part];
        if (!num) return part;
        const source = sources[num - 1];
        return (
          <span 
            key={i} 
            title={source.citation}
            style={{
              color: "var(--accent-primary)",
              fontWeight: 600,
              fontSize: "0.75em",
              verticalAlign: "super",
              cursor: "help",
              borderBottom: "1px dotted var(--accent-primary)",
              marginLeft: "2px"
            }}
          >
            [{num}]
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="layout-container">
      
      {/* ── Sidebar ── */}
      <aside className="sidebar" style={{ transform: isSidebarOpen ? "translateX(0)" : "translateX(-100%)", transition: "transform 0.3s ease", position: "absolute", height: "100%", zIndex: 50, left: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem" }}>
          <h2 className="heading-gradient" style={{ fontSize: "1.25rem", fontWeight: 700 }}>TOS Summarizer</h2>
          <button onClick={() => setSidebarOpen(false)} style={{ color: "var(--text-muted)" }}>
            <Menu size={20} />
          </button>
        </div>

        <div style={{ flex: 1, overflowY: "auto" }}>
          <div style={{ marginBottom: "2rem" }}>
            <h3 style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em", marginBottom: "1rem" }}>
              My Documents
            </h3>
            
            <button 
              className="btn btn-secondary" 
              style={{ width: "100%", justifyContent: "flex-start", marginBottom: "1rem" }}
              onClick={() => { setActiveDoc(null); setSidebarOpen(false); }}
            >
              + New Document
            </button>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              {historyDocs.map(doc => (
                <button 
                  key={doc.id}
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
                    transition: "all 0.2s"
                  }}
                >
                  <FileText size={16} style={{ color: activeDoc?.id === doc.id ? "var(--accent-primary)" : "var(--text-muted)" }} />
                  <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    <div style={{ fontSize: "0.875rem", fontWeight: 500 }}>{doc.filename}</div>
                    <div style={{ fontSize: "0.75rem", opacity: 0.7 }}>{doc.service_name}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div style={{ borderTop: "var(--glass-border)", paddingTop: "1.5rem", marginTop: "auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
            <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: "var(--gradient-brand)", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: "bold" }}>
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

      {/* ── Main Content ── */}
      <main className="main-content" style={{ marginLeft: isSidebarOpen ? "320px" : "0", transition: "margin-left 0.3s ease" }}>
        
        {/* Top Navbar */}
        <header style={{ height: "64px", borderBottom: "var(--glass-border)", display: "flex", alignItems: "center", padding: "0 2rem", background: "rgba(10, 10, 15, 0.8)", backdropFilter: "blur(12px)", position: "sticky", top: 0, zIndex: 40 }}>
          {!isSidebarOpen && (
            <button onClick={() => setSidebarOpen(true)} style={{ marginRight: "1rem", color: "var(--text-muted)" }}>
              <Menu size={20} />
            </button>
          )}
          {activeDoc ? (
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ color: "var(--text-muted)" }}>Analyzing:</span>
              <span style={{ fontWeight: 600, color: "white" }}>{activeDoc.filename}</span>
              <span style={{ fontSize: "0.75rem", padding: "0.25rem 0.5rem", background: "rgba(59, 130, 246, 0.1)", color: "var(--accent-primary)", borderRadius: "1rem", marginLeft: "0.5rem" }}>
                {activeDoc.service_name}
              </span>
            </div>
          ) : (
            <span style={{ fontWeight: 600, color: "white" }}>Upload Document</span>
          )}
        </header>

        {/* Content Area */}
        <div style={{ flex: 1, overflowY: "auto", padding: "2rem" }}>
          
          {!activeDoc ? (
            /* Upload State */
            <div style={{ maxWidth: "600px", margin: "4rem auto 0" }}>
              <div className="glass-card" style={{ textAlign: "center", padding: "4rem 2rem" }}>
                <div style={{ display: "inline-flex", padding: "1rem", background: "rgba(59, 130, 246, 0.1)", borderRadius: "50%", marginBottom: "1.5rem" }}>
                  <Upload size={32} color="var(--accent-primary)" />
                </div>
                <h2 style={{ fontSize: "1.5rem", fontWeight: 600, marginBottom: "0.5rem" }}>Upload a Legal Document</h2>
                <p className="text-muted" style={{ marginBottom: "2rem" }}>We'll index the document into our vector database and prepare the AI.</p>
                
                <form onSubmit={handleUpload} style={{ textAlign: "left" }}>
                  <div className="input-group">
                    <label className="input-label">Service Name (e.g. Spotify, Apple)</label>
                    <input type="text" className="input-field" value={serviceName} onChange={e => setServiceName(e.target.value)} required />
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
                      onChange={e => setFile(e.target.files?.[0] || null)} 
                      required 
                      style={{ padding: "0.5rem", background: "rgba(255,255,255,0.02)" }}
                    />
                  </div>
                  
                  <button type="submit" className="btn btn-primary" style={{ width: "100%", marginTop: "1rem" }} disabled={isUploading}>
                    {isUploading ? <><Loader2 className="spinner" size={18} /> Uploading...</> : "Ingest Document"}
                  </button>
                </form>
              </div>
            </div>
          ) : (
            /* Active Document State (Two Columns: Summary & Chat) */
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem", height: "calc(100vh - 128px)" }}>
              
              {/* Left Column: Summary */}
              <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", overflowY: "auto", paddingRight: "1rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <h2 style={{ fontSize: "1.25rem", fontWeight: 600, display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <Search size={20} color="var(--accent-primary)" /> Global Summary
                  </h2>
                  <button 
                    className="btn btn-secondary" 
                    style={{ padding: "0.5rem 1rem", fontSize: "0.75rem" }}
                    onClick={handleGenerateSummary}
                    disabled={isSummarizing || docStatus === "processing"}
                  >
                    {isSummarizing ? <><Loader2 className="spinner" size={14} /> Analyzing...</> : "Generate Summary"}
                  </button>
                </div>

                {/* Status banner when processing */}
                {docStatus === "processing" && (
                  <div style={{ padding: "1rem", background: "rgba(251,191,36,0.08)", border: "1px solid rgba(251,191,36,0.2)", borderRadius: "0.75rem", display: "flex", alignItems: "center", gap: "0.75rem" }}>
                    <Loader2 className="spinner" size={18} color="#fbbf24" />
                    <span style={{ fontSize: "0.875rem", color: "#fbbf24" }}>Indexing document in background… This may take a minute.</span>
                  </div>
                )}
                
                {summaryTopics.length === 0 && !isSummarizing && docStatus !== "processing" && (
                  <div className="glass-card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "4rem 2rem", opacity: 0.7 }}>
                    <Info size={32} color="var(--text-muted)" style={{ marginBottom: "1rem" }} />
                    <p style={{ textAlign: "center", color: "var(--text-secondary)" }}>Click generate to extract key clauses across 12 legal topics.</p>
                  </div>
                )}
                
                {summaryTopics.map((topic: any, idx: number) => (
                  <div key={idx} className="glass-card" style={{ padding: "1.5rem", animation: "fadeIn 0.4s ease-out" }}>
                    <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.75rem", color: topic.summary.includes("NOT_IN_DOCUMENT") ? "var(--text-muted)" : "white" }}>
                      {topic.label}
                    </h3>
                    <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                      {topic.summary.includes("NOT_IN_DOCUMENT") 
                        ? "This topic is not explicitly covered in the provided text." 
                        : renderCitedText(topic.summary, topic.sources)
                      }
                    </p>
                    
                    {topic.sources && topic.sources.length > 0 && (
                      <div style={{ marginTop: "1rem", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: "1rem" }}>
                        <p style={{ fontSize: "0.75rem", fontWeight: 600, marginBottom: "0.5rem", color: "var(--text-muted)" }}>CITED SOURCES</p>
                        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                          {topic.sources.map((src: Source, sIdx: number) => (
                            <div key={sIdx} style={{ padding: "0.75rem", background: "rgba(0,0,0,0.2)", borderRadius: "0.5rem", border: "1px solid rgba(255,255,255,0.03)" }}>
                              <div style={{ fontSize: "0.75rem", color: "var(--accent-primary)", marginBottom: "0.25rem", fontWeight: 500 }}>
                                [{sIdx + 1}] {src.citation} {src.section ? `· ${src.section}` : ""}
                              </div>
                              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontStyle: "italic" }}>
                                "{src.excerpt}"
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {/* Skeleton card while more topics are being generated */}
                {isSummarizing && (
                  <div className="glass-card" style={{ padding: "1.5rem", opacity: 0.5 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                      <Loader2 className="spinner" size={16} color="var(--accent-primary)" />
                      <span style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>Analyzing next topic…</span>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Right Column: Chat */}
              <div className="glass-card" style={{ display: "flex", flexDirection: "column", padding: "1rem", overflow: "hidden" }}>
                <div style={{ padding: "1rem", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                  <h2 style={{ fontSize: "1.25rem", fontWeight: 600 }}>Chat Interface</h2>
                  <p className="text-muted" style={{ fontSize: "0.75rem" }}>Ask specific questions about the document.</p>
                </div>
                
                <div style={{ flex: 1, overflowY: "auto", padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                  {messages.length === 0 && (
                    <div style={{ margin: "auto", textAlign: "center", color: "var(--text-muted)" }}>
                      <p>No messages yet. Ask a question below!</p>
                    </div>
                  )}
                  
                  {messages.map((msg, idx) => (
                    <div key={idx} style={{ 
                      display: "flex", 
                      flexDirection: "column",
                      alignItems: msg.role === "user" ? "flex-end" : "flex-start",
                      animation: "fadeIn 0.3s ease-out"
                    }}>
                      <div style={{ 
                        maxWidth: "85%",
                        padding: "1rem 1.25rem", 
                        borderRadius: "1rem",
                        background: msg.role === "user" ? "var(--gradient-brand)" : "rgba(255,255,255,0.05)",
                        border: msg.role === "assistant" ? "1px solid rgba(255,255,255,0.1)" : "none",
                        color: "white",
                        fontSize: "0.875rem",
                        lineHeight: 1.6
                      }}>
                        {msg.role === "assistant" ? renderCitedText(msg.content, msg.cited_sources) : msg.content}
                      </div>
                      
                      {/* Show sources directly under assistant messages if available */}
                      {msg.role === "assistant" && msg.cited_sources && msg.cited_sources.length > 0 && (
                        <div style={{ marginTop: "0.5rem", width: "85%", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                          {msg.cited_sources.map((src, sIdx) => (
                            <details key={sIdx} style={{ fontSize: "0.75rem", background: "rgba(0,0,0,0.2)", padding: "0.5rem", borderRadius: "0.5rem", color: "var(--text-muted)" }}>
                              <summary style={{ cursor: "pointer", color: "var(--text-secondary)", fontWeight: 500 }}>
                                [{sIdx + 1}] {src.citation}
                              </summary>
                              <div style={{ marginTop: "0.5rem", paddingLeft: "1rem", borderLeft: "2px solid rgba(255,255,255,0.1)" }}>
                                {src.excerpt}
                              </div>
                            </details>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                  
                  {isChatting && (
                    <div style={{ display: "flex", alignItems: "flex-start" }}>
                      <div style={{ padding: "1rem 1.25rem", borderRadius: "1rem", background: "rgba(255,255,255,0.05)", display: "flex", gap: "0.5rem" }}>
                        <span className="spinner" style={{ width: "8px", height: "8px", background: "var(--accent-primary)", borderRadius: "50%" }}></span>
                        <span className="spinner" style={{ width: "8px", height: "8px", background: "var(--accent-primary)", borderRadius: "50%", animationDelay: "0.2s" }}></span>
                        <span className="spinner" style={{ width: "8px", height: "8px", background: "var(--accent-primary)", borderRadius: "50%", animationDelay: "0.4s" }}></span>
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>
                
                <form onSubmit={handleSendMessage} style={{ marginTop: "auto", paddingTop: "1rem" }}>
                  <div style={{ position: "relative" }}>
                    <input 
                      type="text" 
                      className="input-field" 
                      placeholder={isSummarizing ? "AI is busy generating summary..." : "Ask about data sharing, refunds, governing law..."}
                      value={chatInput}
                      onChange={e => setChatInput(e.target.value)}
                      disabled={isChatting || isSummarizing}
                      style={{ paddingRight: "3rem", background: "rgba(0,0,0,0.3)" }}
                    />
                    <button 
                      type="submit" 
                      disabled={isChatting || isSummarizing || !chatInput.trim()}
                      style={{ 
                        position: "absolute", right: "0.5rem", top: "50%", transform: "translateY(-50%)",
                        background: chatInput.trim() && !isSummarizing ? "var(--accent-primary)" : "transparent",
                        color: chatInput.trim() && !isSummarizing ? "white" : "var(--text-muted)",
                        padding: "0.4rem", borderRadius: "0.375rem", transition: "all 0.2s"
                      }}
                    >
                      <Send size={16} />
                    </button>
                  </div>
                </form>
              </div>

            </div>
          )}
        </div>
      </main>
    </div>
  );
}
