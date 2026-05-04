"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Scale, ArrowRight, Loader2, Mail, Lock } from "lucide-react";

const API_URL = "http://localhost:8000";

export default function AuthPage() {
  const router = useRouter();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Redirect if already logged in
  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (token) {
      router.push("/dashboard");
    }
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);

    const endpoint = isLogin ? "/auth/login" : "/auth/signup";
    
    try {
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Authentication failed.");
      }

      if (isLogin) {
        localStorage.setItem("access_token", data.access_token);
        localStorage.setItem("user_id", data.user_id);
        localStorage.setItem("user_email", data.email);
        router.push("/dashboard");
      } else {
        setSuccess("Account created! Please check your email to confirm, then log in.");
        setIsLogin(true);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ 
      minHeight: "100vh", 
      display: "flex", 
      flexDirection: "column", 
      alignItems: "center", 
      justifyContent: "center",
      padding: "2rem"
    }}>
      
      {/* Decorative background elements */}
      <div style={{
        position: "absolute",
        top: "-10%",
        right: "-5%",
        width: "500px",
        height: "500px",
        borderRadius: "50%",
        background: "radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%)",
        zIndex: 0
      }} />

      <div style={{ position: "relative", zIndex: 1, width: "100%", maxWidth: "420px" }}>
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <div style={{ 
            display: "inline-flex", 
            padding: "1rem", 
            background: "rgba(59, 130, 246, 0.1)", 
            borderRadius: "1rem",
            marginBottom: "1rem"
          }}>
            <Scale size={40} color="var(--accent-primary)" />
          </div>
          <h1 className="heading-gradient" style={{ fontSize: "2.5rem", fontWeight: 700, marginBottom: "0.5rem" }}>
            TOS Summarizer
          </h1>
          <p className="text-muted">AI-powered legal document analysis</p>
        </div>

        <div className="glass-card">
          <div style={{ display: "flex", gap: "1rem", marginBottom: "2rem", borderBottom: "var(--glass-border)", paddingBottom: "1rem" }}>
            <button 
              className="btn" 
              style={{ flex: 1, background: isLogin ? "rgba(255,255,255,0.1)" : "transparent", color: isLogin ? "white" : "var(--text-muted)" }}
              onClick={() => { setIsLogin(true); setError(""); setSuccess(""); }}
            >
              Log In
            </button>
            <button 
              className="btn" 
              style={{ flex: 1, background: !isLogin ? "rgba(255,255,255,0.1)" : "transparent", color: !isLogin ? "white" : "var(--text-muted)" }}
              onClick={() => { setIsLogin(false); setError(""); setSuccess(""); }}
            >
              Sign Up
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <label className="input-label">Email Address</label>
              <div style={{ position: "relative" }}>
                <Mail size={18} style={{ position: "absolute", left: "1rem", top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)" }} />
                <input 
                  type="email" 
                  required
                  className="input-field" 
                  style={{ paddingLeft: "2.75rem" }}
                  placeholder="name@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            <div className="input-group">
              <label className="input-label">Password</label>
              <div style={{ position: "relative" }}>
                <Lock size={18} style={{ position: "absolute", left: "1rem", top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)" }} />
                <input 
                  type="password" 
                  required
                  className="input-field" 
                  style={{ paddingLeft: "2.75rem" }}
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  minLength={isLogin ? 1 : 6}
                />
              </div>
            </div>

            {error && (
              <div style={{ padding: "0.75rem", background: "rgba(239, 68, 68, 0.1)", color: "var(--accent-danger)", borderRadius: "0.5rem", fontSize: "0.875rem", marginBottom: "1rem" }}>
                {error}
              </div>
            )}

            {success && (
              <div style={{ padding: "0.75rem", background: "rgba(16, 185, 129, 0.1)", color: "var(--accent-success)", borderRadius: "0.5rem", fontSize: "0.875rem", marginBottom: "1rem" }}>
                {success}
              </div>
            )}

            <button type="submit" className="btn btn-primary" style={{ width: "100%", marginTop: "1rem" }} disabled={loading}>
              {loading ? <Loader2 className="spinner" size={18} /> : (isLogin ? "Log In" : "Create Account")}
              {!loading && <ArrowRight size={18} />}
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}
