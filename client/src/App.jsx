import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const WS_BASE = API_BASE.replace("http", "ws");

function formatConf(value) {
  if (typeof value !== "number") return "-";
  return `${Math.round(value * 100)}%`;
}

export default function App() {
  const [runtime, setRuntime] = useState(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [source, setSource] = useState("0");
  const [voice, setVoice] = useState(false);
  const [streamStatus, setStreamStatus] = useState("connecting");

  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const activeRef = useRef(true);

  const running = runtime?.running ?? false;

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then((data) => {
        if (!active) return;
        if (data?.runtime) {
          setRuntime(data.runtime);
          setMode(data.runtime.mode ?? "hybrid");
          setVoice(Boolean(data.runtime.voice_enabled));
        }
      })
      .catch(() => {
        if (!active) return;
        setError("Could not reach API. Start runtime_api.py first.");
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    activeRef.current = true;

    function connect() {
      setStreamStatus(reconnectAttemptsRef.current > 0 ? "reconnecting" : "connecting");
      const ws = new WebSocket(`${WS_BASE}/stream`);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectAttemptsRef.current = 0;
        if (!activeRef.current) return;
        setStreamStatus("connected");
      };

      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          setRuntime(data);
          setVoice(Boolean(data.voice_enabled));
          setError(data.error ?? "");
        } catch {
          setError("Failed to parse stream message.");
        }
      };

      ws.onclose = () => {
        if (!activeRef.current) return;
        reconnectAttemptsRef.current += 1;
        const waitMs = Math.min(5000, 500 * 2 ** reconnectAttemptsRef.current);
        setStreamStatus("reconnecting");
        reconnectTimerRef.current = window.setTimeout(connect, waitMs);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      activeRef.current = false;
      setStreamStatus("closed");
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  async function refreshState() {
    const state = await fetch(`${API_BASE}/state`).then((r) => r.json());
    setRuntime(state);
  }

  async function startRuntime() {
    setError("");
    const payload = { mode, source, voice };
    const res = await fetch(`${API_BASE}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      setError(body.detail ?? "Failed to start runtime.");
      return;
    }

    await refreshState();
  }

  async function stopRuntime() {
    setError("");
    await fetch(`${API_BASE}/stop`, { method: "POST" });
    await refreshState();
  }

  async function sendCommand(action) {
    setError("");
    const res = await fetch(`${API_BASE}/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      setError(body.detail ?? `Failed to run ${action}.`);
    }
  }

  const confidencePct = useMemo(() => {
    const conf = runtime?.chosen_conf ?? 0;
    return Math.max(0, Math.min(100, Math.round(conf * 100)));
  }, [runtime?.chosen_conf]);

  const displayTranscript = useMemo(() => {
    if (!runtime) return "";
    if (runtime.transcript) return runtime.transcript;
    if (runtime.mode === "hybrid") {
      return [runtime.phrase?.trim(), runtime.text?.trim()].filter(Boolean).join(" ").trim();
    }
    if (runtime.mode === "words") return runtime.phrase ?? "";
    return runtime.text ?? "";
  }, [runtime]);

  const frameUrl = useMemo(() => {
    const seq = runtime?.seq ?? 0;
    return `${API_BASE}/frame?seq=${seq}`;
  }, [runtime?.seq]);

  return (
    <main className="page-shell">
      <section className="hero-card">
        <h1>Sign-Speak Live Console</h1>
        <p>
          React + Vite interface powered by your Python runtime. Start recognition, then monitor
          live tokens, camera preview, phrase buffer, and spelled text in real time.
        </p>
        <div className="status-row">
          <span className={`status-badge ${streamStatus}`}>stream: {streamStatus}</span>
          <span className={`status-badge ${running ? "running" : "stopped"}`}>
            runtime: {running ? "running" : "stopped"}
          </span>
          <span className={`status-badge ${runtime?.speaking ? "speaking" : "idle"}`}>
            voice: {runtime?.speaking ? "speaking" : voice ? "ready" : "off"}
          </span>
        </div>
        <div className="control-row">
          <label>
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value)} disabled={running}>
              <option value="letters">letters</option>
              <option value="words">words</option>
              <option value="hybrid">hybrid</option>
            </select>
          </label>
          <label>
            Source
            <input value={source} onChange={(e) => setSource(e.target.value)} disabled={running} />
          </label>
          <label className="voice-toggle">
            <input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} disabled={running} />
            Enable voice
          </label>
          <button type="button" className="btn-primary" onClick={startRuntime} disabled={running}>
            Start
          </button>
          <button type="button" className="btn-ghost" onClick={stopRuntime} disabled={!running}>
            Stop
          </button>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="workspace">
        <article className="panel preview-panel">
          <div className="preview-header">
            <h2>Live Camera Preview</h2>
            <button type="button" className="btn-primary" onClick={() => sendCommand("speak")} disabled={!running || !voice}>
              Speak
            </button>
          </div>
          {running ? <img src={frameUrl} alt="live camera" className="preview" /> : <p className="output">Runtime is stopped.</p>}
        </article>

        <div className="right-column">
          <article className="panel status-panel">
            <h2>Status</h2>
            <p>
              Active mode: <strong>{runtime?.mode ?? "-"}</strong>
            </p>
            <p>
              Last token: <strong>{runtime?.chosen_label ?? "-"}</strong>
            </p>
            <p>
              Token channel: <strong>{runtime?.chosen_mode ?? "none"}</strong>
            </p>
            <div className="meter-wrap">
              <span>Confidence</span>
              <div className="meter">
                <div className="meter-fill" style={{ width: `${confidencePct}%` }} />
              </div>
              <span>{formatConf(runtime?.chosen_conf)}</span>
            </div>
          </article>

          <article className="panel text-panel">
            <h2>Live Letter Capture (temporary)</h2>
            <p className="output">{runtime?.text || "(none)"}</p>
            <h2>Live Word Buffer (temporary)</h2>
            <p className="output">{runtime?.phrase || "(empty)"}</p>
            <div className="inline-actions">
              <button type="button" className="btn-ghost" onClick={() => sendCommand("clear_text")} disabled={!running}>
                Clear Text
              </button>
              <button type="button" className="btn-ghost" onClick={() => sendCommand("clear_phrase")} disabled={!running}>
                Clear Phrase
              </button>
            </div>
          </article>

          <article className="panel transcript-panel">
            <h2>Transcript</h2>
            <p className="output">{displayTranscript || "(empty)"}</p>
            <div className="mini-row">
              <span>pending word: {runtime?.pending_word || "-"}</span>
              <span>pending char: {runtime?.pending_char || "-"}</span>
            </div>
          </article>
        </div>
      </section>
    </main>
  );
}
