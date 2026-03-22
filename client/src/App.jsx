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
  const [speed, setSpeed] = useState(50);
  const [selectedVoice, setSelectedVoice] = useState(1);
  const [dotGrid, setDotGrid] = useState(() =>
    Array.from({ length: 900 }, () => Math.random() > 0.95)
  );

  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const activeRef = useRef(true);

  const running = runtime?.running ?? false;

  useEffect(() => {
    const interval = setInterval(() => {
      setDotGrid((prevGrid) =>
        prevGrid.map((isActive) => {
          if (isActive) return Math.random() > 0.4;
          return Math.random() < 0.01;
        })
      );
    }, 150);
    return () => clearInterval(interval);
  }, []);

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

  const darkBackground = "#111";
  const grayBox = "#d3d3d3";

  const outlinedButtonStyle = {
    border: "1px solid white",
    background: "transparent",
    color: "white",
    padding: "10px 18px",
    cursor: "pointer",
    fontFamily: "monospace",
    textTransform: "uppercase",
    fontSize: "1rem",
    fontWeight: "bold",
    letterSpacing: "1px",
    borderRadius: "8px",
  };

  return (
    <div
      style={{
        backgroundColor: darkBackground,
        color: "white",
        fontFamily: "monospace",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          minHeight: "100vh",
          padding: "48px 56px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          boxSizing: "border-box",
          gap: "42px",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", gap: "40px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "20px", flex: 1 }}>
            <div
              style={{
                color: "#666",
                fontSize: "1.4rem",
                fontWeight: "bold",
                letterSpacing: "4px",
              }}
            >
              SIGN-SPEAK CONSOLE
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap" }}>
              <button type="button" style={outlinedButtonStyle}>
                Adjust Sign Speed
              </button>
              <input
                type="range"
                min="0"
                max="100"
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                style={{ accentColor: "white", width: "210px" }}
              />
              <span style={{ color: "#bdbdbd" }}>{speed}</span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
              <button type="button" style={outlinedButtonStyle}>
                Pick A Voice
              </button>
              {[1, 2, 3, 4].map((num) => (
                <button
                  type="button"
                  key={num}
                  onClick={() => setSelectedVoice(num)}
                  style={{
                    ...outlinedButtonStyle,
                    backgroundColor: selectedVoice === num ? "white" : "transparent",
                    color: selectedVoice === num ? "black" : "white",
                    borderRadius: "50%",
                    padding: "11px 16px",
                    fontSize: "1rem",
                  }}
                >
                  {num}
                </button>
              ))}
            </div>

            <div style={{ display: "flex", alignItems: "flex-end", gap: "12px", flexWrap: "wrap" }}>
              <label style={{ display: "flex", flexDirection: "column", gap: "6px", color: "#b8b8b8" }}>
                Mode
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  disabled={running}
                  style={{ ...outlinedButtonStyle, textTransform: "none", minWidth: "120px" }}
                >
                  <option value="letters">letters</option>
                  <option value="words">words</option>
                  <option value="hybrid">hybrid</option>
                </select>
              </label>

              <label style={{ display: "flex", flexDirection: "column", gap: "6px", color: "#b8b8b8" }}>
                Source
                <input
                  value={source}
                  onChange={(e) => setSource(e.target.value)}
                  disabled={running}
                  style={{ ...outlinedButtonStyle, textTransform: "none", minWidth: "90px" }}
                />
              </label>

              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  border: "1px solid #666",
                  padding: "10px 12px",
                  borderRadius: "8px",
                }}
              >
                <input
                  type="checkbox"
                  checked={voice}
                  onChange={(e) => setVoice(e.target.checked)}
                  disabled={running}
                />
                Enable voice
              </label>
            </div>

            <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
              <button
                type="button"
                onClick={startRuntime}
                disabled={running}
                style={{ ...outlinedButtonStyle, fontSize: "1.2rem", padding: "14px 28px", opacity: running ? 0.5 : 1 }}
              >
                Start
              </button>
              <button
                type="button"
                onClick={stopRuntime}
                disabled={!running}
                style={{ ...outlinedButtonStyle, fontSize: "1.2rem", padding: "14px 28px", opacity: !running ? 0.5 : 1 }}
              >
                Stop
              </button>
              <button
                type="button"
                onClick={() => sendCommand("speak")}
                disabled={!running || !voice}
                style={{
                  ...outlinedButtonStyle,
                  backgroundColor: "white",
                  color: "black",
                  fontSize: "1.2rem",
                  padding: "14px 24px",
                  opacity: !running || !voice ? 0.5 : 1,
                }}
              >
                Speak
              </button>
            </div>

            <div style={{ color: error ? "#ff7f8d" : "#bdbdbd", fontSize: "0.95rem", minHeight: "1.2rem" }}>
              {error ||
                `stream: ${streamStatus} | runtime: ${running ? "running" : "stopped"} | voice: ${
                  runtime?.speaking ? "speaking" : voice ? "ready" : "off"
                }`}
            </div>
          </div>

          <div
            style={{
              textAlign: "right",
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-end",
              width: "42%",
              minWidth: "300px",
            }}
          >
            <div style={{ display: "grid", gridTemplateColumns: "repeat(30, 6px)", gap: "6px", marginBottom: "20px" }}>
              {dotGrid.map((isActive, i) => (
                <div
                  key={i}
                  style={{
                    width: "6px",
                    height: "6px",
                    borderRadius: "50%",
                    transition: "background-color 0.3s ease, box-shadow 0.3s ease",
                    backgroundColor: isActive ? "white" : "#222",
                    boxShadow: isActive ? "0 0 10px rgba(255,255,255,0.8)" : "none",
                  }}
                />
              ))}
            </div>
            <p style={{ margin: 0, fontSize: "1.05rem", color: "#999", lineHeight: "1.6", maxWidth: "410px" }}>
              Real-time ASL recognition with live camera preview, confidence meter, phrase buffer,
              and transcript output.
            </p>
          </div>
        </div>

        <div style={{ textAlign: "center", paddingBottom: "8px" }}>
          <h1
            style={{
              margin: 0,
              fontSize: "5rem",
              textTransform: "uppercase",
              fontWeight: "lighter",
              letterSpacing: "12px",
              WebkitTextStroke: "2px white",
              color: darkBackground,
            }}
          >
            SIGN - SPEAK
          </h1>
        </div>
      </div>

      <hr style={{ border: "none", borderTop: "2px solid #333", margin: "0 56px" }} />

      <div
        style={{
          minHeight: "100vh",
          padding: "72px 56px",
          display: "flex",
          gap: "34px",
          boxSizing: "border-box",
          flexWrap: "wrap",
        }}
      >
        <div
          style={{
            flex: "1.5 1 620px",
            backgroundColor: grayBox,
            borderRadius: "20px",
            padding: "24px",
            display: "flex",
            flexDirection: "column",
            gap: "12px",
          }}
        >
          <p style={{ color: "#444", margin: 0, textTransform: "uppercase", fontSize: "1.05rem", fontWeight: "bold", letterSpacing: "2px" }}>
            Live Camera Preview
          </p>
          {running ? (
            <img
              src={frameUrl}
              alt="live camera"
              style={{
                width: "100%",
                borderRadius: "14px",
                border: "1px solid #a8a8a8",
                background: "#9d9d9d",
                aspectRatio: "4/3",
                objectFit: "cover",
              }}
            />
          ) : (
            <div
              style={{
                flexGrow: 1,
                minHeight: "360px",
                borderRadius: "14px",
                border: "1px dashed #777",
                display: "grid",
                placeItems: "center",
                color: "#6b6b6b",
                fontSize: "1.15rem",
                textTransform: "uppercase",
                letterSpacing: "2px",
              }}
            >
              Runtime Stopped
            </div>
          )}
        </div>

        <div style={{ flex: "1 1 420px", display: "flex", flexDirection: "column", gap: "28px" }}>
          <div
            style={{
              flex: 1,
              backgroundColor: grayBox,
              borderRadius: "20px",
              padding: "28px",
              display: "flex",
              flexDirection: "column",
              gap: "10px",
            }}
          >
            <p style={{ color: "#444", margin: 0, textTransform: "uppercase", fontSize: "1.05rem", fontWeight: "bold", letterSpacing: "2px" }}>
              Live Capture (temporary)
            </p>
            <p style={{ color: "#111", margin: 0, fontWeight: "bold" }}>
              last token: {runtime?.chosen_label ?? "-"} | channel: {runtime?.chosen_mode ?? "none"}
            </p>
            <p style={{ color: "#222", margin: 0 }}>confidence: {formatConf(runtime?.chosen_conf)}</p>
            <div style={{ height: "10px", border: "1px solid #888", borderRadius: "999px", overflow: "hidden", background: "#f2f2f2" }}>
              <div
                style={{
                  height: "100%",
                  width: `${confidencePct}%`,
                  background: "linear-gradient(90deg, #44d4b0, #95ffe0)",
                  transition: "width 120ms linear",
                }}
              />
            </div>
            <div style={{ color: "black", fontSize: "1.3rem", lineHeight: 1.35, minHeight: "3.4rem", border: "1px dashed #9d9d9d", borderRadius: "10px", padding: "10px" }}>
              {runtime?.text || "(none)"}
            </div>
            <div style={{ color: "#2f2f2f", fontSize: "1.1rem", minHeight: "2.2rem", border: "1px dashed #9d9d9d", borderRadius: "10px", padding: "10px" }}>
              {runtime?.phrase || "(empty)"}
            </div>
            <div style={{ marginTop: "auto", display: "flex", gap: "10px", flexWrap: "wrap" }}>
              <button
                type="button"
                onClick={() => sendCommand("clear_text")}
                disabled={!running}
                style={{ ...outlinedButtonStyle, color: "#2d2d2d", borderColor: "#7f7f7f", opacity: !running ? 0.5 : 1 }}
              >
                Clear Text
              </button>
              <button
                type="button"
                onClick={() => sendCommand("clear_phrase")}
                disabled={!running}
                style={{ ...outlinedButtonStyle, color: "#2d2d2d", borderColor: "#7f7f7f", opacity: !running ? 0.5 : 1 }}
              >
                Clear Phrase
              </button>
            </div>
          </div>

          <div
            style={{
              flex: 1,
              backgroundColor: grayBox,
              borderRadius: "20px",
              padding: "28px",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <p style={{ color: "#444", margin: "0 0 12px 0", textTransform: "uppercase", fontSize: "1.05rem", fontWeight: "bold", letterSpacing: "2px" }}>
              Transcript
            </p>
            <p style={{ color: "black", fontSize: "1.4rem", lineHeight: 1.45, margin: 0, minHeight: "5.4rem" }}>
              {displayTranscript || "TEXT -> SPEECH SHOWN HERE"}
            </p>
            <p style={{ marginTop: "auto", marginBottom: 0, fontSize: "0.95rem", color: "#555", fontWeight: "bold" }}>
              pending word: {runtime?.pending_word || "-"} | pending char: {runtime?.pending_char || "-"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
