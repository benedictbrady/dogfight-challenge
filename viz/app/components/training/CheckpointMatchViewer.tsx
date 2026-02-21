"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import Scene from "../Scene";
import Controls from "../Controls";
import { useMatch, Frame } from "../../hooks/useMatch";

const OPPONENTS = ["brawler", "ace", "chaser", "dogfighter", "do_nothing"];

const DEFAULT_FRAME: Frame = {
  tick: 0,
  fighters: [
    { x: -200, y: 300, yaw: 0, speed: 40, hp: 5, alive: true },
    { x: 200, y: 300, yaw: Math.PI, speed: 40, hp: 5, alive: true },
  ],
  bullets: [],
  hits: [],
};

type ViewerStatus = "idle" | "checking-server" | "downloading" | "running" | "playing" | "error";

interface MatchRecord {
  winner: number | null;
  reason: string;
}

interface CheckpointMatchViewerProps {
  checkpoints: string[];
  runName: string;
}

/** Parse step number from checkpoint name like "step_500000" or "curriculum_final". */
function parseStep(name: string): number | null {
  const m = name.match(/step_(\d+)/);
  return m ? parseInt(m[1], 10) : null;
}

/** Format step as human-readable string. */
function formatStep(step: number): string {
  if (step >= 1_000_000) return `${(step / 1_000_000).toFixed(1)}M`;
  if (step >= 1_000) return `${(step / 1_000).toFixed(0)}k`;
  return String(step);
}

/** Guess training stage from step count. */
function guessStage(name: string): string {
  if (name.includes("curriculum")) return "Curriculum";
  if (name.includes("transition")) return "Transition";
  if (name.includes("final")) return "Final";
  return "";
}

/** Build a label for a checkpoint. */
function checkpointLabel(name: string): string {
  const step = parseStep(name);
  if (step !== null) return formatStep(step);
  // Named checkpoints like "curriculum_final"
  return name.replace(/_/g, " ");
}

/** Key for result tracking. */
function resultKey(checkpoint: string, opponent: string): string {
  return `${checkpoint}:${opponent}`;
}

export default function CheckpointMatchViewer({
  checkpoints,
  runName,
}: CheckpointMatchViewerProps) {
  const [selectedCheckpoint, setSelectedCheckpoint] = useState(
    checkpoints.length > 0 ? checkpoints[checkpoints.length - 1] : ""
  );
  const [opponent, setOpponent] = useState("brawler");
  const [seed, setSeed] = useState(42);
  const [status, setStatus] = useState<ViewerStatus>("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const [results, setResults] = useState<Record<string, MatchRecord>>({});
  const [expanded, setExpanded] = useState(false);
  const matchStartedRef = useRef(false);

  const {
    frames,
    currentFrameIndex,
    setCurrentFrameIndex,
    isPlaying,
    setIsPlaying,
    speed,
    setSpeed,
    matchResult,
    startMatch,
  } = useMatch();

  // Check server status on mount and when expanded
  useEffect(() => {
    if (!expanded) return;
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch("http://localhost:3001/api/policies", {
          signal: AbortSignal.timeout(3000),
        });
        if (!cancelled) setServerOnline(res.ok);
      } catch {
        if (!cancelled) setServerOnline(false);
      }
    };
    check();
    return () => { cancelled = true; };
  }, [expanded]);

  // Track match results
  useEffect(() => {
    if (matchResult && matchStartedRef.current) {
      const key = resultKey(selectedCheckpoint, opponent);
      setResults((prev) => ({ ...prev, [key]: matchResult }));
      setStatus("playing");
      matchStartedRef.current = false;
    }
  }, [matchResult, selectedCheckpoint, opponent]);

  // When frames start arriving, update status
  useEffect(() => {
    if (frames.length > 0 && status === "running") {
      setStatus("playing");
    }
  }, [frames.length, status]);

  const handleWatch = useCallback(async () => {
    setErrorMsg("");

    // 1. Check server
    setStatus("checking-server");
    try {
      const res = await fetch("http://localhost:3001/api/policies", {
        signal: AbortSignal.timeout(3000),
      });
      if (!res.ok) throw new Error("Server returned error");
      setServerOnline(true);
    } catch {
      setServerOnline(false);
      setStatus("error");
      setErrorMsg("Rust backend not running. Start it with: make serve");
      return;
    }

    // 2. Download ONNX checkpoint
    setStatus("downloading");
    try {
      const res = await fetch(
        `/api/training/checkpoint-download?run=${encodeURIComponent(runName)}&checkpoint=${encodeURIComponent(selectedCheckpoint)}`
      );
      if (!res.ok) {
        const data = await res.json().catch(() => ({ error: "Download failed" }));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      const { path: onnxPath } = await res.json();

      // 3. Start match via WebSocket
      setStatus("running");
      matchStartedRef.current = true;
      startMatch({ p0: onnxPath, p1: opponent, seed });
    } catch (err) {
      setStatus("error");
      setErrorMsg(err instanceof Error ? err.message : "Failed to download checkpoint");
    }
  }, [runName, selectedCheckpoint, opponent, seed, startMatch]);

  // Sort checkpoints for timeline: numeric steps first (sorted), then named
  const sortedCheckpoints = useMemo(() => {
    return [...checkpoints].sort((a, b) => {
      const sa = parseStep(a);
      const sb = parseStep(b);
      if (sa !== null && sb !== null) return sa - sb;
      if (sa !== null) return -1;
      if (sb !== null) return 1;
      return a.localeCompare(b);
    });
  }, [checkpoints]);

  if (checkpoints.length === 0) return null;

  const currentFrame = frames[currentFrameIndex] ?? DEFAULT_FRAME;
  const totalFrames = frames.length;
  const isLoading = status === "checking-server" || status === "downloading" || status === "running";

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
    >
      {/* Header — always visible, click to expand */}
      <button
        onClick={() => setExpanded((e) => !e)}
        className="w-full flex items-center justify-between px-4 py-3 transition-colors"
        style={{ background: expanded ? "transparent" : "transparent" }}
      >
        <div className="flex items-center gap-2">
          <span
            className="text-[11px] font-semibold uppercase tracking-wider"
            style={{ color: "var(--text-secondary)" }}
          >
            Checkpoint Match Viewer
          </span>
          <span
            className="text-[10px] px-1.5 py-0.5 rounded"
            style={{
              background: "rgba(91, 155, 245, 0.1)",
              color: "var(--accent)",
              border: "1px solid rgba(91, 155, 245, 0.15)",
            }}
          >
            {checkpoints.length} checkpoints
          </span>
        </div>
        <span
          className="text-xs transition-transform"
          style={{
            color: "var(--text-secondary)",
            transform: expanded ? "rotate(180deg)" : "rotate(0deg)",
          }}
        >
          &#9662;
        </span>
      </button>

      {!expanded ? null : (
        <div className="px-4 pb-4">
          {/* Server status warning */}
          {serverOnline === false && status !== "error" && (
            <div
              className="text-xs rounded-md px-3 py-2 mb-3"
              style={{
                background: "rgba(251, 146, 60, 0.08)",
                border: "1px solid rgba(251, 146, 60, 0.2)",
                color: "rgb(251, 146, 60)",
              }}
            >
              Rust backend not detected on :3001. Run <code
                style={{
                  background: "rgba(255,255,255,0.06)",
                  padding: "1px 4px",
                  borderRadius: 3,
                  fontFamily: "var(--font-mono, monospace)",
                }}
              >make serve</code> to enable match playback.
            </div>
          )}

          {/* Timeline */}
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-secondary)" }}>
                Timeline
              </span>
            </div>
            <div className="relative">
              {/* Track line */}
              <div
                className="absolute top-1/2 left-0 right-0 h-px"
                style={{ background: "var(--border)", transform: "translateY(-50%)" }}
              />
              {/* Markers */}
              <div className="relative flex items-center" style={{ minHeight: 36 }}>
                {sortedCheckpoints.map((cp, i) => {
                  const isSelected = cp === selectedCheckpoint;
                  const step = parseStep(cp);
                  const rKey = resultKey(cp, opponent);
                  const record = results[rKey];

                  // Position: evenly spaced
                  const pct = sortedCheckpoints.length === 1
                    ? 50
                    : (i / (sortedCheckpoints.length - 1)) * 100;

                  return (
                    <button
                      key={cp}
                      onClick={() => setSelectedCheckpoint(cp)}
                      className="absolute flex flex-col items-center transition-all group"
                      style={{
                        left: `${pct}%`,
                        transform: "translateX(-50%)",
                        zIndex: isSelected ? 10 : 1,
                      }}
                      title={`${cp}${step !== null ? ` (step ${step.toLocaleString()})` : ""}`}
                    >
                      {/* Result badge */}
                      {record && (
                        <span
                          className="text-[8px] font-bold mb-0.5 rounded px-1"
                          style={{
                            background: record.winner === 0
                              ? "rgba(74, 222, 128, 0.15)"
                              : record.winner === 1
                              ? "rgba(248, 113, 113, 0.15)"
                              : "rgba(107, 107, 107, 0.15)",
                            color: record.winner === 0
                              ? "var(--green)"
                              : record.winner === 1
                              ? "var(--red)"
                              : "var(--text-secondary)",
                          }}
                        >
                          {record.winner === 0 ? "W" : record.winner === 1 ? "L" : "D"}
                        </span>
                      )}
                      {/* Dot */}
                      <span
                        className="rounded-full transition-all"
                        style={{
                          width: isSelected ? 10 : 6,
                          height: isSelected ? 10 : 6,
                          background: isSelected ? "var(--accent)" : "var(--text-secondary)",
                          boxShadow: isSelected ? "0 0 8px var(--accent)" : "none",
                          opacity: isSelected ? 1 : 0.5,
                        }}
                      />
                      {/* Label — show on selected or hover */}
                      <span
                        className="text-[9px] mt-0.5 whitespace-nowrap transition-opacity"
                        style={{
                          color: isSelected ? "var(--accent)" : "var(--text-secondary)",
                          opacity: isSelected ? 1 : 0,
                          fontFamily: "var(--font-mono, monospace)",
                        }}
                      >
                        {checkpointLabel(cp)}
                        {guessStage(cp) && (
                          <span style={{ opacity: 0.6 }}> {guessStage(cp)}</span>
                        )}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Controls row */}
          <div className="flex items-end gap-3 flex-wrap mb-3">
            <div>
              <label className="block text-[10px] mb-1" style={{ color: "var(--text-secondary)" }}>
                Checkpoint
              </label>
              <span
                className="text-xs px-2 py-1 rounded"
                style={{
                  background: "var(--bg-secondary)",
                  border: "1px solid var(--border)",
                  fontFamily: "var(--font-mono, monospace)",
                  color: "var(--text-primary)",
                }}
              >
                {selectedCheckpoint}
              </span>
            </div>

            <div className="min-w-[110px]">
              <label className="block text-[10px] mb-1" style={{ color: "var(--text-secondary)" }}>
                Opponent
              </label>
              <select
                style={{
                  background: "var(--bg-secondary)",
                  color: "var(--text-primary)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  padding: "4px 6px",
                  fontSize: 12,
                  fontFamily: "var(--font-mono, monospace)",
                  outline: "none",
                  width: "100%",
                }}
                value={opponent}
                onChange={(e) => setOpponent(e.target.value)}
              >
                {OPPONENTS.map((opp) => (
                  <option key={opp} value={opp}>
                    {opp}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-[10px] mb-1" style={{ color: "var(--text-secondary)" }}>
                Seed
              </label>
              <input
                type="number"
                style={{
                  background: "var(--bg-secondary)",
                  color: "var(--text-primary)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  padding: "4px 6px",
                  fontSize: 12,
                  fontFamily: "var(--font-mono, monospace)",
                  outline: "none",
                  width: 64,
                }}
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </div>

            <button
              className="px-4 py-1.5 rounded-md text-xs font-semibold tracking-wider transition-all disabled:opacity-40"
              style={{
                background: isLoading ? "var(--bg-secondary)" : "var(--accent)",
                color: isLoading ? "var(--text-secondary)" : "#0a0a0a",
                border: isLoading ? "1px solid var(--border)" : "none",
              }}
              disabled={isLoading}
              onClick={handleWatch}
            >
              {status === "checking-server"
                ? "Checking..."
                : status === "downloading"
                ? "Downloading..."
                : status === "running"
                ? "Starting..."
                : "Watch"}
            </button>
          </div>

          {/* Error message */}
          {status === "error" && errorMsg && (
            <div
              className="text-xs rounded-md px-3 py-2 mb-3"
              style={{
                background: "rgba(248, 113, 113, 0.08)",
                border: "1px solid rgba(248, 113, 113, 0.2)",
                color: "var(--red)",
              }}
            >
              {errorMsg}
            </div>
          )}

          {/* Result banner */}
          {matchResult && (
            <div
              className="text-xs text-center py-2 rounded-md mb-3"
              style={{
                fontFamily: "var(--font-mono, monospace)",
                background: matchResult.winner === 0
                  ? "rgba(74, 222, 128, 0.06)"
                  : matchResult.winner === 1
                  ? "rgba(248, 113, 113, 0.06)"
                  : "var(--bg-secondary)",
                border: `1px solid ${
                  matchResult.winner === 0
                    ? "rgba(74, 222, 128, 0.2)"
                    : matchResult.winner === 1
                    ? "rgba(248, 113, 113, 0.2)"
                    : "var(--border)"
                }`,
              }}
            >
              <span
                style={{
                  color: matchResult.winner === 0
                    ? "var(--green)"
                    : matchResult.winner === 1
                    ? "var(--red)"
                    : "var(--text-secondary)",
                  fontWeight: 600,
                }}
              >
                {matchResult.winner === null
                  ? "Draw"
                  : matchResult.winner === 0
                  ? "Agent Wins"
                  : "Agent Loses"}
              </span>
              {matchResult.reason && (
                <span style={{ color: "var(--text-secondary)", marginLeft: 8 }}>
                  ({matchResult.reason})
                </span>
              )}
            </div>
          )}

          {/* Scene — only show when we have frames or are running */}
          {(frames.length > 0 || status === "running") && (
            <>
              <div className="rounded-md overflow-hidden" style={{ height: 360, border: "1px solid var(--border)" }}>
                <Scene
                  frame={currentFrame}
                  trailFrames={frames.slice(0, currentFrameIndex + 1)}
                />
              </div>

              {/* Playback controls */}
              <div
                className="flex items-center gap-3 px-3 py-2 rounded-b-md"
                style={{ background: "var(--bg-secondary)", borderTop: "1px solid var(--border)" }}
              >
                <button
                  className="px-3 py-1 text-[11px] rounded transition-colors"
                  style={{
                    background: "var(--bg-card)",
                    color: "var(--text-primary)",
                    border: "1px solid var(--border)",
                  }}
                  disabled={totalFrames === 0}
                  onClick={() => setIsPlaying(!isPlaying)}
                >
                  {isPlaying ? "Pause" : "Play"}
                </button>

                <div className="flex-1">
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, totalFrames - 1)}
                    value={currentFrameIndex}
                    onChange={(e) => setCurrentFrameIndex(Number(e.target.value))}
                    className="w-full h-1 rounded-lg appearance-none cursor-pointer"
                    style={{ accentColor: "var(--accent)" }}
                    disabled={totalFrames === 0}
                  />
                </div>

                <span
                  className="text-[10px] min-w-[60px] text-right"
                  style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono, monospace)" }}
                >
                  {currentFrameIndex + 1} / {totalFrames}
                </span>

                <div className="flex items-center gap-1">
                  {[0.5, 1, 2, 4].map((s) => (
                    <button
                      key={s}
                      className="px-1.5 py-0.5 text-[10px] rounded transition-colors"
                      style={{
                        background: speed === s ? "rgba(91, 155, 245, 0.15)" : "transparent",
                        color: speed === s ? "var(--accent)" : "var(--text-secondary)",
                        border: speed === s ? "1px solid rgba(91, 155, 245, 0.25)" : "1px solid transparent",
                      }}
                      onClick={() => setSpeed(s)}
                    >
                      {s}x
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
