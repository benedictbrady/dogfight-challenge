"use client";

import { useState } from "react";
import Scene from "../Scene";
import Controls from "../Controls";
import { useMatch, Frame } from "../../hooks/useMatch";

const OPPONENTS = ["do_nothing", "dogfighter", "chaser", "ace", "brawler"];

const DEFAULT_FRAME: Frame = {
  tick: 0,
  fighters: [
    { x: -200, y: 300, yaw: 0, speed: 40, hp: 5, alive: true },
    { x: 200, y: 300, yaw: Math.PI, speed: 40, hp: 5, alive: true },
  ],
  bullets: [],
  hits: [],
};

interface CheckpointMatchViewerProps {
  checkpoints: string[];
  runName: string;
}

const selectStyle: React.CSSProperties = {
  background: "var(--bg-secondary)",
  color: "var(--text-primary)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  padding: "6px 8px",
  fontSize: 13,
  fontFamily: "var(--font-mono, monospace)",
  outline: "none",
};

const inputStyle: React.CSSProperties = {
  ...selectStyle,
  width: 72,
};

export default function CheckpointMatchViewer({
  checkpoints,
  runName,
}: CheckpointMatchViewerProps) {
  const [selectedCheckpoint, setSelectedCheckpoint] = useState(
    checkpoints.length > 0 ? checkpoints[checkpoints.length - 1] : ""
  );
  const [opponent, setOpponent] = useState("brawler");
  const [seed, setSeed] = useState(42);

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

  if (checkpoints.length === 0) return null;

  const handleWatchMatch = () => {
    const onnxPath = `training/dashboard_data/${runName}/checkpoints/${selectedCheckpoint}.onnx`;
    startMatch({ p0: onnxPath, p1: opponent, seed });
  };

  const currentFrame = frames[currentFrameIndex] ?? DEFAULT_FRAME;
  const totalFrames = frames.length;

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
    >
      <div className="px-4 pt-4 pb-3">
        <h3
          className="text-[11px] font-semibold uppercase tracking-wider mb-3"
          style={{ color: "var(--text-secondary)" }}
        >
          Checkpoint Match Viewer
        </h3>

        {/* Controls */}
        <div className="flex items-end gap-3 flex-wrap">
          <div className="flex-1 min-w-[140px]">
            <label className="block text-[10px] mb-1" style={{ color: "var(--text-secondary)" }}>
              Checkpoint
            </label>
            <select
              style={{ ...selectStyle, width: "100%" }}
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
            >
              {checkpoints.map((cp) => (
                <option key={cp} value={cp}>
                  {cp}
                </option>
              ))}
            </select>
          </div>

          <div className="flex-1 min-w-[120px]">
            <label className="block text-[10px] mb-1" style={{ color: "var(--text-secondary)" }}>
              Opponent
            </label>
            <select
              style={{ ...selectStyle, width: "100%" }}
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
              style={inputStyle}
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>

          <button
            className="px-4 py-1.5 rounded-md text-xs font-semibold tracking-wider transition-all"
            style={{
              background: "var(--accent)",
              color: "#0a0a0a",
              border: "none",
            }}
            onClick={handleWatchMatch}
          >
            Watch
          </button>
        </div>
      </div>

      {/* Result banner */}
      {matchResult && (
        <div
          className="mx-4 mb-3 text-xs text-center py-2 rounded-md"
          style={{
            fontFamily: "var(--font-mono, monospace)",
            background: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            color: "var(--text-secondary)",
          }}
        >
          <span
            style={{
              color:
                matchResult.winner === 0
                  ? "var(--accent)"
                  : matchResult.winner === 1
                  ? "var(--red)"
                  : "var(--text-secondary)",
            }}
          >
            {matchResult.winner === null
              ? "Draw"
              : `Player ${matchResult.winner} wins`}
          </span>
          {matchResult.reason && (
            <span style={{ color: "var(--text-secondary)", marginLeft: 8 }}>
              ({matchResult.reason})
            </span>
          )}
        </div>
      )}

      {/* Scene */}
      <div style={{ height: 400 }}>
        <Scene
          frame={currentFrame}
          trailFrames={frames.slice(0, currentFrameIndex + 1)}
        />
      </div>

      {/* Playback controls */}
      <Controls
        currentFrame={currentFrameIndex}
        totalFrames={totalFrames}
        isPlaying={isPlaying}
        speed={speed}
        onSetFrame={setCurrentFrameIndex}
        onTogglePlay={() => setIsPlaying(!isPlaying)}
        onSetSpeed={setSpeed}
      />
    </div>
  );
}
