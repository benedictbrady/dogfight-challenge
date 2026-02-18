"use client";

import { useMemo, useState } from "react";
import Scene from "./components/Scene";
import Controls from "./components/Controls";
import MatchSetup from "./components/MatchSetup";
import DebugOverlay from "./components/DebugOverlay";
import { useMatch, MatchConfig, Frame } from "./hooks/useMatch";

export default function Home() {
  const {
    frames,
    currentFrameIndex,
    setCurrentFrameIndex,
    isPlaying,
    setIsPlaying,
    speed,
    setSpeed,
    matchResult,
    isConnected,
    startMatch,
  } = useMatch();

  const [cameraMode] = useState<"free" | "chase0" | "chase1">("free");

  // Default frame: both planes at starting positions before match begins
  const idleFrame = useMemo<Frame>(() => ({
    tick: 0,
    fighters: [
      { x: -200, y: 300, yaw: 0, speed: 40, hp: 3, alive: true },
      { x: 200, y: 300, yaw: Math.PI, speed: 40, hp: 3, alive: true },
    ],
    bullets: [],
    hits: [],
  }), []);

  const currentFrame = frames[currentFrameIndex] ?? idleFrame;
  const prevFrame = currentFrameIndex > 0 ? (frames[currentFrameIndex - 1] ?? null) : null;
  const totalFrames = frames.length;

  const handleStartMatch = (config: MatchConfig) => {
    startMatch(config);
  };

  return (
    <div className="w-screen h-screen flex flex-col bg-[#1c1a14]">
      {/* Top bar */}
      <div className="flex-shrink-0 h-10 bg-[#1c1a14] border-b border-[#3a3526] flex items-center px-4 gap-4 z-10">
        <span className="text-sm font-bold text-[#c4b894] tracking-[0.15em]">
          DOGFIGHT
        </span>

        {/* P0 health bar */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-[#c4a050]">P0</span>
          <div className="w-24 h-3 bg-[#2a2015] rounded-sm border border-[#3a3526] overflow-hidden">
            <div
              className="h-full bg-[#c4a050] transition-all duration-200"
              style={{ width: `${(currentFrame.fighters[0].hp / 3) * 100}%` }}
            />
          </div>
        </div>

        {/* P1 health bar */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-[#b83030]">P1</span>
          <div className="w-24 h-3 bg-[#2a2015] rounded-sm border border-[#3a3526] overflow-hidden">
            <div
              className="h-full bg-[#b83030] transition-all duration-200"
              style={{ width: `${(currentFrame.fighters[1].hp / 3) * 100}%` }}
            />
          </div>
        </div>

        <span className="text-xs text-[#6b5a40]">
          {totalFrames > 0
            ? `Frame ${currentFrameIndex + 1}/${totalFrames}`
            : "Ready"}
        </span>
        {matchResult && (
          <span className="text-xs ml-auto font-mono">
            Result:{" "}
            <span
              className={
                matchResult.winner === 0
                  ? "text-[#c4a050]"
                  : matchResult.winner === 1
                  ? "text-[#b83030]"
                  : "text-[#8b7a58]"
              }
            >
              {matchResult.winner === null
                ? "Draw"
                : `Pilot ${matchResult.winner} wins`}
            </span>
            {matchResult.reason && (
              <span className="text-[#6b5a40] ml-2">({matchResult.reason})</span>
            )}
          </span>
        )}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar */}
        <div className="flex-shrink-0 w-64 bg-[#1c1a14]/90 border-r border-[#3a3526] overflow-y-auto z-10">
          <MatchSetup onStartMatch={handleStartMatch} isConnected={isConnected} />
        </div>

        {/* Scene */}
        <div className="flex-1 relative">
          <Scene
            frame={currentFrame}
            cameraMode={cameraMode}
            trailFrames={frames.slice(
              Math.max(0, currentFrameIndex - 60),
              currentFrameIndex + 1
            )}
          />
          <DebugOverlay
            frame={currentFrame}
            prevFrame={prevFrame}
            frameIndex={currentFrameIndex}
            totalFrames={totalFrames}
          />
        </div>
      </div>

      {/* Bottom controls */}
      <div className="flex-shrink-0 z-10">
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
    </div>
  );
}
