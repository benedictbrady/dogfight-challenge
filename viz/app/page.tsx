"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Scene from "./components/Scene";
import Controls from "./components/Controls";
import MatchSetup from "./components/MatchSetup";
import DebugOverlay from "./components/DebugOverlay";
import { useMatch, MatchConfig, Frame, HumanInput } from "./hooks/useMatch";

const MAX_HP = 5;

const DEFAULT_FRAME: Frame = {
  tick: 0,
  fighters: [
    { x: -200, y: 300, yaw: 0, speed: 40, hp: 5, alive: true },
    { x: 200, y: 300, yaw: Math.PI, speed: 40, hp: 5, alive: true },
  ],
  bullets: [],
  hits: [],
};

const CRUISE_THROTTLE = 0.65;

function normalizeControlKey(key: string): string | null {
  const k = key.toLowerCase();
  if (
    k === "a" ||
    k === "d" ||
    k === "w" ||
    k === "s" ||
    k === "arrowleft" ||
    k === "arrowright" ||
    k === "arrowup" ||
    k === "arrowdown"
  ) {
    return k;
  }
  if (k === " " || k === "space" || k === "spacebar") {
    return "space";
  }
  return null;
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement
  );
}

function computeHumanInput(pressed: Set<string>): HumanInput {
  const left = pressed.has("a") || pressed.has("arrowleft");
  const right = pressed.has("d") || pressed.has("arrowright");
  const up = pressed.has("w") || pressed.has("arrowup");
  const down = pressed.has("s") || pressed.has("arrowdown");
  const shoot = pressed.has("space");

  return {
    yaw_input: (right ? 1 : 0) - (left ? 1 : 0),
    throttle: up ? 1 : down ? 0 : CRUISE_THROTTLE,
    shoot,
  };
}

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
    isHumanMatch,
    startMatch,
    sendHumanInput,
  } = useMatch();

  const [spawnFrame, setSpawnFrame] = useState<Frame>(DEFAULT_FRAME);
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const lastInputRef = useRef<string>("");

  // Fetch starting positions for a given seed
  const fetchSpawn = useCallback(async (seed: number) => {
    try {
      const res = await fetch(`http://localhost:3001/api/spawn?seed=${seed}`);
      if (res.ok) {
        const data = await res.json();
        if (data.fighters && data.fighters.length === 2) {
          setSpawnFrame({
            tick: 0,
            fighters: [
              { ...data.fighters[0], alive: true },
              { ...data.fighters[1], alive: true },
            ],
            bullets: [],
            hits: [],
          });
        }
      }
    } catch {
      // Server offline — keep default positions
    }
  }, []);

  const currentFrame = frames[currentFrameIndex] ?? spawnFrame;
  const prevFrame = currentFrameIndex > 0 ? (frames[currentFrameIndex - 1] ?? null) : null;
  const totalFrames = frames.length;

  const handleStartMatch = (config: MatchConfig) => {
    startMatch(config);
  };

  useEffect(() => {
    if (!isConnected || !isHumanMatch) return;

    const sendCurrentInput = () => {
      const input = computeHumanInput(pressedKeysRef.current);
      const signature = JSON.stringify(input);
      if (signature === lastInputRef.current) return;
      lastInputRef.current = signature;
      sendHumanInput(input);
    };

    const resetInput = () => {
      if (pressedKeysRef.current.size === 0) return;
      pressedKeysRef.current.clear();
      sendCurrentInput();
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target)) return;
      const normalized = normalizeControlKey(e.key);
      if (!normalized) return;
      e.preventDefault();

      if (!pressedKeysRef.current.has(normalized)) {
        pressedKeysRef.current.add(normalized);
        sendCurrentInput();
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const normalized = normalizeControlKey(e.key);
      if (!normalized) return;
      e.preventDefault();

      if (pressedKeysRef.current.delete(normalized)) {
        sendCurrentInput();
      }
    };

    const onVisibilityChange = () => {
      if (document.hidden) resetInput();
    };

    sendCurrentInput();
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", resetInput);
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", resetInput);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      resetInput();
      lastInputRef.current = "";
    };
  }, [isConnected, isHumanMatch, sendHumanInput]);

  return (
    <div className="w-screen h-screen flex flex-col bg-[#f8f8f8]">
      <div className="flex-shrink-0 h-10 bg-white border-b border-gray-200 flex items-center px-4 gap-4 z-10">
        <span className="text-sm font-bold text-gray-800 tracking-[0.15em]">
          DOGFIGHT
        </span>

        {/* P0 health bar */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-blue-600">P0</span>
          <div className="w-24 h-3 bg-gray-100 rounded-sm border border-gray-200 overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-200"
              style={{ width: `${(currentFrame.fighters[0].hp / MAX_HP) * 100}%` }}
            />
          </div>
        </div>

        {/* P1 health bar */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-red-600">P1</span>
          <div className="w-24 h-3 bg-gray-100 rounded-sm border border-gray-200 overflow-hidden">
            <div
              className="h-full bg-red-500 transition-all duration-200"
              style={{ width: `${(currentFrame.fighters[1].hp / MAX_HP) * 100}%` }}
            />
          </div>
        </div>

        <span className="text-xs text-gray-500">
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
                  ? "text-blue-600"
                  : matchResult.winner === 1
                  ? "text-red-600"
                  : "text-gray-500"
              }
            >
              {matchResult.winner === null
                ? "Draw"
                : `Player ${matchResult.winner} wins`}
            </span>
            {matchResult.reason && (
              <span className="text-gray-400 ml-2">({matchResult.reason})</span>
            )}
          </span>
        )}
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-shrink-0 w-64 bg-white/95 backdrop-blur-sm border-r border-gray-200 overflow-y-auto z-10">
          <MatchSetup
            onStartMatch={handleStartMatch}
            onSeedChange={fetchSpawn}
            isConnected={isConnected}
          />
        </div>

        <div className="flex-1 relative">
          <Scene
            frame={currentFrame}
            trailFrames={frames.slice(0, currentFrameIndex + 1)}
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
