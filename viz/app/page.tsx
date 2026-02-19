"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Scene from "./components/Scene";
import Controls from "./components/Controls";
import MatchSetup from "./components/MatchSetup";
import { useMatch, MatchConfig, Frame, ManualAction } from "./hooks/useMatch";

const CRUISE_THROTTLE = 0.82;
const BOOST_THROTTLE = 1.0;
const BRAKE_THROTTLE = 0.4;
const TURN_SMOOTHING_TIME = 0.1; // seconds
const MANUAL_CONTROL_INTERVAL_MS = 16;
const ACTION_EPSILON = 0.01;
const STALL_WARNING_SPEED = 38;

const DEFAULT_FRAME: Frame = {
  tick: 0,
  fighters: [
    { x: -200, y: 300, yaw: 0, speed: 40, hp: 5, alive: true },
    { x: 200, y: 300, yaw: Math.PI, speed: 40, hp: 5, alive: true },
  ],
  bullets: [],
  hits: [],
};

interface ManualHudState {
  yawInput: number;
  targetYaw: number;
  throttle: number;
  shoot: boolean;
  boosting: boolean;
  braking: boolean;
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
    manualPlayers,
    isLiveMode,
    startMatch,
    sendManualInput,
  } = useMatch();

  const [cameraMode] = useState<"free" | "chase0" | "chase1">("free");
  const [spawnFrame, setSpawnFrame] = useState<Frame>(DEFAULT_FRAME);
  const [manualHud, setManualHud] = useState<ManualHudState | null>(null);
  const lastSentActionRef = useRef<ManualAction | null>(null);

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
  const totalFrames = frames.length;

  const handleStartMatch = (config: MatchConfig) => {
    startMatch(config);
  };

  const activeManualPlayer: 0 | 1 | null = manualPlayers.p0
    ? 0
    : manualPlayers.p1
    ? 1
    : null;
  const manualFighterState =
    activeManualPlayer !== null ? currentFrame.fighters[activeManualPlayer] : null;
  const isStallWarning =
    manualFighterState !== null &&
    manualFighterState.alive &&
    manualFighterState.speed < STALL_WARNING_SPEED;

  useEffect(() => {
    if (activeManualPlayer === null || !isLiveMode) {
      lastSentActionRef.current = null;
      setManualHud(null);
      return;
    }

    const pressed = new Set<string>();
    const relevantCodes = new Set([
      "KeyA",
      "KeyD",
      "KeyW",
      "KeyS",
      "ArrowLeft",
      "ArrowRight",
      "ArrowUp",
      "ArrowDown",
      "Space",
    ]);

    const isTypingTarget = (target: EventTarget | null): boolean =>
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      (target instanceof HTMLElement && target.isContentEditable);

    const hasTurnLeft = () => pressed.has("ArrowLeft") || pressed.has("KeyA");
    const hasTurnRight = () => pressed.has("ArrowRight") || pressed.has("KeyD");
    const hasBoost = () => pressed.has("ArrowUp") || pressed.has("KeyW");
    const hasBrake = () => pressed.has("ArrowDown") || pressed.has("KeyS");

    const getTargetYaw = (): number => {
      const left = hasTurnLeft();
      const right = hasTurnRight();
      if (left && !right) return 1;
      if (right && !left) return -1;
      return 0;
    };

    const getThrottle = (): number => {
      const boosting = hasBoost();
      const braking = hasBrake();
      if (boosting && !braking) return BOOST_THROTTLE;
      if (braking && !boosting) return BRAKE_THROTTLE;
      return CRUISE_THROTTLE;
    };

    const shouldSend = (prev: ManualAction | null, next: ManualAction): boolean => {
      if (!prev) return true;
      return (
        Math.abs(prev.yaw_input - next.yaw_input) > ACTION_EPSILON ||
        Math.abs(prev.throttle - next.throttle) > ACTION_EPSILON ||
        prev.shoot !== next.shoot
      );
    };

    let yawInput = 0;
    let lastTickMs = performance.now();
    let lastHudUpdateMs = 0;

    const tick = () => {
      if (!isConnected) return;

      const now = performance.now();
      const dt = Math.max(0.001, (now - lastTickMs) / 1000);
      lastTickMs = now;

      const targetYaw = getTargetYaw();
      const smoothingAlpha = 1 - Math.exp(-dt / TURN_SMOOTHING_TIME);
      yawInput += (targetYaw - yawInput) * smoothingAlpha;
      if (targetYaw === 0 && Math.abs(yawInput) < 0.005) {
        yawInput = 0;
      }

      const throttle = getThrottle();
      const shoot = pressed.has("Space");

      const action: ManualAction = {
        yaw_input: Number(yawInput.toFixed(3)),
        throttle: Number(throttle.toFixed(3)),
        shoot,
      };

      if (shouldSend(lastSentActionRef.current, action)) {
        sendManualInput(activeManualPlayer, action);
        lastSentActionRef.current = action;
      }

      if (now - lastHudUpdateMs >= 80) {
        setManualHud({
          yawInput: action.yaw_input,
          targetYaw,
          throttle: action.throttle,
          shoot,
          boosting: hasBoost(),
          braking: hasBrake(),
        });
        lastHudUpdateMs = now;
      }
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (isTypingTarget(event.target) || !relevantCodes.has(event.code)) {
        return;
      }
      event.preventDefault();
      pressed.add(event.code);
    };

    const onKeyUp = (event: KeyboardEvent) => {
      if (!relevantCodes.has(event.code)) {
        return;
      }
      event.preventDefault();
      pressed.delete(event.code);
    };

    const onWindowBlur = () => {
      pressed.clear();
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onWindowBlur);
    lastSentActionRef.current = null;
    lastTickMs = performance.now();
    const intervalId = window.setInterval(tick, MANUAL_CONTROL_INTERVAL_MS);
    tick();

    return () => {
      window.clearInterval(intervalId);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onWindowBlur);
      if (isConnected) {
        sendManualInput(activeManualPlayer, {
          yaw_input: 0,
          throttle: CRUISE_THROTTLE,
          shoot: false,
        });
      }
      setManualHud(null);
      lastSentActionRef.current = null;
    };
  }, [activeManualPlayer, isConnected, isLiveMode, sendManualInput]);

  return (
    <div className="w-screen h-screen flex flex-col bg-[#f8f8f8]">
      {/* Top bar */}
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
              style={{ width: `${(currentFrame.fighters[0].hp / 5) * 100}%` }}
            />
          </div>
        </div>

        {/* P1 health bar */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-red-600">P1</span>
          <div className="w-24 h-3 bg-gray-100 rounded-sm border border-gray-200 overflow-hidden">
            <div
              className="h-full bg-red-500 transition-all duration-200"
              style={{ width: `${(currentFrame.fighters[1].hp / 5) * 100}%` }}
            />
          </div>
        </div>

        <span className="text-xs text-gray-500">
          {totalFrames > 0
            ? `Frame ${currentFrameIndex + 1}/${totalFrames}`
            : "Ready"}
        </span>
        {activeManualPlayer !== null && (
          <span className="text-xs font-semibold text-emerald-600">
            Manual: P{activeManualPlayer}
          </span>
        )}
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
        {/* Left sidebar */}
        <div className="flex-shrink-0 w-64 bg-white/95 backdrop-blur-sm border-r border-gray-200 overflow-y-auto z-10">
          <MatchSetup
            onStartMatch={handleStartMatch}
            onSeedChange={fetchSpawn}
            isConnected={isConnected}
          />
        </div>

        {/* Scene */}
        <div className="flex-1 relative">
          {isLiveMode && activeManualPlayer !== null && (
            <>
              <div className="absolute left-3 top-3 z-20 bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs px-2 py-1 rounded">
                Live Manual P{activeManualPlayer}: WASD or Arrows to fly, Space to fire
              </div>
              {manualHud && (
                <div className="absolute left-3 top-12 z-20 w-64 bg-white/95 border border-gray-200 rounded shadow-sm p-2 text-xs text-gray-700">
                  <div className="font-semibold text-gray-800 mb-1">Manual Flight HUD</div>
                  <div className="grid grid-cols-[72px,1fr] gap-y-1 items-center">
                    <span className="text-gray-500">Turn</span>
                    <div className="flex items-center gap-2">
                      <div className="h-2 flex-1 bg-gray-100 rounded overflow-hidden">
                        <div
                          className="h-full bg-indigo-500"
                          style={{
                            width: `${Math.abs(manualHud.yawInput) * 50}%`,
                            marginLeft: manualHud.yawInput >= 0 ? "50%" : `${50 - Math.abs(manualHud.yawInput) * 50}%`,
                          }}
                        />
                      </div>
                      <span className="font-mono w-10 text-right">{manualHud.yawInput.toFixed(2)}</span>
                    </div>

                    <span className="text-gray-500">Throttle</span>
                    <div className="flex items-center gap-2">
                      <div className="h-2 flex-1 bg-gray-100 rounded overflow-hidden">
                        <div
                          className={`h-full ${manualHud.braking ? "bg-amber-500" : manualHud.boosting ? "bg-emerald-500" : "bg-sky-500"}`}
                          style={{ width: `${Math.round(manualHud.throttle * 100)}%` }}
                        />
                      </div>
                      <span className="font-mono w-10 text-right">{Math.round(manualHud.throttle * 100)}%</span>
                    </div>

                    <span className="text-gray-500">Speed</span>
                    <span className={`font-mono ${isStallWarning ? "text-amber-600 font-semibold" : ""}`}>
                      {manualFighterState ? `${manualFighterState.speed.toFixed(1)}` : "--"}
                    </span>

                    <span className="text-gray-500">Fire</span>
                    <span className={manualHud.shoot ? "text-red-600 font-semibold" : "text-gray-500"}>
                      {manualHud.shoot ? "FIRING" : "ready"}
                    </span>
                  </div>
                  <div className="mt-2 pt-1 border-t border-gray-100 text-[11px] text-gray-500">
                    <span>W/Up: boost</span>
                    <span className="mx-2">S/Down: brake</span>
                    <span>A,D/Left,Right: steer</span>
                  </div>
                  {isStallWarning && (
                    <div className="mt-1 text-[11px] text-amber-700 font-semibold">
                      Stall risk: add throttle and reduce climb angle.
                    </div>
                  )}
                </div>
              )}
            </>
          )}
          <Scene
            frame={currentFrame}
            cameraMode={cameraMode}
            trailFrames={frames.slice(0, currentFrameIndex + 1)}
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
          isLive={isLiveMode}
          onSetFrame={setCurrentFrameIndex}
          onTogglePlay={() => setIsPlaying(!isPlaying)}
          onSetSpeed={setSpeed}
        />
      </div>
    </div>
  );
}
