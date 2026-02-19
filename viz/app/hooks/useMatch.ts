"use client";

import { useState, useEffect, useRef, useCallback } from "react";

// --- Types ---

export interface FighterState {
  x: number;
  y: number;
  yaw: number;
  speed: number;
  hp: number;
  alive: boolean;
}

export interface BulletState {
  x: number;
  y: number;
  vx: number;
  vy: number;
  owner: number;
}

export interface HitEvent {
  target: number;
  x: number;
  y: number;
}

export interface Frame {
  tick: number;
  fighters: [FighterState, FighterState];
  bullets: BulletState[];
  hits: HitEvent[];
}

export interface MatchResult {
  winner: number | null;
  reason: string;
}

export interface MatchConfig {
  p0: string;
  p1: string;
  seed: number;
  randomize_spawns?: boolean;
}

export interface ManualPlayers {
  p0: boolean;
  p1: boolean;
}

export interface ManualAction {
  yaw_input: number;
  throttle: number;
  shoot: boolean;
}

// --- Hook ---

export function useMatch() {
  const [frames, setFrames] = useState<Frame[]>([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [matchResult, setMatchResult] = useState<MatchResult | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [manualPlayers, setManualPlayers] = useState<ManualPlayers>({ p0: false, p1: false });

  const wsRef = useRef<WebSocket | null>(null);
  const playbackRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isLiveMode = (manualPlayers.p0 || manualPlayers.p1) && !isComplete;

  // Playback timer
  useEffect(() => {
    if (playbackRef.current) {
      clearInterval(playbackRef.current);
      playbackRef.current = null;
    }

    if (isPlaying && frames.length > 0 && !isLiveMode) {
      // Base rate: 30fps (one tick per ~33ms at 1x speed)
      const interval = Math.max(8, Math.round(33 / speed));
      playbackRef.current = setInterval(() => {
        setCurrentFrameIndex((prev) => {
          const next = prev + 1;
          if (next >= frames.length) {
            setIsPlaying(false);
            return prev;
          }
          return next;
        });
      }, interval);
    }

    return () => {
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
      }
    };
  }, [isPlaying, speed, frames.length, isLiveMode]);

  const startMatch = useCallback((config: MatchConfig) => {
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Reset state
    setFrames([]);
    setCurrentFrameIndex(0);
    setIsPlaying(false);
    setMatchResult(null);
    setIsComplete(false);
    const nextManualPlayers: ManualPlayers = {
      p0: config.p0 === "manual",
      p1: config.p1 === "manual",
    };
    setManualPlayers(nextManualPlayers);
    const liveMatch = nextManualPlayers.p0 || nextManualPlayers.p1;

    const ws = new WebSocket("ws://localhost:3001/api/match");
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      // Send match configuration
      ws.send(JSON.stringify(config));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "frame") {
          const frame: Frame = {
            tick: msg.tick,
            fighters: msg.fighters,
            bullets: msg.bullets ?? [],
            hits: [],
          };
          if (liveMatch) {
            setFrames((prev) => {
              const next = [...prev, frame];
              setCurrentFrameIndex(next.length - 1);
              return next;
            });
          } else {
            setFrames((prev) => {
              // Auto-start playback once we have a few frames buffered
              if (prev.length === 10) {
                setIsPlaying(true);
              }
              return [...prev, frame];
            });
          }
        } else if (msg.type === "result") {
          const winner =
            msg.outcome === "Player0Win" ? 0 :
            msg.outcome === "Player1Win" ? 1 : null;
          setMatchResult({
            winner,
            reason: msg.reason ?? "",
          });
          setIsComplete(true);
          if (liveMatch) {
            setIsPlaying(false);
          }
        } else if (msg.type === "error") {
          console.error("Match error:", msg.error);
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setIsConnected(false);
    };
  }, []);

  const sendManualInput = useCallback((player: 0 | 1, action: ManualAction) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }

    ws.send(
      JSON.stringify({
        type: "input",
        player,
        action,
      })
    );
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    frames,
    currentFrameIndex,
    setCurrentFrameIndex,
    isPlaying,
    setIsPlaying,
    speed,
    setSpeed,
    matchResult,
    isConnected,
    isComplete,
    manualPlayers,
    isLiveMode,
    startMatch,
    sendManualInput,
  };
}
