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

export interface HumanInput {
  yaw_input: number;
  throttle: number;
  shoot: boolean;
}

const OUTCOME_TO_WINNER: Record<string, number | null> = {
  Player0Win: 0,
  Player1Win: 1,
};

// --- Hook ---

export function useMatch() {
  const [frames, setFrames] = useState<Frame[]>([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [matchResult, setMatchResult] = useState<MatchResult | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [isHumanMatch, setIsHumanMatch] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const playbackRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isHumanMatchRef = useRef(false);

  // Playback timer
  useEffect(() => {
    if (playbackRef.current) {
      clearInterval(playbackRef.current);
      playbackRef.current = null;
    }

    if (isPlaying && frames.length > 0) {
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
  }, [isPlaying, speed, frames.length]);

  const startMatch = useCallback((config: MatchConfig) => {
    const humanMatch = config.p0 === "human" || config.p1 === "human";
    isHumanMatchRef.current = humanMatch;
    setIsHumanMatch(humanMatch);

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
          if (isHumanMatchRef.current) {
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
          const winner = OUTCOME_TO_WINNER[msg.outcome] ?? null;
          setMatchResult({
            winner,
            reason: msg.reason ?? "",
          });
          setIsComplete(true);
          setIsPlaying(false);
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

  const sendHumanInput = useCallback((input: HumanInput) => {
    if (!isHumanMatchRef.current) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(
      JSON.stringify({
        type: "input",
        yaw_input: input.yaw_input,
        throttle: input.throttle,
        shoot: input.shoot,
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
    isHumanMatch,
    startMatch,
    sendHumanInput,
  };
}
