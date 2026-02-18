"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Frame, FighterState } from "../hooks/useMatch";

interface DebugOverlayProps {
  frame: Frame;
  prevFrame: Frame | null;
  frameIndex: number;
  totalFrames: number;
}

const TELEPORT_THRESHOLD = 12;

interface TeleportEvent {
  frameIndex: number;
  player: number;
  from: { x: number; y: number };
  to: { x: number; y: number };
  delta: number;
}

function fmtNum(n: number): string {
  return n.toFixed(1);
}

function fmtYaw(rad: number): string {
  return `${((rad * 180) / Math.PI).toFixed(0)}°`;
}

function FighterRow({
  label,
  state,
  color,
}: {
  label: string;
  state: FighterState;
  color: string;
}) {
  return (
    <tr className="border-b border-white/10">
      <td className="pr-3 font-bold" style={{ color }}>
        {label}
      </td>
      <td className="pr-2 text-right">{fmtNum(state.x)}</td>
      <td className="pr-2 text-right">{fmtNum(state.y)}</td>
      <td className="pr-2 text-right">{fmtYaw(state.yaw)}</td>
      <td className="pr-2 text-right">{fmtNum(state.speed)}</td>
      <td className="pr-2 text-right">{state.hp}</td>
      <td className="text-right">{state.alive ? "Y" : "N"}</td>
    </tr>
  );
}

export default function DebugOverlay({
  frame,
  prevFrame,
  frameIndex,
  totalFrames,
}: DebugOverlayProps) {
  const [visible, setVisible] = useState(false);
  const teleportLog = useRef<TeleportEvent[]>([]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "d" || e.key === "D") {
        if (
          e.target instanceof HTMLInputElement ||
          e.target instanceof HTMLTextAreaElement
        )
          return;
        setVisible((v) => !v);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const teleports = useMemo(() => {
    if (!prevFrame) return [];
    const events: TeleportEvent[] = [];
    for (let p = 0; p < 2; p++) {
      const prev = prevFrame.fighters[p];
      const curr = frame.fighters[p];
      if (!prev || !curr) continue;
      const dx = curr.x - prev.x;
      const dy = curr.y - prev.y;
      const delta = Math.sqrt(dx * dx + dy * dy);
      if (delta > TELEPORT_THRESHOLD) {
        const event: TeleportEvent = {
          frameIndex,
          player: p,
          from: { x: prev.x, y: prev.y },
          to: { x: curr.x, y: curr.y },
          delta,
        };
        events.push(event);
        console.warn(
          `[TELEPORT] P${p} jumped ${delta.toFixed(1)} units: ` +
            `(${fmtNum(prev.x)}, ${fmtNum(prev.y)}) → (${fmtNum(curr.x)}, ${fmtNum(curr.y)}) ` +
            `at frame ${frameIndex} (tick ${frame.tick})`
        );
      }
    }
    return events;
  }, [frame, prevFrame, frameIndex]);

  useEffect(() => {
    if (teleports.length > 0) {
      teleportLog.current = [
        ...teleportLog.current.slice(-19),
        ...teleports,
      ];
    }
  }, [teleports]);

  useEffect(() => {
    if (frameIndex === 0 && totalFrames > 0) {
      console.log(
        "[DEBUG] Frame 0 fighters:",
        JSON.stringify(frame.fighters, null, 2)
      );
    }
  }, [frameIndex, totalFrames, frame.fighters]);

  if (!visible) return null;

  return (
    <div className="absolute top-12 right-2 z-50 bg-black/80 text-white text-[10px] font-mono p-2 rounded border border-white/20 pointer-events-none select-none min-w-[340px]">
      <div className="text-[11px] font-bold mb-1 text-yellow-400">
        DEBUG (press D to hide)
      </div>

      <div className="mb-1 text-gray-400">
        Frame {frameIndex}/{totalFrames} &middot; Tick {frame.tick}
      </div>

      <table className="w-full mb-1">
        <thead>
          <tr className="text-gray-500 border-b border-white/20">
            <th className="text-left pr-3">ID</th>
            <th className="text-right pr-2">X</th>
            <th className="text-right pr-2">Y</th>
            <th className="text-right pr-2">Yaw</th>
            <th className="text-right pr-2">Spd</th>
            <th className="text-right pr-2">HP</th>
            <th className="text-right">Alive</th>
          </tr>
        </thead>
        <tbody>
          <FighterRow
            label="P0"
            state={frame.fighters[0]}
            color="#2563eb"
          />
          <FighterRow
            label="P1"
            state={frame.fighters[1]}
            color="#dc2626"
          />
        </tbody>
      </table>

      {prevFrame && (
        <div className="text-gray-500 mb-1">
          &Delta;P0: (
          {fmtNum(frame.fighters[0].x - prevFrame.fighters[0].x)},{" "}
          {fmtNum(frame.fighters[0].y - prevFrame.fighters[0].y)}) &middot;
          &Delta;P1: (
          {fmtNum(frame.fighters[1].x - prevFrame.fighters[1].x)},{" "}
          {fmtNum(frame.fighters[1].y - prevFrame.fighters[1].y)})
        </div>
      )}

      {teleportLog.current.length > 0 && (
        <div className="mt-1 border-t border-red-500/50 pt-1">
          <div className="text-red-400 font-bold text-[11px]">
            TELEPORT DETECTED ({teleportLog.current.length})
          </div>
          {teleportLog.current.slice(-5).map((t, i) => (
            <div key={i} className="text-red-300">
              F{t.frameIndex} P{t.player}: ({fmtNum(t.from.x)},
              {fmtNum(t.from.y)}) &rarr; ({fmtNum(t.to.x)},{fmtNum(t.to.y)})
              &Delta;{fmtNum(t.delta)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
