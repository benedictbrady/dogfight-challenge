"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { PoolData, PoolEntry } from "../../lib/training-types";

const tooltipStyle = {
  backgroundColor: "#1a1a1a",
  border: "1px solid #232323",
  borderRadius: 6,
  fontSize: 11,
  color: "#e5e5e5",
};

const axisStyle = { fontSize: 10, fill: "#6b6b6b" };

/** Format a generation number nicely. */
function fmtGen(gen: number): string {
  if (gen >= 1000) return `${(gen / 1000).toFixed(1)}k`;
  return String(gen);
}

type Phase = "curriculum" | "transition" | "self-play" | "other";

const PHASE_COLORS: Record<Phase, string> = {
  curriculum: "#facc15",   // yellow
  transition: "#fb923c",   // orange
  "self-play": "#5b9bf5",  // blue (accent)
  other: "#6b6b6b",
};

function entryPhase(name: string): Phase {
  if (name.startsWith("sp_")) return "self-play";
  if (name.startsWith("trans_")) return "transition";
  if (name.startsWith("curriculum")) return "curriculum";
  return "other";
}

/** Derive a readable label from the raw pool entry name. */
function entryLabel(entry: PoolEntry): string {
  const sp = entry.name.match(/^sp_(\d+)$/);
  if (sp) return fmtGen(parseInt(sp[1], 10));
  const tr = entry.name.match(/^trans_(\d+)$/);
  if (tr) return fmtGen(parseInt(tr[1], 10));
  if (entry.name === "curriculum_final") return "final";
  const s = entry.name.match(/^scripted_(.+)$/);
  if (s) return s[1];
  return entry.name;
}

function EloBar({ elo, maxElo }: { elo: number; maxElo: number }) {
  const pct = maxElo > 0 ? Math.max((elo / maxElo) * 100, 2) : 0;
  return (
    <div
      className="h-1.5 rounded-full"
      style={{ background: "rgba(255,255,255,0.04)", width: "100%" }}
    >
      <div
        className="h-full rounded-full transition-all"
        style={{
          width: `${pct}%`,
          background: elo >= 1200 ? "var(--green)" : elo >= 1000 ? "var(--accent)" : "var(--text-secondary)",
        }}
      />
    </div>
  );
}

export default function PoolViewer({ pool }: { pool: PoolData | null }) {
  if (!pool || !pool.entries || pool.entries.length === 0) {
    return null;
  }

  const sorted = useMemo(
    () => [...pool.entries].sort((a, b) => b.elo - a.elo),
    [pool.entries]
  );
  const maxElo = sorted.length > 0 ? sorted[0].elo : 1000;

  // Build ELO-over-generation chart data (entries sorted by generation)
  const eloByGen = useMemo(() => {
    const byGen = [...pool.entries].sort((a, b) => a.generation - b.generation);
    return byGen.map((e) => ({
      gen: e.generation,
      elo: Math.round(e.elo),
      label: entryLabel(e),
      phase: entryPhase(e.name),
    }));
  }, [pool.entries]);

  const topEntry = sorted[0];
  const bottomEntry = sorted[sorted.length - 1];
  const totalGames = sorted.reduce((sum, e) => sum + e.games, 0);

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
    >
      {/* Header */}
      <div className="px-4 pt-4 pb-1">
        <div className="flex items-center justify-between mb-1">
          <h3
            className="text-[11px] font-semibold uppercase tracking-wider"
            style={{ color: "var(--text-secondary)" }}
          >
            Self-Play Opponent Pool
          </h3>
          <span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>
            {sorted.length} snapshots
          </span>
        </div>
        <p className="text-[10px] mb-3" style={{ color: "var(--text-secondary)", opacity: 0.6, lineHeight: 1.5 }}>
          Snapshots of the agent saved at different points during training. The current learner
          plays against these past selves to stay robust. ELO measures relative strength — all
          start at 1000 and diverge as they play each other.
        </p>

        {/* Phase legend */}
        <div
          className="rounded-md px-3 py-2.5 mb-3"
          style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)" }}
        >
          <div className="text-[9px] uppercase tracking-wider mb-2" style={{ color: "var(--text-secondary)" }}>
            Training Phases
          </div>
          <div className="flex flex-col gap-1.5">
            <div className="flex items-start gap-2">
              <span
                className="inline-block rounded-sm flex-shrink-0 mt-0.5"
                style={{ width: 8, height: 8, background: PHASE_COLORS.curriculum }}
              />
              <span className="text-[10px] leading-snug" style={{ color: "var(--text-secondary)" }}>
                <strong style={{ color: "var(--text-primary)" }}>Curriculum</strong> — Learns basics against scripted bots (do_nothing, dogfighter, chaser, ace, brawler) in order of difficulty
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span
                className="inline-block rounded-sm flex-shrink-0 mt-0.5"
                style={{ width: 8, height: 8, background: PHASE_COLORS.transition }}
              />
              <span className="text-[10px] leading-snug" style={{ color: "var(--text-secondary)" }}>
                <strong style={{ color: "var(--text-primary)" }}>Transition</strong> — Mix of scripted bots and past selves, easing into self-play
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span
                className="inline-block rounded-sm flex-shrink-0 mt-0.5"
                style={{ width: 8, height: 8, background: PHASE_COLORS["self-play"] }}
              />
              <span className="text-[10px] leading-snug" style={{ color: "var(--text-secondary)" }}>
                <strong style={{ color: "var(--text-primary)" }}>Self-Play</strong> — Trains exclusively against its own past snapshots, discovering new strategies
              </span>
            </div>
          </div>
        </div>

        {/* Summary stats */}
        <div className="flex items-center gap-5 mb-3">
          <div>
            <div className="text-[9px] uppercase tracking-wider" style={{ color: "var(--text-secondary)" }}>
              Strongest
            </div>
            <div className="text-xs font-semibold" style={{ color: "var(--accent)", fontFamily: "var(--font-mono, monospace)" }}>
              {Math.round(topEntry.elo)} ELO
            </div>
          </div>
          <div>
            <div className="text-[9px] uppercase tracking-wider" style={{ color: "var(--text-secondary)" }}>
              Weakest
            </div>
            <div className="text-xs font-semibold" style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono, monospace)" }}>
              {Math.round(bottomEntry.elo)} ELO
            </div>
          </div>
          <div>
            <div className="text-[9px] uppercase tracking-wider" style={{ color: "var(--text-secondary)" }}>
              Spread
            </div>
            <div className="text-xs font-semibold" style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono, monospace)" }}>
              {Math.round(topEntry.elo - bottomEntry.elo)}
            </div>
          </div>
          <div>
            <div className="text-[9px] uppercase tracking-wider" style={{ color: "var(--text-secondary)" }}>
              Total Games
            </div>
            <div className="text-xs font-semibold" style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono, monospace)" }}>
              {totalGames.toLocaleString()}
            </div>
          </div>
        </div>
      </div>

      {/* ELO over generation chart */}
      {eloByGen.length > 2 && (
        <div className="px-4 pb-3" style={{ height: 160 }}>
          <div className="text-[9px] uppercase tracking-wider mb-1" style={{ color: "var(--text-secondary)" }}>
            ELO by Generation
          </div>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={eloByGen} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
              <CartesianGrid stroke="#1e1e1e" strokeDasharray="none" />
              <XAxis
                dataKey="gen"
                tick={axisStyle}
                tickLine={false}
                axisLine={{ stroke: "#232323" }}
                tickFormatter={(v) => fmtGen(v)}
              />
              <YAxis
                tick={axisStyle}
                tickLine={false}
                axisLine={{ stroke: "#232323" }}
                width={42}
                domain={["dataMin - 50", "dataMax + 50"]}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                labelFormatter={(_label, payload) => {
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  const pt = payload?.[0]?.payload as any;
                  if (!pt) return "";
                  const phase = pt.phase as Phase;
                  const phaseLabel = phase === "self-play" ? "Self-Play" : phase === "transition" ? "Transition" : phase === "curriculum" ? "Curriculum" : "";
                  return `${phaseLabel} @ update ${pt.gen}`;
                }}
                formatter={(value: number) => [Math.round(value), "ELO"]}
              />
              <Line
                type="monotone"
                dataKey="elo"
                stroke="var(--accent)"
                strokeWidth={1.5}
                dot={(props) => {
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  const { cx, cy, payload } = props as any;
                  const color = PHASE_COLORS[(payload?.phase as Phase) ?? "other"];
                  return <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r={3} fill={color} stroke="none" />;
                }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Ranked table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <th className="py-2 px-3 text-left font-semibold" style={{ color: "var(--text-secondary)" }}>#</th>
              <th className="py-2 px-3 text-left font-semibold" style={{ color: "var(--text-secondary)" }}>Snapshot</th>
              <th className="py-2 px-3 text-right font-semibold" style={{ color: "var(--text-secondary)", width: 60 }}>ELO</th>
              <th className="py-2 px-3 font-semibold" style={{ color: "var(--text-secondary)", width: "30%" }}></th>
              <th className="py-2 px-3 text-right font-semibold" style={{ color: "var(--text-secondary)" }}>W / L / D</th>
              <th className="py-2 px-3 text-right font-semibold" style={{ color: "var(--text-secondary)" }}>Win%</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((entry, i) => {
              const winPct =
                entry.games > 0
                  ? ((entry.wins / entry.games) * 100).toFixed(0)
                  : "—";
              return (
                <tr
                  key={entry.name}
                  style={{
                    borderBottom: "1px solid var(--border)",
                    background: i % 2 === 1 ? "rgba(255,255,255,0.015)" : "transparent",
                  }}
                >
                  <td className="py-1.5 px-3" style={{ color: "var(--text-secondary)" }}>
                    {i + 1}
                  </td>
                  <td className="py-1.5 px-3">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="inline-block rounded-sm flex-shrink-0"
                        style={{ width: 6, height: 6, background: PHASE_COLORS[entryPhase(entry.name)] }}
                      />
                      <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono, monospace)" }}>
                        {entryLabel(entry)}
                      </span>
                    </div>
                  </td>
                  <td
                    className="py-1.5 px-3 text-right font-semibold"
                    style={{
                      color: entry.elo >= 1200 ? "var(--green)" : entry.elo >= 1000 ? "var(--accent)" : "var(--text-primary)",
                      fontFamily: "var(--font-mono, monospace)",
                    }}
                  >
                    {Math.round(entry.elo)}
                  </td>
                  <td className="py-1.5 px-3">
                    <EloBar elo={entry.elo} maxElo={maxElo} />
                  </td>
                  <td className="py-1.5 px-3 text-right whitespace-nowrap" style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono, monospace)" }}>
                    {entry.wins} / {entry.losses} / {entry.draws}
                  </td>
                  <td className="py-1.5 px-3 text-right" style={{ color: "var(--text-secondary)" }}>
                    {winPct}{winPct !== "—" ? "%" : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
