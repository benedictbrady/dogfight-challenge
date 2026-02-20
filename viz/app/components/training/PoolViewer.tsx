"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { PoolData } from "../../lib/training-types";

const tooltipStyle = {
  backgroundColor: "#1a1a1a",
  border: "1px solid #232323",
  borderRadius: 6,
  fontSize: 11,
  color: "#e5e5e5",
};

const axisStyle = { fontSize: 10, fill: "#6b6b6b" };

export default function PoolViewer({ pool }: { pool: PoolData | null }) {
  if (!pool || !pool.entries || pool.entries.length === 0) {
    return null;
  }

  const sorted = [...pool.entries].sort((a, b) => b.elo - a.elo);

  const barData = sorted.map((e) => ({
    name: e.name.length > 14 ? e.name.slice(0, 12) + ".." : e.name,
    elo: Math.round(e.elo),
  }));

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
    >
      <div className="px-4 pt-4 pb-2 flex items-center justify-between">
        <h3
          className="text-[11px] font-semibold uppercase tracking-wider"
          style={{ color: "var(--text-secondary)" }}
        >
          Opponent Pool
        </h3>
        <span className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
          {sorted.length} entries
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["#", "Name", "ELO", "Games", "Win%", "Gen"].map((h) => (
                <th
                  key={h}
                  className={`py-2 px-3 font-semibold ${h === "#" || h === "Name" ? "text-left" : "text-right"}`}
                  style={{ color: "var(--text-secondary)" }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((entry, i) => {
              const winPct =
                entry.games > 0
                  ? ((entry.wins / entry.games) * 100).toFixed(1)
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
                  <td
                    className="py-1.5 px-3"
                    style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono, monospace)" }}
                  >
                    {entry.name}
                  </td>
                  <td
                    className="py-1.5 px-3 text-right font-semibold"
                    style={{
                      color: entry.elo >= 1000 ? "var(--accent)" : "var(--text-primary)",
                      fontFamily: "var(--font-mono, monospace)",
                    }}
                  >
                    {Math.round(entry.elo)}
                  </td>
                  <td className="py-1.5 px-3 text-right" style={{ color: "var(--text-secondary)" }}>
                    {entry.games.toLocaleString()}
                  </td>
                  <td className="py-1.5 px-3 text-right" style={{ color: "var(--text-secondary)" }}>
                    {winPct}{winPct !== "—" ? "%" : ""}
                  </td>
                  <td className="py-1.5 px-3 text-right" style={{ color: "var(--text-secondary)" }}>
                    {entry.generation}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ELO bar chart */}
      {barData.length > 0 && (
        <div className="px-4 pb-4 pt-2" style={{ height: 200 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} layout="vertical" margin={{ left: 0, right: 16 }}>
              <CartesianGrid stroke="#1e1e1e" strokeDasharray="none" horizontal={false} />
              <XAxis type="number" tick={axisStyle} tickLine={false} axisLine={{ stroke: "#232323" }} />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ ...axisStyle, fontFamily: "var(--font-mono, monospace)" }}
                tickLine={false}
                axisLine={{ stroke: "#232323" }}
                width={90}
              />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
              <Bar dataKey="elo" fill="var(--accent)" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
