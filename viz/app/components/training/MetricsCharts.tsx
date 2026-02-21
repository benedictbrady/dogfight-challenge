"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { TrainingMetrics, DataPoint } from "../../lib/training-types";

const MAX_POINTS = 500;

/**
 * Largest Triangle Three Buckets (LTTB) downsampling.
 * Reduces a sorted array of {step, value} points to ~targetBuckets points
 * while preserving the visual shape of the data.
 */
function lttbDownsample(
  points: DataPoint[],
  targetBuckets: number = MAX_POINTS
): DataPoint[] {
  if (points.length <= targetBuckets) return points;

  const out: DataPoint[] = [points[0]]; // Always keep first
  const bucketSize = (points.length - 2) / (targetBuckets - 2);

  let prevSelected = 0;

  for (let i = 0; i < targetBuckets - 2; i++) {
    // Current bucket range
    const bucketStart = Math.floor((i + 0) * bucketSize) + 1;
    const bucketEnd = Math.min(
      Math.floor((i + 1) * bucketSize) + 1,
      points.length - 1
    );

    // Next bucket range (for computing average)
    const nextStart = Math.floor((i + 1) * bucketSize) + 1;
    const nextEnd = Math.min(
      Math.floor((i + 2) * bucketSize) + 1,
      points.length - 1
    );

    // Average of next bucket
    let avgStep = 0;
    let avgValue = 0;
    const nextCount = nextEnd - nextStart;
    if (nextCount > 0) {
      for (let j = nextStart; j < nextEnd; j++) {
        avgStep += points[j].step;
        avgValue += points[j].value;
      }
      avgStep /= nextCount;
      avgValue /= nextCount;
    } else {
      // Last bucket — use the last point
      avgStep = points[points.length - 1].step;
      avgValue = points[points.length - 1].value;
    }

    // Pick point in current bucket forming largest triangle
    let maxArea = -1;
    let bestIdx = bucketStart;
    const prevPoint = points[prevSelected];

    for (let j = bucketStart; j < bucketEnd; j++) {
      const area = Math.abs(
        (prevPoint.step - avgStep) * (points[j].value - prevPoint.value) -
          (prevPoint.step - points[j].step) * (avgValue - prevPoint.value)
      );
      if (area > maxArea) {
        maxArea = area;
        bestIdx = j;
      }
    }

    out.push(points[bestIdx]);
    prevSelected = bestIdx;
  }

  out.push(points[points.length - 1]); // Always keep last
  return out;
}

const COLORS = {
  blue: "#5b9bf5",
  red: "#f87171",
  green: "#4ade80",
  orange: "#fb923c",
};

function mergeByStep(
  series: Record<string, DataPoint[]>
): Record<string, number>[] {
  const map = new Map<number, Record<string, number>>();
  for (const [name, points] of Object.entries(series)) {
    for (const pt of points) {
      let row = map.get(pt.step);
      if (!row) {
        row = { step: pt.step };
        map.set(pt.step, row);
      }
      row[name] = pt.value;
    }
  }
  const merged = Array.from(map.values());
  merged.sort((a, b) => a.step - b.step);
  return merged;
}

interface ChartCardProps {
  title: string;
  description: string;
  children: React.ReactNode;
}

function ChartCard({ title, description, children }: ChartCardProps) {
  return (
    <div
      className="rounded-lg p-4"
      style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border)",
      }}
    >
      <h3
        className="text-[11px] font-semibold uppercase tracking-wider"
        style={{ color: "var(--text-secondary)" }}
      >
        {title}
      </h3>
      <p
        className="text-[10px] mt-0.5 mb-3"
        style={{ color: "var(--text-secondary)", opacity: 0.6 }}
      >
        {description}
      </p>
      <div style={{ height: 260 }}>{children}</div>
    </div>
  );
}

interface ChartDef {
  title: string;
  description: string;
  series: {
    key: string;
    source: "losses" | "charts" | "selfplay" | "eval";
    color: string;
  }[];
}

const CHART_DEFS: ChartDef[] = [
  {
    title: "ELO Progression",
    description: "Learner vs opponent pool skill rating. Higher = stronger agent.",
    series: [
      { key: "learner_elo", source: "selfplay", color: COLORS.blue },
      { key: "opponent_elo", source: "selfplay", color: COLORS.red },
    ],
  },
  {
    title: "Win Rate",
    description: "Fraction of episodes won against current opponents.",
    series: [
      { key: "win_rate", source: "charts", color: COLORS.blue },
      { key: "brawler_win_rate", source: "eval", color: COLORS.green },
    ],
  },
  {
    title: "Episode Return",
    description: "Mean cumulative reward per episode. Tracks overall learning signal.",
    series: [
      { key: "ep_return_mean", source: "charts", color: COLORS.blue },
    ],
  },
  {
    title: "Training Losses",
    description: "PPO policy and value function losses. Should decrease then stabilize.",
    series: [
      { key: "policy_loss", source: "losses", color: COLORS.blue },
      { key: "value_loss", source: "losses", color: COLORS.red },
    ],
  },
  {
    title: "Entropy & KL",
    description: "Action entropy (exploration) and KL divergence (policy change per update).",
    series: [
      { key: "entropy", source: "losses", color: COLORS.blue },
      { key: "approx_kl", source: "losses", color: COLORS.red },
    ],
  },
  {
    title: "Exploration",
    description: "Learned log-std for yaw and throttle. Lower = more deterministic policy.",
    series: [
      { key: "log_std_0", source: "charts", color: COLORS.blue },
      { key: "log_std_1", source: "charts", color: COLORS.red },
    ],
  },
];

const tooltipStyle = {
  backgroundColor: "#1a1a1a",
  border: "1px solid #232323",
  borderRadius: 6,
  fontSize: 11,
  color: "#e5e5e5",
};

const axisStyle = { fontSize: 10, fill: "#6b6b6b" };

function fmt(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 10) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(2);
  if (abs >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}

function formatStep(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}k`;
  return String(v);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function tooltipFormatter(value: any) {
  return fmt(Number(value));
}

export default function MetricsCharts({
  metrics,
}: {
  metrics: TrainingMetrics | null;
}) {
  if (!metrics) return null;

  // Build only charts that have data
  const charts: {
    def: ChartDef;
    seriesMap: Record<string, DataPoint[]>;
    data: Record<string, number>[];
  }[] = [];

  for (const def of CHART_DEFS) {
    const seriesMap: Record<string, DataPoint[]> = {};
    for (const s of def.series) {
      const sourceData = metrics[s.source];
      const points = sourceData?.[s.key];
      if (points && points.length > 0) {
        seriesMap[s.key] = points;
      }
    }
    // Only include charts where at least one series has data
    if (Object.keys(seriesMap).length > 0) {
      // Downsample each series before merging to reduce chart rendering load
      const downsampled: Record<string, DataPoint[]> = {};
      for (const [key, pts] of Object.entries(seriesMap)) {
        downsampled[key] = lttbDownsample(pts);
      }
      charts.push({ def, seriesMap, data: mergeByStep(downsampled) });
    }
  }

  if (charts.length === 0) return null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {charts.map(({ def, seriesMap, data }) => (
        <ChartCard key={def.title} title={def.title} description={def.description}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid stroke="#1e1e1e" strokeDasharray="none" />
              <XAxis
                dataKey="step"
                tick={axisStyle}
                tickLine={false}
                axisLine={{ stroke: "#232323" }}
                tickFormatter={formatStep}
              />
              <YAxis
                tick={axisStyle}
                tickLine={false}
                axisLine={{ stroke: "#232323" }}
                width={52}
                tickFormatter={fmt}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={tooltipFormatter}
                labelFormatter={(label) => `Step ${formatStep(Number(label))}`}
              />
              <Legend
                wrapperStyle={{ fontSize: 10, color: "#6b6b6b" }}
                iconType="plainline"
                iconSize={10}
              />
              {def.series.map((s) =>
                seriesMap[s.key] ? (
                  <Line
                    key={s.key}
                    type="monotone"
                    dataKey={s.key}
                    stroke={s.color}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                ) : null
              )}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      ))}
    </div>
  );
}
