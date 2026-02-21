"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import {
  useTrainingRuns,
  useTrainingMetrics,
  usePoolData,
  useRunConfig,
} from "../hooks/useTrainingData";
import type { RunInfo, RunStatus, TrainingMetrics } from "../lib/training-types";
import MetricsCharts from "../components/training/MetricsCharts";
import PoolViewer from "../components/training/PoolViewer";

const CheckpointMatchViewer = dynamic(
  () => import("../components/training/CheckpointMatchViewer"),
  { ssr: false }
);

function StatusDot({ status, size = 6 }: { status: RunStatus; size?: number }) {
  const color =
    status === "live"
      ? "var(--green)"
      : status === "completed"
      ? "var(--text-secondary)"
      : "var(--red)";
  return (
    <span
      className="inline-block rounded-full flex-shrink-0"
      style={{
        width: size,
        height: size,
        background: color,
        boxShadow: status === "live" ? `0 0 ${size}px ${color}` : "none",
      }}
    />
  );
}

function RunCard({
  run,
  selected,
  onSelect,
}: {
  run: RunInfo;
  selected: boolean;
  onSelect: () => void;
}) {
  const label = run.name.split("/").pop() ?? run.name;
  const category = run.name.includes("/") ? run.name.split("/")[0] : "curriculum";

  return (
    <button
      onClick={onSelect}
      className="w-full text-left px-3 py-2.5 rounded transition-all"
      style={{
        background: selected ? "var(--bg-card)" : "transparent",
        border: `1px solid ${selected ? (run.status === "live" ? "rgba(74, 222, 128, 0.25)" : "var(--accent-dim)") : "transparent"}`,
      }}
    >
      <div className="flex items-center gap-2">
        <StatusDot status={run.status} />
        <span
          className="text-sm truncate"
          style={{
            color: selected ? "var(--text-primary)" : "var(--text-secondary)",
            fontFamily: "var(--font-mono, monospace)",
          }}
        >
          {label}
        </span>
      </div>
      <div className="flex items-center gap-2 mt-1 pl-3.5">
        <span className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-secondary)" }}>
          {category}
        </span>
        {run.has_pool && (
          <span className="text-[10px]" style={{ color: "var(--accent)" }}>pool</span>
        )}
        {run.checkpoints.length > 0 && (
          <span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>
            {run.checkpoints.length} ckpt
          </span>
        )}
      </div>
    </button>
  );
}

function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} />;
}

function SidebarSkeleton() {
  return (
    <div className="space-y-2 p-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="space-y-1.5">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-3 w-1/2" />
        </div>
      ))}
    </div>
  );
}

function ChartsSkeleton() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="rounded-lg p-4"
          style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
        >
          <Skeleton className="h-3 w-32 mb-4" />
          <Skeleton className="h-[260px] w-full" />
        </div>
      ))}
    </div>
  );
}

interface LiveStats {
  step: number;
  totalPoints: number;
  winRate: number | null;
  elo: number | null;
  policyLoss: number | null;
  epReturn: number | null;
  parsedAt: string;
  /** Current update number (derived from step / (n_envs * n_steps * action_repeat)) */
  update: number;
}

/** Extract summary stats from metrics for the live header. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function useLiveStats(metrics: TrainingMetrics | null, config: Record<string, any> | null): LiveStats | null {
  return useMemo(() => {
    if (!metrics) return null;

    let latestStep = 0;
    let totalPoints = 0;
    const allCategories = ["losses", "charts", "selfplay", "eval"] as const;

    for (const cat of allCategories) {
      const catData = metrics[cat];
      if (!catData) continue;
      for (const points of Object.values(catData)) {
        if (!points || points.length === 0) continue;
        totalPoints += points.length;
        const lastPt = points[points.length - 1];
        if (lastPt.step > latestStep) latestStep = lastPt.step;
      }
    }

    const latest = (cat: string, key: string): number | null => {
      const catData = metrics[cat as keyof typeof metrics];
      if (!catData || typeof catData !== "object") return null;
      const points = (catData as Record<string, { step: number; value: number }[]>)[key];
      if (!points || points.length === 0) return null;
      return points[points.length - 1].value;
    };

    // Derive current update number from step and config
    const nEnvs = config?.training?.n_envs ?? 256;
    const nSteps = config?.training?.n_steps ?? 2048;
    const stepsPerUpdate = nEnvs * nSteps;
    const update = stepsPerUpdate > 0 ? Math.floor(latestStep / stepsPerUpdate) : 0;

    return {
      step: latestStep,
      totalPoints,
      winRate: latest("charts", "win_rate"),
      elo: latest("selfplay", "learner_elo"),
      policyLoss: latest("losses", "policy_loss"),
      epReturn: latest("charts", "ep_return_mean"),
      parsedAt: metrics.parsed_at,
      update,
    };
  }, [metrics, config]);
}

type DrMode = "none" | "narrow" | "full";

/** Derive the current domain randomization mode from config + update. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function deriveDrMode(config: Record<string, any> | null, update: number): DrMode | null {
  if (!config) return null;
  const noneUntil = config.domain_randomization?.none_until ?? 0;
  const narrowUntil = config.domain_randomization?.narrow_until ?? 0;
  if (noneUntil === 0 && narrowUntil === 0) return null; // no DR config
  if (update < noneUntil) return "none";
  if (update < narrowUntil) return "narrow";
  return "full";
}

const DR_BADGE_STYLES: Record<DrMode, { bg: string; color: string; border: string }> = {
  none: {
    bg: "rgba(107, 107, 107, 0.1)",
    color: "var(--text-secondary)",
    border: "1px solid rgba(107, 107, 107, 0.2)",
  },
  narrow: {
    bg: "rgba(250, 204, 21, 0.1)",
    color: "rgb(250, 204, 21)",
    border: "1px solid rgba(250, 204, 21, 0.2)",
  },
  full: {
    bg: "rgba(74, 222, 128, 0.1)",
    color: "var(--green)",
    border: "1px solid rgba(74, 222, 128, 0.2)",
  },
};

interface StageInfo {
  name: string;
  current: number;
  total: number;
  /** Overall progress 0..1 across all stages */
  overallProgress: number;
  totalUpdates: number;
  /** ETA string like "~2h 15m" */
  eta: string | null;
}

/** Derive the current training stage from config + current update. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function deriveStage(config: Record<string, any> | null, update: number, metrics: TrainingMetrics | null): StageInfo | null {
  if (!config) return null;

  const currUpdates = config.curriculum?.updates ?? 0;
  const transUpdates = config.transition?.updates ?? 0;
  const spUpdates = config.selfplay?.updates ?? 0;
  const totalUpdates = currUpdates + transUpdates + spUpdates;

  if (totalUpdates === 0) {
    // Not a unified pipeline — maybe simple curriculum
    const numUpdates = config.training?.num_updates;
    if (numUpdates) {
      return {
        name: "Training",
        current: update,
        total: numUpdates,
        overallProgress: Math.min(update / numUpdates, 1),
        totalUpdates: numUpdates,
        eta: estimateEta(metrics, update, numUpdates),
      };
    }
    return null;
  }

  let stageName: string;
  let stageStart: number;
  let stageEnd: number;

  if (update < currUpdates) {
    stageName = "Curriculum";
    stageStart = 0;
    stageEnd = currUpdates;
  } else if (update < currUpdates + transUpdates) {
    stageName = "Transition";
    stageStart = currUpdates;
    stageEnd = currUpdates + transUpdates;
  } else {
    stageName = "Self-Play";
    stageStart = currUpdates + transUpdates;
    stageEnd = totalUpdates;
  }

  return {
    name: stageName,
    current: update - stageStart,
    total: stageEnd - stageStart,
    overallProgress: Math.min(update / totalUpdates, 1),
    totalUpdates,
    eta: estimateEta(metrics, update, totalUpdates),
  };
}

/** Estimate time remaining by measuring wall-time velocity from metrics. */
function estimateEta(metrics: TrainingMetrics | null, currentUpdate: number, totalUpdates: number): string | null {
  if (!metrics || currentUpdate <= 1) return null;

  // Find the earliest and latest wall_time across all metrics
  let earliest = Infinity;
  let latest = 0;
  let earliestStep = Infinity;
  let latestStep = 0;

  for (const cat of ["losses", "charts", "selfplay", "eval"] as const) {
    const catData = metrics[cat];
    if (!catData) continue;
    for (const points of Object.values(catData)) {
      if (!points || points.length < 2) continue;
      const first = points[0];
      const last = points[points.length - 1];
      if (first.wall_time > 0 && first.wall_time < earliest) {
        earliest = first.wall_time;
        earliestStep = first.step;
      }
      if (last.wall_time > latest) {
        latest = last.wall_time;
        latestStep = last.step;
      }
    }
  }

  if (earliest >= latest || earliestStep >= latestStep) return null;

  const elapsedSecs = latest - earliest;
  const stepsCompleted = latestStep - earliestStep;
  if (stepsCompleted <= 0) return null;

  // Extrapolate: how much wall-time per update
  const secsPerStep = elapsedSecs / stepsCompleted;
  // We need the steps-per-update to convert remaining updates to steps
  // Use the ratio: currentUpdate corresponds to latestStep
  const stepsPerUpdate = latestStep > 0 ? latestStep / currentUpdate : 0;
  if (stepsPerUpdate <= 0) return null;

  const remainingUpdates = totalUpdates - currentUpdate;
  const remainingSteps = remainingUpdates * stepsPerUpdate;
  const remainingSecs = remainingSteps * secsPerStep;

  if (remainingSecs <= 0) return null;

  const hours = Math.floor(remainingSecs / 3600);
  const mins = Math.floor((remainingSecs % 3600) / 60);

  if (hours > 24) return `~${Math.round(hours / 24)}d ${hours % 24}h`;
  if (hours > 0) return `~${hours}h ${mins}m`;
  if (mins > 0) return `~${mins}m`;
  return "<1m";
}

function LiveHeader({
  run,
  stats,
  stage,
  drMode,
}: {
  run: RunInfo;
  stats: LiveStats | null;
  stage: StageInfo | null;
  drMode: DrMode | null;
}) {
  const label = run.name.split("/").pop() ?? run.name;

  return (
    <div
      className="rounded-lg px-5 py-4"
      style={{
        background: "linear-gradient(135deg, rgba(74, 222, 128, 0.04) 0%, rgba(91, 155, 245, 0.04) 100%)",
        border: "1px solid rgba(74, 222, 128, 0.15)",
      }}
    >
      {/* Top row: name + stage + time */}
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        <span
          className="inline-block w-2 h-2 rounded-full animate-pulse"
          style={{ background: "var(--green)", boxShadow: "0 0 8px var(--green)" }}
        />
        <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--green)" }}>
          Live
        </span>
        <span
          className="text-sm font-semibold"
          style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono, monospace)" }}
        >
          {label}
        </span>

        {stage && (
          <span
            className="text-[11px] px-2 py-0.5 rounded"
            style={{
              background: "rgba(91, 155, 245, 0.1)",
              color: "var(--accent)",
              border: "1px solid rgba(91, 155, 245, 0.2)",
              fontFamily: "var(--font-mono, monospace)",
            }}
          >
            {stage.name} {stage.current}/{stage.total}
          </span>
        )}

        {drMode && (
          <span
            className="text-[11px] px-2 py-0.5 rounded"
            style={{
              ...DR_BADGE_STYLES[drMode],
              fontFamily: "var(--font-mono, monospace)",
            }}
          >
            DR {drMode}
          </span>
        )}

        {stage?.eta && (
          <span className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
            ETA {stage.eta}
          </span>
        )}

        {stats?.parsedAt && (
          <span className="text-[10px] ml-auto" style={{ color: "var(--text-secondary)" }}>
            Updated {new Date(stats.parsedAt).toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* Progress bar */}
      {stage && stage.overallProgress > 0 && (
        <div className="mb-3">
          <div
            className="h-1 rounded-full overflow-hidden"
            style={{ background: "rgba(255,255,255,0.05)" }}
          >
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${Math.min(stage.overallProgress * 100, 100)}%`,
                background: "linear-gradient(90deg, var(--green), var(--accent))",
              }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>
              Update {stats?.update ?? 0} / {stage.totalUpdates}
            </span>
            <span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>
              {(stage.overallProgress * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* Stats row */}
      {stats && stats.step > 0 && (
        <div className="flex items-center gap-6 flex-wrap">
          <Stat label="Step" value={formatStep(stats.step)} />
          {stats.elo !== null && <Stat label="ELO" value={Math.round(stats.elo).toLocaleString()} accent />}
          {stats.winRate !== null && <Stat label="Win Rate" value={`${(stats.winRate * 100).toFixed(1)}%`} />}
          {stats.epReturn !== null && <Stat label="Ep Return" value={stats.epReturn.toFixed(2)} />}
          {stats.policyLoss !== null && <Stat label="Policy Loss" value={stats.policyLoss.toFixed(4)} />}
          <Stat label="Data Points" value={stats.totalPoints.toLocaleString()} />
        </div>
      )}

      {stats && stats.step === 0 && (
        <div className="flex items-center gap-2">
          <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
            Waiting for first metrics...
          </span>
          <span className="inline-block w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "var(--text-secondary)" }} />
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-secondary)" }}>
        {label}
      </div>
      <div
        className="text-sm font-semibold"
        style={{
          color: accent ? "var(--accent)" : "var(--text-primary)",
          fontFamily: "var(--font-mono, monospace)",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function formatStep(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}k`;
  return String(v);
}

export default function TrainingPage() {
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [managing, setManaging] = useState(false);

  const {
    runs,
    loading: runsLoading,
    error: runsError,
    refresh: refreshRuns,
  } = useTrainingRuns(autoRefresh);

  const currentRun = runs.find((r) => r.name === selectedRun);
  const isLive = currentRun?.status === "live";

  const { metrics, loading: metricsLoading, error: metricsError } =
    useTrainingMetrics(selectedRun, isLive, autoRefresh);
  const { pool, loading: poolLoading } = usePoolData(selectedRun, isLive, autoRefresh);
  const runConfig = useRunConfig(isLive ? selectedRun : null);

  const liveStats = useLiveStats(metrics, runConfig);
  const stage = useMemo(
    () => deriveStage(runConfig, liveStats?.update ?? 0, metrics),
    [runConfig, liveStats?.update, metrics]
  );
  const drMode = useMemo(
    () => deriveDrMode(runConfig, liveStats?.update ?? 0),
    [runConfig, liveStats?.update]
  );

  // Auto-select: prefer live run, then completed, then first
  useEffect(() => {
    if (runs.length === 0) return;
    if (selectedRun && runs.find((r) => r.name === selectedRun)) return;
    const best =
      runs.find((r) => r.status === "live") ??
      runs.find((r) => r.status === "completed") ??
      runs[0];
    setSelectedRun(best.name);
  }, [runs, selectedRun]);

  const manageRun = useCallback(
    async (action: "hide" | "delete", run: string) => {
      const msg =
        action === "delete"
          ? `Permanently delete "${run}" from Modal volume?`
          : `Hide "${run}" from the dashboard?`;
      if (!confirm(msg)) return;
      setManaging(true);
      try {
        await fetch("/api/training/manage", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action, run }),
        });
        refreshRuns();
      } finally {
        setManaging(false);
      }
    },
    [refreshRuns]
  );

  // Group runs
  const liveRuns = runs.filter((r) => r.status === "live");
  const completedRuns = runs.filter((r) => r.status === "completed");
  const failedRuns = runs.filter((r) => r.status === "failed");

  return (
    <div className="flex" style={{ minHeight: "calc(100vh - 48px)" }}>
      {/* Sidebar */}
      <aside
        className="w-64 flex-shrink-0 border-r overflow-y-auto"
        style={{
          background: "var(--bg-secondary)",
          borderColor: "var(--border)",
          maxHeight: "calc(100vh - 48px)",
          position: "sticky",
          top: 48,
        }}
      >
        <div
          className="px-4 py-3 border-b flex items-center justify-between"
          style={{ borderColor: "var(--border)" }}
        >
          <label
            className="flex items-center gap-2 text-[11px] cursor-pointer select-none"
            style={{ color: "var(--text-secondary)" }}
          >
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="accent-blue-500"
            />
            Auto-refresh
          </label>
          {isLive && (
            <span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>
              10s
            </span>
          )}
        </div>

        {runsLoading ? (
          <SidebarSkeleton />
        ) : runsError ? (
          <div className="p-4 text-xs" style={{ color: "var(--red)" }}>
            {runsError}
          </div>
        ) : (
          <div className="py-2">
            {liveRuns.length > 0 && (
              <RunGroup label="Live" runs={liveRuns} selectedRun={selectedRun} onSelect={setSelectedRun} />
            )}
            {completedRuns.length > 0 && (
              <RunGroup label="Completed" runs={completedRuns} selectedRun={selectedRun} onSelect={setSelectedRun} />
            )}
            {failedRuns.length > 0 && (
              <RunGroup label="Failed" runs={failedRuns} selectedRun={selectedRun} onSelect={setSelectedRun} />
            )}
          </div>
        )}
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
        {/* Live header — prominent when viewing a live run */}
        {currentRun?.status === "live" && (
          <LiveHeader run={currentRun} stats={liveStats} stage={stage} drMode={drMode} />
        )}

        {/* Non-live header */}
        {currentRun && currentRun.status !== "live" && (
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2.5">
              <StatusDot status={currentRun.status} />
              <h1
                className="text-lg font-semibold"
                style={{ fontFamily: "var(--font-mono, monospace)" }}
              >
                {currentRun.name.split("/").pop()}
              </h1>
              <span
                className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded"
                style={{
                  color: currentRun.status === "completed" ? "var(--text-secondary)" : "var(--red)",
                  background: currentRun.status === "completed" ? "rgba(107,107,107,0.1)" : "rgba(248,113,113,0.1)",
                  border: "1px solid currentColor",
                  opacity: 0.8,
                }}
              >
                {currentRun.status}
              </span>
            </div>
            <div className="flex items-center gap-3 ml-auto">
              {metrics?.parsed_at && (
                <span className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                  {new Date(metrics.parsed_at).toLocaleString()}
                </span>
              )}
              <button
                className="text-[11px] transition-colors hover:opacity-100 opacity-50 disabled:opacity-30"
                style={{ color: "var(--text-secondary)" }}
                disabled={managing}
                onClick={() => manageRun("hide", currentRun.name)}
              >
                Hide
              </button>
              <button
                className="text-[11px] transition-colors hover:opacity-100 opacity-50 disabled:opacity-30"
                style={{ color: "var(--red)" }}
                disabled={managing}
                onClick={() => manageRun("delete", currentRun.name)}
              >
                Delete
              </button>
            </div>
          </div>
        )}

        {/* Error */}
        {metricsError && (
          <div
            className="text-xs rounded-lg px-4 py-3"
            style={{
              color: "var(--red)",
              background: "rgba(248,113,113,0.05)",
              border: "1px solid rgba(248,113,113,0.15)",
            }}
          >
            {metricsError}
          </div>
        )}

        {/* Charts */}
        {metricsLoading ? <ChartsSkeleton /> : selectedRun ? <MetricsCharts metrics={metrics} /> : null}

        {/* Pool */}
        {!poolLoading && currentRun?.has_pool && <PoolViewer pool={pool} />}

        {/* Checkpoint Match Viewer — only when run has ONNX checkpoints */}
        {currentRun && currentRun.onnx_checkpoints && currentRun.onnx_checkpoints.length > 0 && (
          <CheckpointMatchViewer
            checkpoints={currentRun.onnx_checkpoints}
            runName={currentRun.name}
          />
        )}

        {/* Empty */}
        {!selectedRun && !runsLoading && runs.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 text-center" style={{ color: "var(--text-secondary)" }}>
            <p className="text-sm mb-2">No training runs found</p>
            <p className="text-xs opacity-60">Run a training job on Modal to see data here</p>
          </div>
        )}
      </main>
    </div>
  );
}

function RunGroup({
  label,
  runs,
  selectedRun,
  onSelect,
}: {
  label: string;
  runs: RunInfo[];
  selectedRun: string | null;
  onSelect: (name: string) => void;
}) {
  return (
    <div className="mb-1">
      <div
        className="px-4 py-1.5 text-[10px] uppercase tracking-wider font-semibold"
        style={{ color: "var(--text-secondary)" }}
      >
        {label}
      </div>
      <div className="px-2 space-y-0.5">
        {runs.map((r) => (
          <RunCard key={r.name} run={r} selected={r.name === selectedRun} onSelect={() => onSelect(r.name)} />
        ))}
      </div>
    </div>
  );
}

