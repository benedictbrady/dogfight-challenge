"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { RunInfo, TrainingMetrics, PoolData } from "../lib/training-types";

const LIVE_INTERVAL = 10_000; // 10s for live runs
const IDLE_INTERVAL = 60_000; // 60s for completed/failed

export function useTrainingRuns(autoRefresh: boolean = true) {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/training/runs");
      if (!res.ok) throw new Error(`Failed to fetch runs: ${res.status}`);
      const data = await res.json();
      setRuns(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch runs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Faster polling when any run is live
  const hasLive = runs.some((r) => r.status === "live");
  const interval = hasLive ? LIVE_INTERVAL : IDLE_INTERVAL;

  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(refresh, interval);
    return () => clearInterval(id);
  }, [autoRefresh, refresh, interval]);

  return { runs, loading, error, refresh };
}

export function useTrainingMetrics(
  runName: string | null,
  isLive: boolean = false,
  autoRefresh: boolean = true
) {
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initialLoad = useRef(true);

  const refresh = useCallback(async () => {
    if (!runName) return;
    // Only show loading spinner on first load, not on refreshes
    if (initialLoad.current) {
      setLoading(true);
      initialLoad.current = false;
    }
    try {
      const res = await fetch(
        `/api/training/metrics?run=${encodeURIComponent(runName)}`
      );
      if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.status}`);
      const data = await res.json();
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch metrics");
    } finally {
      setLoading(false);
    }
  }, [runName]);

  // Reset when runName changes
  useEffect(() => {
    initialLoad.current = true;
    setMetrics(null);
    setError(null);
    if (runName) refresh();
  }, [runName, refresh]);

  const interval = isLive ? LIVE_INTERVAL : IDLE_INTERVAL;
  useEffect(() => {
    if (!autoRefresh || !runName) return;
    const id = setInterval(refresh, interval);
    return () => clearInterval(id);
  }, [autoRefresh, runName, refresh, interval]);

  return { metrics, loading, error, refresh };
}

export function usePoolData(
  runName: string | null,
  isLive: boolean = false,
  autoRefresh: boolean = true
) {
  const [pool, setPool] = useState<PoolData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!runName) return;
    try {
      setLoading(true);
      const res = await fetch(
        `/api/training/pool?run=${encodeURIComponent(runName)}`
      );
      if (res.status === 404) {
        setPool(null);
        setError(null);
        return;
      }
      if (!res.ok) throw new Error(`Failed to fetch pool data: ${res.status}`);
      const data = await res.json();
      setPool(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch pool data");
    } finally {
      setLoading(false);
    }
  }, [runName]);

  useEffect(() => {
    setPool(null);
    setError(null);
    if (runName) refresh();
  }, [runName, refresh]);

  const interval = isLive ? LIVE_INTERVAL : IDLE_INTERVAL;
  useEffect(() => {
    if (!autoRefresh || !runName) return;
    const id = setInterval(refresh, interval);
    return () => clearInterval(id);
  }, [autoRefresh, runName, refresh, interval]);

  return { pool, loading, error, refresh };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function useRunConfig(runName: string | null): Record<string, any> | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [config, setConfig] = useState<Record<string, any> | null>(null);

  useEffect(() => {
    setConfig(null);
    if (!runName) return;
    fetch(`/api/training/config?run=${encodeURIComponent(runName)}`)
      .then((r) => r.json())
      .then((d) => setConfig(d))
      .catch(() => {});
  }, [runName]);

  return config;
}
