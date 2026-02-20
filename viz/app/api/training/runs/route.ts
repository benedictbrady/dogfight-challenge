import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";
import type { RunInfo } from "@/app/lib/training-types";

export const dynamic = "force-dynamic";

const MODAL_RUNS_URL =
  "https://benedictbrady--dogfight-dashboard-runs.modal.run";
const DASHBOARD_DATA_DIR = path.join(
  process.cwd(),
  "..",
  "training",
  "dashboard_data"
);

export async function GET() {
  // Try Modal endpoint first, fall back to local files
  try {
    const res = await fetch(MODAL_RUNS_URL, {
      signal: AbortSignal.timeout(10_000),
    });
    if (res.ok) {
      const modalRuns = await res.json();
      const runs: RunInfo[] = modalRuns.map(
        (r: {
          name: string;
          status?: string;
          has_metrics: boolean;
          has_pool: boolean;
          checkpoints: string[];
        }) => ({
          name: r.name,
          status: r.status ?? "completed",
          has_metrics: r.has_metrics,
          has_pool: r.has_pool,
          checkpoints: r.checkpoints,
        })
      );
      return NextResponse.json(runs);
    }
  } catch {
    // Modal unavailable, fall through to local
  }

  // Fallback: read from local dashboard_data/
  try {
    let entries: string[];
    try {
      const dirents = await fs.readdir(DASHBOARD_DATA_DIR, {
        withFileTypes: true,
      });
      entries = dirents.filter((d) => d.isDirectory()).map((d) => d.name);
    } catch {
      return NextResponse.json([]);
    }

    const runs: RunInfo[] = await Promise.all(
      entries.map(async (name) => {
        const runDir = path.join(DASHBOARD_DATA_DIR, name);
        let has_metrics = false;
        try {
          await fs.access(path.join(runDir, "metrics.json"));
          has_metrics = true;
        } catch {}
        let has_pool = false;
        try {
          await fs.access(path.join(runDir, "pool.json"));
          has_pool = true;
        } catch {}
        let checkpoints: string[] = [];
        try {
          const cpDir = path.join(runDir, "checkpoints");
          const files = await fs.readdir(cpDir);
          checkpoints = files
            .filter((f) => f.endsWith(".onnx"))
            .map((f) => f.replace(/\.onnx$/, ""))
            .sort();
        } catch {}
        return { name, status: "completed" as const, has_metrics, has_pool, checkpoints };
      })
    );

    return NextResponse.json(runs);
  } catch (error) {
    console.error("Failed to list training runs:", error);
    return NextResponse.json(
      { error: "Failed to list training runs" },
      { status: 500 }
    );
  }
}
