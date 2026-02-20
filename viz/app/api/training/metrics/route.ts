import { NextRequest, NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";

const MODAL_METRICS_URL =
  "https://benedictbrady--dogfight-dashboard-metrics.modal.run";
const DASHBOARD_DATA_DIR = path.join(
  process.cwd(),
  "..",
  "training",
  "dashboard_data"
);

export async function GET(request: NextRequest) {
  const run = request.nextUrl.searchParams.get("run");

  if (!run) {
    return NextResponse.json(
      { error: "Missing required query parameter: run" },
      { status: 400 }
    );
  }

  // Try Modal endpoint first
  try {
    const res = await fetch(
      `${MODAL_METRICS_URL}?run=${encodeURIComponent(run)}`,
      { signal: AbortSignal.timeout(30_000) }
    );
    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    }
    // If Modal returns 404, fall through to local
    if (res.status !== 404) {
      // Other error — still try local
    }
  } catch {
    // Modal unavailable, fall through to local
  }

  // Fallback: read from local file
  const metricsPath = path.join(DASHBOARD_DATA_DIR, run, "metrics.json");
  try {
    const data = await fs.readFile(metricsPath, "utf-8");
    return NextResponse.json(JSON.parse(data));
  } catch (error) {
    if (
      error instanceof Error &&
      "code" in error &&
      (error as NodeJS.ErrnoException).code === "ENOENT"
    ) {
      return NextResponse.json(
        { error: `Metrics not found for run: ${run}` },
        { status: 404 }
      );
    }
    console.error("Failed to read metrics:", error);
    return NextResponse.json(
      { error: "Failed to read metrics" },
      { status: 500 }
    );
  }
}
