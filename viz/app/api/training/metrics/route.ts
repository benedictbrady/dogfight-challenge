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
  const status = request.nextUrl.searchParams.get("status");

  if (!run) {
    return NextResponse.json(
      { error: "Missing required query parameter: run" },
      { status: 400 }
    );
  }

  const localCachePath = path.join(DASHBOARD_DATA_DIR, run, "metrics.json");
  const isCompleted = status !== "live";

  // For completed runs, check local cache first
  if (isCompleted) {
    try {
      const data = await fs.readFile(localCachePath, "utf-8");
      return NextResponse.json(JSON.parse(data));
    } catch {
      // No local cache — fall through to Modal
    }
  }

  // Fetch from Modal
  let modalData: unknown = null;
  try {
    const res = await fetch(
      `${MODAL_METRICS_URL}?run=${encodeURIComponent(run)}`,
      { signal: AbortSignal.timeout(30_000) }
    );
    if (res.ok) {
      modalData = await res.json();

      // Cache locally for completed runs
      if (isCompleted && modalData) {
        try {
          await fs.mkdir(path.join(DASHBOARD_DATA_DIR, run), {
            recursive: true,
          });
          await fs.writeFile(
            localCachePath,
            JSON.stringify(modalData),
            "utf-8"
          );
        } catch {
          // Cache write failed — non-fatal
        }
      }

      return NextResponse.json(modalData);
    }
    // If Modal returns 404, fall through to local
    if (res.status !== 404) {
      // Other error — still try local
    }
  } catch {
    // Modal unavailable, fall through to local
  }

  // Fallback: read from local file (for live runs where cache wasn't checked above)
  try {
    const data = await fs.readFile(localCachePath, "utf-8");
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
