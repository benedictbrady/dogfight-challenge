import { NextRequest, NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";
import type { PoolData, PoolEntry } from "@/app/lib/training-types";

export const dynamic = "force-dynamic";

const MODAL_POOL_URL =
  "https://benedictbrady--dogfight-dashboard-pool.modal.run";
const DASHBOARD_DATA_DIR = path.join(
  process.cwd(),
  "..",
  "training",
  "dashboard_data"
);

// Raw pool.json has entries with update_num instead of generation
interface RawPoolEntry {
  name: string;
  elo: number;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  update_num?: number;
  generation?: number;
  path?: string;
}

function normalizePool(raw: Record<string, unknown>): PoolData {
  const rawEntries = (raw.entries ?? []) as RawPoolEntry[];
  const entries: PoolEntry[] = rawEntries.map((e) => ({
    name: e.name,
    elo: e.elo,
    games: e.games,
    wins: e.wins,
    losses: e.losses,
    draws: e.draws,
    generation: e.generation ?? e.update_num ?? 0,
  }));
  return {
    entries,
    updated_at: raw.updated_at as string | undefined,
  };
}

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
      `${MODAL_POOL_URL}?run=${encodeURIComponent(run)}`,
      { signal: AbortSignal.timeout(10_000) }
    );
    if (res.ok) {
      const raw = await res.json();
      return NextResponse.json(normalizePool(raw));
    }
  } catch {
    // Modal unavailable, fall through to local
  }

  // Fallback: read from local file
  const poolPath = path.join(DASHBOARD_DATA_DIR, run, "pool.json");
  try {
    const data = await fs.readFile(poolPath, "utf-8");
    const raw = JSON.parse(data);
    return NextResponse.json(normalizePool(raw));
  } catch (error) {
    if (
      error instanceof Error &&
      "code" in error &&
      (error as NodeJS.ErrnoException).code === "ENOENT"
    ) {
      return NextResponse.json(
        { error: `Pool data not found for run: ${run}` },
        { status: 404 }
      );
    }
    console.error("Failed to read pool data:", error);
    return NextResponse.json(
      { error: "Failed to read pool data" },
      { status: 500 }
    );
  }
}
