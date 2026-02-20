import { NextRequest, NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";
import type { CheckpointInfo } from "@/app/lib/training-types";

export const dynamic = "force-dynamic";

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

  const checkpointsDir = path.join(DASHBOARD_DATA_DIR, run, "checkpoints");

  try {
    const files = await fs.readdir(checkpointsDir);
    const checkpoints: CheckpointInfo[] = files
      .filter((f) => f.endsWith(".onnx"))
      .sort()
      .map((f) => ({
        name: f.replace(/\.onnx$/, ""),
        path: `training/dashboard_data/${run}/checkpoints/${f}`,
      }));

    return NextResponse.json(checkpoints);
  } catch (error) {
    if (
      error instanceof Error &&
      "code" in error &&
      (error as NodeJS.ErrnoException).code === "ENOENT"
    ) {
      return NextResponse.json([]);
    }
    console.error("Failed to list checkpoints:", error);
    return NextResponse.json(
      { error: "Failed to list checkpoints" },
      { status: 500 }
    );
  }
}
