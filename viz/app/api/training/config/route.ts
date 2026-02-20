import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const MODAL_CONFIG_URL =
  "https://benedictbrady--dogfight-dashboard-config.modal.run";

export async function GET(request: NextRequest) {
  const run = request.nextUrl.searchParams.get("run");

  if (!run) {
    return NextResponse.json(
      { error: "Missing required query parameter: run" },
      { status: 400 }
    );
  }

  try {
    const res = await fetch(
      `${MODAL_CONFIG_URL}?run=${encodeURIComponent(run)}`,
      { signal: AbortSignal.timeout(10_000) }
    );
    if (res.status === 404) {
      return NextResponse.json(null);
    }
    if (!res.ok) {
      return NextResponse.json(null);
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(null);
  }
}
