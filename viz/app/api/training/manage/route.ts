import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const MODAL_BASE = "https://benedictbrady--dogfight-dashboard";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { action, run } = body as { action: string; run: string };

  if (!action || !run) {
    return NextResponse.json(
      { error: "Missing action or run" },
      { status: 400 }
    );
  }

  const endpoints: Record<string, string> = {
    hide: `${MODAL_BASE}-hide.modal.run`,
    unhide: `${MODAL_BASE}-unhide.modal.run`,
    delete: `${MODAL_BASE}-delete.modal.run`,
  };

  const url = endpoints[action];
  if (!url) {
    return NextResponse.json(
      { error: `Unknown action: ${action}` },
      { status: 400 }
    );
  }

  try {
    const res = await fetch(`${url}?run=${encodeURIComponent(run)}`, {
      signal: AbortSignal.timeout(30_000),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    console.error(`Failed to ${action} run:`, error);
    return NextResponse.json(
      { error: `Failed to ${action} run` },
      { status: 500 }
    );
  }
}
