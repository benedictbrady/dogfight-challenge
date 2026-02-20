"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function TrainingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <>
      <style>{`
        html, body {
          overflow: auto !important;
          height: auto !important;
          background: #0a0a0a !important;
          color: #e5e5e5 !important;
          color-scheme: dark;
        }

        .training-root {
          --bg-primary: #0a0a0a;
          --bg-secondary: #111111;
          --bg-card: #161616;
          --text-primary: #e5e5e5;
          --text-secondary: #6b6b6b;
          --border: #232323;
          --accent: #5b9bf5;
          --accent-dim: #2a4a7a;
          --green: #4ade80;
          --red: #f87171;
          --yellow: #facc15;
          color-scheme: dark;
        }

        .training-root *, .training-root *::before, .training-root *::after {
          border-color: var(--border);
        }

        .training-root ::selection {
          background: var(--accent-dim);
          color: var(--text-primary);
        }

        .training-root select,
        .training-root input[type="checkbox"] {
          color-scheme: dark;
        }

        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        .skeleton {
          background: linear-gradient(90deg, var(--bg-card) 25%, #1e1e1e 50%, var(--bg-card) 75%);
          background-size: 200% 100%;
          animation: shimmer 1.5s ease-in-out infinite;
          border-radius: 4px;
        }
      `}</style>
      <div className="training-root min-h-screen" style={{ background: "var(--bg-primary)", color: "var(--text-primary)" }}>
        <nav
          className="sticky top-0 z-20 flex items-center gap-6 px-6 h-12 border-b"
          style={{ background: "var(--bg-secondary)", borderColor: "var(--border)" }}
        >
          <span
            className="text-sm font-bold tracking-[0.2em]"
            style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono, monospace)" }}
          >
            DOGFIGHT
          </span>
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="text-xs transition-colors"
              style={{
                color: pathname === "/" ? "var(--text-primary)" : "var(--text-secondary)",
              }}
            >
              Match Viz
            </Link>
            <Link
              href="/training"
              className="text-xs transition-colors"
              style={{
                color: pathname === "/training" ? "var(--accent)" : "var(--text-secondary)",
              }}
            >
              Training
            </Link>
          </div>
        </nav>
        {children}
      </div>
    </>
  );
}
