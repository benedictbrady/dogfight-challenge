"use client";

import { useState, useEffect } from "react";
import type { MatchConfig } from "../hooks/useMatch";

interface MatchSetupProps {
  onStartMatch: (config: MatchConfig) => void;
  isConnected: boolean;
}

export default function MatchSetup({ onStartMatch, isConnected }: MatchSetupProps) {
  const [policies, setPolicies] = useState<string[]>([]);
  const [p0, setP0] = useState("");
  const [p1, setP1] = useState("");
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [serverOnline, setServerOnline] = useState(false);

  useEffect(() => {
    async function fetchPolicies() {
      try {
        const res = await fetch("http://localhost:3001/api/policies");
        if (res.ok) {
          const data = await res.json();
          const policyList = data.policies ?? data ?? [];
          setPolicies(policyList);
          setServerOnline(true);
          if (policyList.length > 0) {
            setP0(policyList[0]);
            setP1(policyList[Math.min(1, policyList.length - 1)]);
          }
        }
      } catch {
        setServerOnline(false);
        console.warn("Could not fetch policies from backend");
      }
    }
    fetchPolicies();
  }, []);

  const handleStart = () => {
    if (!p0 || !p1) return;
    setLoading(true);
    onStartMatch({ p0, p1, seed });
    setTimeout(() => setLoading(false), 1000);
  };

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-sm font-semibold text-[#c4b894] uppercase tracking-[0.2em]">
        Sortie Orders
      </h2>

      {/* Player 0 */}
      <div>
        <label className="block text-xs text-[#c4a050] mb-1">
          Pilot 0 (Tan)
        </label>
        {policies.length > 0 ? (
          <select
            className="w-full bg-[#2a2618] text-[#d4c8a0] text-sm rounded px-2 py-1.5 border border-[#4a4030] focus:border-[#8b7748] focus:outline-none"
            value={p0}
            onChange={(e) => setP0(e.target.value)}
          >
            {policies.map((pol) => (
              <option key={pol} value={pol}>
                {pol}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            className="w-full bg-[#2a2618] text-[#d4c8a0] text-sm rounded px-2 py-1.5 border border-[#4a4030] focus:border-[#8b7748] focus:outline-none"
            placeholder="Policy name..."
            value={p0}
            onChange={(e) => setP0(e.target.value)}
          />
        )}
      </div>

      {/* Player 1 */}
      <div>
        <label className="block text-xs text-[#b83030] mb-1">
          Pilot 1 (Red)
        </label>
        {policies.length > 0 ? (
          <select
            className="w-full bg-[#2a2618] text-[#d4c8a0] text-sm rounded px-2 py-1.5 border border-[#4a4030] focus:border-[#8b3030] focus:outline-none"
            value={p1}
            onChange={(e) => setP1(e.target.value)}
          >
            {policies.map((pol) => (
              <option key={pol} value={pol}>
                {pol}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            className="w-full bg-[#2a2618] text-[#d4c8a0] text-sm rounded px-2 py-1.5 border border-[#4a4030] focus:border-[#8b3030] focus:outline-none"
            placeholder="Policy name..."
            value={p1}
            onChange={(e) => setP1(e.target.value)}
          />
        )}
      </div>

      {/* Seed */}
      <div>
        <label className="block text-xs text-[#8b7a58] mb-1">Seed</label>
        <input
          type="number"
          className="w-full bg-[#2a2618] text-[#d4c8a0] text-sm rounded px-2 py-1.5 border border-[#4a4030] focus:border-[#8b7748] focus:outline-none"
          value={seed}
          onChange={(e) => setSeed(Number(e.target.value))}
        />
      </div>

      {/* Start button */}
      <button
        className="w-full py-2 rounded text-sm font-semibold tracking-wider transition-colors disabled:opacity-40
          bg-[#5a4a28] hover:bg-[#6b5a32] text-[#e8d8a8] border border-[#8b7748]"
        onClick={handleStart}
        disabled={loading || (!p0 && !p1)}
      >
        {loading ? "Scrambling..." : isConnected ? "New Sortie" : "Scramble!"}
      </button>

      {/* Connection status */}
      <div className="flex items-center gap-2 text-xs text-[#6b5a40]">
        <div
          className={`w-2 h-2 rounded-full ${
            serverOnline ? "bg-[#6b8b40]" : "bg-[#8b3030]"
          }`}
        />
        {isConnected ? "On sortie" : serverOnline ? "HQ online" : "No signal"}
      </div>

      {/* Info */}
      <div className="mt-4 text-xs text-[#5a4a38] space-y-1">
        <p>HQ: ws://localhost:3001</p>
        <p>Controls:</p>
        <p className="pl-2">- Scroll to zoom</p>
        <p className="pl-2">- Right-click drag to pan</p>
      </div>
    </div>
  );
}
