"use client";

import { useState, useEffect } from "react";
import type { MatchConfig } from "../hooks/useMatch";

interface MatchSetupProps {
  onStartMatch: (config: MatchConfig) => void;
  onSeedChange: (seed: number) => void;
  isConnected: boolean;
}

function PlayerSelector({
  label,
  colorClass,
  focusClass,
  policies,
  value,
  onChange,
}: {
  label: string;
  colorClass: string;
  focusClass: string;
  policies: string[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div>
      <label className={`block text-xs ${colorClass} mb-1 font-medium`}>
        {label}
      </label>
      {policies.length > 0 ? (
        <select
          className={`w-full bg-white text-gray-800 text-sm rounded px-2 py-1.5 border border-gray-300 ${focusClass} focus:outline-none`}
          value={value}
          onChange={(e) => onChange(e.target.value)}
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
          className={`w-full bg-white text-gray-800 text-sm rounded px-2 py-1.5 border border-gray-300 ${focusClass} focus:outline-none`}
          placeholder="Policy name..."
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </div>
  );
}

export default function MatchSetup({ onStartMatch, onSeedChange, isConnected }: MatchSetupProps) {
  const [policies, setPolicies] = useState<string[]>([]);
  const [p0, setP0] = useState("");
  const [p1, setP1] = useState("");
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1000000));
  const [loading, setLoading] = useState(false);
  const [serverOnline, setServerOnline] = useState(false);

  // Notify parent of initial seed so spawn positions load on mount
  useEffect(() => {
    onSeedChange(seed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    onStartMatch({ p0, p1, seed, randomize_spawns: true });
    setTimeout(() => setLoading(false), 1000);
  };

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-sm font-semibold text-gray-800 uppercase tracking-[0.2em]">
        Match Setup
      </h2>

      <PlayerSelector
        label="Player 0 (Blue)"
        colorClass="text-blue-600"
        focusClass="focus:border-indigo-500"
        policies={policies}
        value={p0}
        onChange={setP0}
      />

      <PlayerSelector
        label="Player 1 (Red)"
        colorClass="text-red-600"
        focusClass="focus:border-red-500"
        policies={policies}
        value={p1}
        onChange={setP1}
      />

      {/* Seed */}
      <div>
        <label className="block text-xs text-gray-500 mb-1">Seed</label>
        <input
          type="number"
          className="w-full bg-white text-gray-800 text-sm rounded px-2 py-1.5 border border-gray-300 focus:border-indigo-500 focus:outline-none"
          value={seed}
          onChange={(e) => {
            const newSeed = Number(e.target.value);
            setSeed(newSeed);
            onSeedChange(newSeed);
          }}
        />
      </div>

      {/* Start button */}
      <button
        className="w-full py-2 rounded text-sm font-semibold tracking-wider transition-colors disabled:opacity-40
          bg-indigo-600 hover:bg-indigo-700 text-white border border-indigo-700"
        onClick={handleStart}
        disabled={loading || (!p0 && !p1)}
      >
        {loading ? "Starting..." : isConnected ? "New Match" : "Start Match"}
      </button>

      {/* Connection status */}
      <div className="flex items-center gap-2 text-xs text-gray-500">
        <div
          className={`w-2 h-2 rounded-full ${
            serverOnline ? "bg-green-500" : "bg-red-500"
          }`}
        />
        {isConnected ? "Match active" : serverOnline ? "Server online" : "No signal"}
      </div>

      {/* Info */}
      <div className="mt-4 text-xs text-gray-400 space-y-1">
        <p>Server: ws://localhost:3001</p>
        <p>Controls:</p>
        <p className="pl-2">- Scroll to zoom</p>
        <p className="pl-2">- Right-click drag to pan</p>
      </div>
    </div>
  );
}
