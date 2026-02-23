"use client";

import { useState, useEffect } from "react";
import type { MatchConfig } from "../hooks/useMatch";

interface MatchSetupProps {
  onStartMatch: (config: MatchConfig) => void;
  onSeedChange: (seed: number) => void;
  isConnected: boolean;
}

const OPPONENT_RANK: Record<string, number> = {
  chaser: 1,
  ace: 2,
  dogfighter: 3,
  brawler: 4,
};

const OPPONENT_DOTS: Record<string, string> = {
  chaser: "\u2022",
  ace: "\u2022\u2022",
  dogfighter: "\u2022\u2022\u2022",
  brawler: "\u2022\u2022\u2022\u2022",
};

export default function MatchSetup({ onStartMatch, onSeedChange, isConnected }: MatchSetupProps) {
  const [userModels, setUserModels] = useState<string[]>([]);
  const [opponents, setOpponents] = useState<string[]>([]);
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
          const models: string[] = data.user_models ?? [];
          const opps: string[] = data.opponents ?? [];
          setUserModels(models);
          opps.sort((a, b) => (OPPONENT_RANK[a] ?? 99) - (OPPONENT_RANK[b] ?? 99));
          setOpponents(opps);
          setServerOnline(true);
          if (models.length > 0) setP0(models[0]);
          if (opps.length > 0) setP1(opps[0]);
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

      {/* P0: Your Model (Blue) */}
      <div>
        <label className="block text-xs text-blue-600 mb-1 font-medium">Your Model (Blue)</label>
        {userModels.length > 1 ? (
          <select
            className="w-full bg-white text-gray-800 text-sm rounded px-2 py-1.5 border border-gray-300 focus:border-indigo-500 focus:outline-none"
            value={p0}
            onChange={(e) => setP0(e.target.value)}
          >
            {userModels.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        ) : (
          <div className="w-full bg-gray-50 text-gray-800 text-sm rounded px-2 py-1.5 border border-gray-200">
            {userModels[0] ?? "No models found"}
          </div>
        )}
      </div>

      {/* P1: Opponent (Red) */}
      <div>
        <label className="block text-xs text-red-600 mb-1 font-medium">Opponent (Red)</label>
        {opponents.length > 0 ? (
          <select
            className="w-full bg-white text-gray-800 text-sm rounded px-2 py-1.5 border border-gray-300 focus:border-red-500 focus:outline-none"
            value={p1}
            onChange={(e) => setP1(e.target.value)}
          >
            {opponents.map((o) => (
              <option key={o} value={o}>
                {o}{OPPONENT_DOTS[o] ? ` ${OPPONENT_DOTS[o]}` : ""}
              </option>
            ))}
          </select>
        ) : (
          <div className="w-full bg-gray-50 text-gray-500 text-sm rounded px-2 py-1.5 border border-gray-200">
            No opponents available
          </div>
        )}
      </div>

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
