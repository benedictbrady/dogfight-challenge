"use client";

interface ControlsProps {
  currentFrame: number;
  totalFrames: number;
  isPlaying: boolean;
  speed: number;
  onSetFrame: (frame: number) => void;
  onTogglePlay: () => void;
  onSetSpeed: (speed: number) => void;
}

const SPEEDS = [0.25, 0.5, 1, 2, 4];

export default function Controls({
  currentFrame,
  totalFrames,
  isPlaying,
  speed,
  onSetFrame,
  onTogglePlay,
  onSetSpeed,
}: ControlsProps) {
  const canStep = totalFrames > 0;

  return (
    <div className="bg-[#1c1a14] border-t border-[#3a3526] px-4 py-2 flex items-center gap-3">
      {/* Playback controls */}
      <div className="flex items-center gap-1">
        <button
          className="px-2 py-1 text-xs bg-[#2a2618] hover:bg-[#3a3526] rounded text-[#c4b894] disabled:opacity-30 border border-[#3a3526]"
          disabled={!canStep || currentFrame <= 0}
          onClick={() => onSetFrame(Math.max(0, currentFrame - 1))}
          title="Step backward"
        >
          &lt;
        </button>

        <button
          className="px-3 py-1 text-xs bg-[#2a2618] hover:bg-[#3a3526] rounded text-[#c4b894] disabled:opacity-30 min-w-[60px] border border-[#3a3526]"
          disabled={totalFrames === 0}
          onClick={onTogglePlay}
        >
          {isPlaying ? "Pause" : "Play"}
        </button>

        <button
          className="px-2 py-1 text-xs bg-[#2a2618] hover:bg-[#3a3526] rounded text-[#c4b894] disabled:opacity-30 border border-[#3a3526]"
          disabled={!canStep || currentFrame >= totalFrames - 1}
          onClick={() => onSetFrame(Math.min(totalFrames - 1, currentFrame + 1))}
          title="Step forward"
        >
          &gt;
        </button>
      </div>

      {/* Scrub bar */}
      <div className="flex-1 mx-2">
        <input
          type="range"
          min={0}
          max={Math.max(0, totalFrames - 1)}
          value={currentFrame}
          onChange={(e) => onSetFrame(Number(e.target.value))}
          className="w-full h-1 bg-[#3a3526] rounded-lg appearance-none cursor-pointer accent-[#8b7748]"
          disabled={totalFrames === 0}
        />
      </div>

      {/* Frame counter */}
      <span className="text-xs text-[#8b7a58] font-mono min-w-[80px] text-right">
        {currentFrame + 1} / {totalFrames}
      </span>

      {/* Speed selector */}
      <div className="flex items-center gap-1 ml-2">
        <span className="text-xs text-[#8b7a58] mr-1">Speed:</span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            className={`px-2 py-1 text-xs rounded border ${
              speed === s
                ? "bg-[#5a4a28] text-[#e8d8a8] border-[#8b7748]"
                : "bg-[#2a2618] hover:bg-[#3a3526] text-[#8b7a58] border-[#3a3526]"
            }`}
            onClick={() => onSetSpeed(s)}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  );
}
