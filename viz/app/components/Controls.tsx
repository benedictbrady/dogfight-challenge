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
    <div className="bg-white border-t border-gray-200 px-4 py-2 flex items-center gap-3">
      {/* Playback controls */}
      <div className="flex items-center gap-1">
        <button
          className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded text-gray-700 disabled:opacity-30 border border-gray-200"
          disabled={!canStep || currentFrame <= 0}
          onClick={() => onSetFrame(Math.max(0, currentFrame - 1))}
          title="Step backward"
        >
          &lt;
        </button>

        <button
          className="px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded text-gray-700 disabled:opacity-30 min-w-[60px] border border-gray-200"
          disabled={totalFrames === 0}
          onClick={onTogglePlay}
        >
          {isPlaying ? "Pause" : "Play"}
        </button>

        <button
          className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded text-gray-700 disabled:opacity-30 border border-gray-200"
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
          className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-500"
          disabled={totalFrames === 0}
        />
      </div>

      {/* Frame counter */}
      <span className="text-xs text-gray-500 font-mono min-w-[80px] text-right">
        {currentFrame + 1} / {totalFrames}
      </span>

      {/* Speed selector */}
      <div className="flex items-center gap-1 ml-2">
        <span className="text-xs text-gray-500 mr-1">Speed:</span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            className={`px-2 py-1 text-xs rounded border ${
              speed === s
                ? "bg-indigo-100 text-indigo-700 border-indigo-300"
                : "bg-gray-100 hover:bg-gray-200 text-gray-500 border-gray-200"
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
