"use client";

import { useMemo } from "react";
import { Line, Text } from "@react-three/drei";

const ARENA_MIN_X = -500;
const ARENA_MAX_X = 500;
const ARENA_MIN_Y = 0;
const ARENA_MAX_Y = 600;
const GRID_STEP = 100;

export default function Arena() {
  // Grid lines
  const gridLines = useMemo(() => {
    const lines: { points: [number, number, number][]; key: string }[] = [];
    // Vertical lines
    for (let x = ARENA_MIN_X; x <= ARENA_MAX_X; x += GRID_STEP) {
      lines.push({
        key: `v${x}`,
        points: [
          [x, ARENA_MIN_Y, -1],
          [x, ARENA_MAX_Y, -1],
        ],
      });
    }
    // Horizontal lines
    for (let y = ARENA_MIN_Y; y <= ARENA_MAX_Y; y += GRID_STEP) {
      lines.push({
        key: `h${y}`,
        points: [
          [ARENA_MIN_X, y, -1],
          [ARENA_MAX_X, y, -1],
        ],
      });
    }
    return lines;
  }, []);

  // Coordinate labels
  const xLabels = useMemo(() => {
    const labels: { x: number; text: string }[] = [];
    for (let x = ARENA_MIN_X; x <= ARENA_MAX_X; x += 200) {
      labels.push({ x, text: `${x}` });
    }
    return labels;
  }, []);

  const yLabels = useMemo(() => {
    const labels: { y: number; text: string }[] = [];
    for (let y = ARENA_MIN_Y; y <= ARENA_MAX_Y; y += 200) {
      labels.push({ y, text: `${y}` });
    }
    return labels;
  }, []);

  // Arena boundary as dashed rectangle
  const boundaryPoints: [number, number, number][] = [
    [ARENA_MIN_X, ARENA_MIN_Y, 0],
    [ARENA_MAX_X, ARENA_MIN_Y, 0],
    [ARENA_MAX_X, ARENA_MAX_Y, 0],
    [ARENA_MIN_X, ARENA_MAX_Y, 0],
    [ARENA_MIN_X, ARENA_MIN_Y, 0],
  ];

  return (
    <group>
      {/* Grid */}
      {gridLines.map((line) => (
        <Line
          key={line.key}
          points={line.points}
          color="#e5e7eb"
          lineWidth={1}
        />
      ))}

      {/* Dashed arena boundary */}
      <Line
        points={boundaryPoints}
        color="#9ca3af"
        lineWidth={1.5}
        dashed
        dashSize={12}
        gapSize={8}
      />

      {/* X-axis labels along bottom */}
      {xLabels.map((l) => (
        <Text
          key={`xl-${l.x}`}
          position={[l.x, ARENA_MIN_Y - 20, 0]}
          fontSize={12}
          color="#9ca3af"
          anchorX="center"
          anchorY="top"
        >
          {l.text}
        </Text>
      ))}

      {/* Y-axis labels along left */}
      {yLabels.map((l) => (
        <Text
          key={`yl-${l.y}`}
          position={[ARENA_MIN_X - 25, l.y, 0]}
          fontSize={12}
          color="#9ca3af"
          anchorX="right"
          anchorY="middle"
        >
          {l.text}
        </Text>
      ))}
    </group>
  );
}
