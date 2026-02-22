"use client";

import { useMemo } from "react";
import { Line, Text } from "@react-three/drei";

const ARENA_MIN_X = -500;
const ARENA_MAX_X = 500;
const ARENA_MIN_Y = 0;
const ARENA_MAX_Y = 600;
const GRID_STEP = 100;
const GROUND_DEATH_Y = 5;
const CEILING_ZONE_Y = 550;

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

      {/* Ground death zone: solid red line at y=GROUND_DEATH_Y */}
      <Line
        points={[
          [ARENA_MIN_X, GROUND_DEATH_Y, 0],
          [ARENA_MAX_X, GROUND_DEATH_Y, 0],
        ]}
        color="#dc2626"
        lineWidth={3}
      />

      {/* Ground death zone: semi-transparent red fill below */}
      <mesh position={[0, (GROUND_DEATH_Y + ARENA_MIN_Y - 30) / 2, -0.5]}>
        <planeGeometry args={[ARENA_MAX_X - ARENA_MIN_X, GROUND_DEATH_Y - ARENA_MIN_Y + 30]} />
        <meshBasicMaterial color="#dc2626" transparent opacity={0.12} />
      </mesh>

      {/* Ceiling zone: semi-transparent blue/gray band */}
      <mesh position={[0, (CEILING_ZONE_Y + ARENA_MAX_Y) / 2, -0.5]}>
        <planeGeometry args={[ARENA_MAX_X - ARENA_MIN_X, ARENA_MAX_Y - CEILING_ZONE_Y]} />
        <meshBasicMaterial color="#6b7280" transparent opacity={0.08} />
      </mesh>

      {/* Wrap boundary indicators: subtle dashed vertical lines at x=±500 */}
      <Line
        points={[
          [ARENA_MIN_X, ARENA_MIN_Y, 0],
          [ARENA_MIN_X, ARENA_MAX_Y, 0],
        ]}
        color="#9ca3af"
        lineWidth={1}
        dashed
        dashSize={8}
        gapSize={12}
      />
      <Line
        points={[
          [ARENA_MAX_X, ARENA_MIN_Y, 0],
          [ARENA_MAX_X, ARENA_MAX_Y, 0],
        ]}
        color="#9ca3af"
        lineWidth={1}
        dashed
        dashSize={8}
        gapSize={12}
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
