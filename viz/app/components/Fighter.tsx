"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import type { FighterState } from "../hooks/useMatch";

interface FighterProps {
  state: FighterState;
  color: string;
  label: string;
  trail: THREE.Vector3[];
}

const BG: [number, number, number] = [0xf8 / 255, 0xf8 / 255, 0xf8 / 255];
const FLASH_DECAY_RATE = 5;

// Smoke cloud circles for dead fighter display
const SMOKE_CLOUDS = [
  { position: [0, 0, 0] as const, radius: 8, segments: 20, color: "#9ca3af", opacity: 0.85 },
  { position: [-5, 0, 0.1] as const, radius: 5, segments: 16, color: "#b0b5bc", opacity: 0.7 },
  { position: [5, 0, 0.1] as const, radius: 5, segments: 16, color: "#b0b5bc", opacity: 0.7 },
  { position: [0, 5, 0.1] as const, radius: 5, segments: 16, color: "#b0b5bc", opacity: 0.7 },
  { position: [0, -5, 0.1] as const, radius: 5, segments: 16, color: "#b0b5bc", opacity: 0.7 },
];

// Debris fragment positions for explosion (pre-computed, scattered outward)
const DEBRIS = [
  { angle: 0, dist: 12, size: 2.5 },
  { angle: 0.9, dist: 16, size: 1.8 },
  { angle: 1.7, dist: 10, size: 3 },
  { angle: 2.5, dist: 18, size: 2 },
  { angle: 3.3, dist: 13, size: 2.2 },
  { angle: 4.2, dist: 15, size: 1.5 },
  { angle: 5.0, dist: 11, size: 2.8 },
  { angle: 5.8, dist: 17, size: 1.6 },
];

export default function Fighter({ state, color, trail }: FighterProps) {
  const matRef = useRef<THREE.MeshBasicMaterial>(null);
  const prevHpRef = useRef(state.hp);
  const flashRef = useRef(0);
  const teamColor = useMemo(() => new THREE.Color(color), [color]);
  const flashColor = useMemo(() => new THREE.Color("#ffffff"), []);

  // Detect HP drop → trigger flash
  useFrame((_, delta) => {
    if (state.hp < prevHpRef.current) {
      flashRef.current = 1;
    }
    prevHpRef.current = state.hp;

    if (flashRef.current > 0) {
      flashRef.current = Math.max(0, flashRef.current - delta * FLASH_DECAY_RATE);
      if (matRef.current) {
        matRef.current.color.copy(teamColor).lerp(flashColor, flashRef.current);
      }
    } else if (matRef.current) {
      matRef.current.color.copy(teamColor);
    }
  });

  const triangleShape = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(10, 0);
    s.lineTo(-6, 5);
    s.lineTo(-3, 0);
    s.lineTo(-6, -5);
    s.closePath();
    return s;
  }, []);

  // Trail with per-vertex fade, split at wrap boundaries
  const trailSegments = useMemo(() => {
    if (trail.length < 2) return null;
    const c = new THREE.Color(color);
    const cr = c.r, cg = c.g, cb = c.b;

    // Compute per-vertex color based on global position in trail
    const vertexColor = (globalIdx: number): [number, number, number] => {
      const t = trail.length > 1 ? globalIdx / (trail.length - 1) : 1;
      return [
        BG[0] + (cr - BG[0]) * t,
        BG[1] + (cg - BG[1]) * t,
        BG[2] + (cb - BG[2]) * t,
      ];
    };

    // Split trail at wrap points (|dx| > 500 between consecutive points)
    const segments: { points: [number, number, number][]; colors: [number, number, number][] }[] = [];
    let currentPts: [number, number, number][] = [[trail[0].x, trail[0].y, 0]];
    let currentColors: [number, number, number][] = [vertexColor(0)];

    for (let i = 1; i < trail.length; i++) {
      const dx = Math.abs(trail[i].x - trail[i - 1].x);
      if (dx > 500) {
        // Wrap detected — end current segment, start new one
        if (currentPts.length >= 2) {
          segments.push({ points: currentPts, colors: currentColors });
        }
        currentPts = [[trail[i].x, trail[i].y, 0]];
        currentColors = [vertexColor(i)];
      } else {
        currentPts.push([trail[i].x, trail[i].y, 0]);
        currentColors.push(vertexColor(i));
      }
    }

    if (currentPts.length >= 2) {
      segments.push({ points: currentPts, colors: currentColors });
    }

    return segments;
  }, [trail, color]);

  return (
    <group position={[state.x, state.y, 1]}>
      {state.alive ? (
        /* Alive: chevron with hit flash */
        <group rotation={[0, 0, state.yaw]}>
          <mesh>
            <shapeGeometry args={[triangleShape]} />
            <meshBasicMaterial ref={matRef} color={color} side={THREE.DoubleSide} />
          </mesh>
        </group>
      ) : (
        /* Dead: symmetric smoke cloud */
        <>
          {SMOKE_CLOUDS.map((cloud, i) => (
            <mesh key={i} position={cloud.position}>
              <circleGeometry args={[cloud.radius, cloud.segments]} />
              <meshBasicMaterial color={cloud.color} transparent opacity={cloud.opacity} />
            </mesh>
          ))}
        </>
      )}

      {/* Fading trail — persists after death, split at wrap boundaries */}
      {trailSegments && (
        <group position={[-state.x, -state.y, -1]}>
          {trailSegments.map((seg, i) => (
            <Line
              key={i}
              points={seg.points}
              vertexColors={seg.colors}
              lineWidth={1.5}
            />
          ))}
        </group>
      )}
    </group>
  );
}
