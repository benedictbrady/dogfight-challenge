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
      flashRef.current = Math.max(0, flashRef.current - delta * 5); // ~0.2s flash
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

  // Trail with per-vertex fade
  const { trailPoints, trailColors } = useMemo(() => {
    if (trail.length < 2) return { trailPoints: null, trailColors: null };
    const pts = trail.map((v): [number, number, number] => [v.x, v.y, 0]);
    const c = new THREE.Color(color);
    const cr = c.r, cg = c.g, cb = c.b;
    const colors = trail.map((_, i) => {
      const t = trail.length > 1 ? i / (trail.length - 1) : 1;
      return [
        BG[0] + (cr - BG[0]) * t,
        BG[1] + (cg - BG[1]) * t,
        BG[2] + (cb - BG[2]) * t,
      ] as [number, number, number];
    });
    return { trailPoints: pts, trailColors: colors };
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
          <mesh position={[0, 0, 0]}>
            <circleGeometry args={[8, 20]} />
            <meshBasicMaterial color="#9ca3af" transparent opacity={0.85} />
          </mesh>
          <mesh position={[-5, 0, 0.1]}>
            <circleGeometry args={[5, 16]} />
            <meshBasicMaterial color="#b0b5bc" transparent opacity={0.7} />
          </mesh>
          <mesh position={[5, 0, 0.1]}>
            <circleGeometry args={[5, 16]} />
            <meshBasicMaterial color="#b0b5bc" transparent opacity={0.7} />
          </mesh>
          <mesh position={[0, 5, 0.1]}>
            <circleGeometry args={[5, 16]} />
            <meshBasicMaterial color="#b0b5bc" transparent opacity={0.7} />
          </mesh>
          <mesh position={[0, -5, 0.1]}>
            <circleGeometry args={[5, 16]} />
            <meshBasicMaterial color="#b0b5bc" transparent opacity={0.7} />
          </mesh>
        </>
      )}

      {/* Fading trail — persists after death */}
      {trailPoints && trailColors && (
        <group position={[-state.x, -state.y, -1]}>
          <Line
            points={trailPoints}
            vertexColors={trailColors}
            lineWidth={1.5}
          />
        </group>
      )}
    </group>
  );
}
