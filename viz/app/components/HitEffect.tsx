"use client";

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { HitEvent } from "../hooks/useMatch";

const HIT_RING_SCALE_RATE = 50;
const HIT_EFFECT_DURATION = 0.3;

interface HitEffectProps {
  hit: HitEvent;
}

export default function HitEffect({ hit }: HitEffectProps) {
  const ringRef = useRef<THREE.Mesh>(null);
  const timeRef = useRef(0);

  useFrame((_state, delta) => {
    timeRef.current += delta;
    const t = timeRef.current;

    if (ringRef.current) {
      const s = 1 + t * HIT_RING_SCALE_RATE;
      ringRef.current.scale.set(s, s, 1);
      const mat = ringRef.current.material as THREE.MeshBasicMaterial;
      mat.opacity = Math.max(0, 1 - t / HIT_EFFECT_DURATION);
    }
  });

  return (
    <group position={[hit.x, hit.y, 2]}>
      <mesh ref={ringRef}>
        <ringGeometry args={[5, 7, 20]} />
        <meshBasicMaterial
          color="#f59e0b"
          transparent
          opacity={1}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}
