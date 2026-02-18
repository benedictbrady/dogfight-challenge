"use client";

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { HitEvent } from "../hooks/useMatch";

interface HitEffectProps {
  hit: HitEvent;
}

export default function HitEffect({ hit }: HitEffectProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const scaleRef = useRef(1);

  useFrame((_state, delta) => {
    if (meshRef.current) {
      scaleRef.current += delta * 35;
      const s = scaleRef.current;
      meshRef.current.scale.set(s, s, s);
      const mat = meshRef.current.material as THREE.MeshStandardMaterial;
      mat.opacity = Math.max(0, 1 - s / 12);
    }
  });

  return (
    <mesh ref={meshRef} position={[hit.x, hit.y, 2]}>
      <circleGeometry args={[8, 12]} />
      <meshStandardMaterial
        color="#ff8800"
        emissive="#cc4400"
        emissiveIntensity={2.5}
        transparent
        opacity={1}
        depthWrite={false}
      />
    </mesh>
  );
}
