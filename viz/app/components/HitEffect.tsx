"use client";

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { HitEvent } from "../hooks/useMatch";

interface HitEffectProps {
  hit: HitEvent;
}

// 6 spark directions, ~60 degrees apart
const SPARK_ANGLES = [0, 1.05, 2.1, 3.14, 4.19, 5.24];

export default function HitEffect({ hit }: HitEffectProps) {
  const ring1Ref = useRef<THREE.Mesh>(null);
  const ring2Ref = useRef<THREE.Mesh>(null);
  const ring3Ref = useRef<THREE.Mesh>(null);
  const sparkRefs = useRef<(THREE.Mesh | null)[]>([]);
  const timeRef = useRef(0);

  useFrame((_state, delta) => {
    timeRef.current += delta;
    const t = timeRef.current;

    // Ring 1 — fastest expanding
    if (ring1Ref.current) {
      const s1 = 1 + t * 40;
      ring1Ref.current.scale.set(s1, s1, 1);
      const mat = ring1Ref.current.material as THREE.MeshStandardMaterial;
      mat.opacity = Math.max(0, 1 - t * 3);
    }

    // Ring 2 — medium speed
    if (ring2Ref.current) {
      const s2 = 0.5 + t * 28;
      ring2Ref.current.scale.set(s2, s2, 1);
      const mat = ring2Ref.current.material as THREE.MeshStandardMaterial;
      mat.opacity = Math.max(0, 0.8 - t * 2.5);
    }

    // Ring 3 — slowest, brightest
    if (ring3Ref.current) {
      const s3 = 0.3 + t * 18;
      ring3Ref.current.scale.set(s3, s3, 1);
      const mat = ring3Ref.current.material as THREE.MeshStandardMaterial;
      mat.opacity = Math.max(0, 0.6 - t * 2);
    }

    // Sparks — fly outward, decelerating
    sparkRefs.current.forEach((spark, i) => {
      if (spark) {
        const angle = SPARK_ANGLES[i];
        const dist = t * 60 * Math.max(0, 1 - t * 1.5);
        spark.position.x = Math.cos(angle) * dist;
        spark.position.y = Math.sin(angle) * dist;
        const mat = spark.material as THREE.MeshStandardMaterial;
        mat.opacity = Math.max(0, 1 - t * 3);
      }
    });
  });

  return (
    <group position={[hit.x, hit.y, 2]}>
      {/* Ring 1 — outer, orange */}
      <mesh ref={ring1Ref}>
        <ringGeometry args={[6, 8, 16]} />
        <meshStandardMaterial
          color="#ff8800"
          emissive="#ff6600"
          emissiveIntensity={2}
          transparent
          opacity={1}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Ring 2 — middle, yellow-orange */}
      <mesh ref={ring2Ref}>
        <ringGeometry args={[4, 6, 16]} />
        <meshStandardMaterial
          color="#ffaa00"
          emissive="#cc6600"
          emissiveIntensity={2}
          transparent
          opacity={0.8}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Ring 3 — inner, white-yellow flash */}
      <mesh ref={ring3Ref}>
        <ringGeometry args={[2, 4, 16]} />
        <meshStandardMaterial
          color="#ffdd44"
          emissive="#ffaa00"
          emissiveIntensity={2.5}
          transparent
          opacity={0.6}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Spark particles */}
      {SPARK_ANGLES.map((_, i) => (
        <mesh
          key={i}
          ref={(el) => {
            sparkRefs.current[i] = el;
          }}
        >
          <circleGeometry args={[1.5, 6]} />
          <meshStandardMaterial
            color="#ffcc00"
            emissive="#ff8800"
            emissiveIntensity={2}
            transparent
            opacity={1}
            depthWrite={false}
          />
        </mesh>
      ))}
    </group>
  );
}
