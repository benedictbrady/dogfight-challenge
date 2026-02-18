"use client";

import { Line } from "@react-three/drei";
import type { BulletState } from "../hooks/useMatch";

interface BulletProps {
  bullet: BulletState;
}

// Warm tracer colors — yellow-orange
const OWNER_COLORS = ["#e8c840", "#ff6830"];

export default function Bullet({ bullet }: BulletProps) {
  const color = OWNER_COLORS[bullet.owner] ?? "#ffcc00";
  const angle = Math.atan2(bullet.vy, bullet.vx);

  // Unit direction for tracer tail
  const speed = Math.sqrt(bullet.vx * bullet.vx + bullet.vy * bullet.vy);
  const dx = speed > 0 ? bullet.vx / speed : 1;
  const dy = speed > 0 ? bullet.vy / speed : 0;

  // Tail points trailing behind the bullet
  const tailPoints: [number, number, number][] = [
    [0, 0, 0],
    [-dx * 4, -dy * 4, 0],
    [-dx * 8, -dy * 8, 0],
    [-dx * 12, -dy * 12, 0],
  ];

  return (
    <group position={[bullet.x, bullet.y, 1]}>
      {/* Elongated tracer body */}
      <mesh rotation={[0, 0, angle]}>
        <planeGeometry args={[8, 2]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={1.5}
        />
      </mesh>
      {/* Glow halo */}
      <mesh rotation={[0, 0, angle]}>
        <planeGeometry args={[10, 4]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.8}
          transparent
          opacity={0.25}
        />
      </mesh>
      {/* Tracer tail */}
      <Line
        points={tailPoints}
        color={color}
        lineWidth={1.5}
        transparent
        opacity={0.4}
      />
    </group>
  );
}
