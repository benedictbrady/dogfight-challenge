"use client";

import type { BulletState } from "../hooks/useMatch";

interface BulletProps {
  bullet: BulletState;
}

// Warm tracer colors — yellow-orange
const OWNER_COLORS = ["#e8c840", "#ff6830"];

export default function Bullet({ bullet }: BulletProps) {
  const color = OWNER_COLORS[bullet.owner] ?? "#ffcc00";

  return (
    <mesh position={[bullet.x, bullet.y, 1]}>
      <circleGeometry args={[4, 8]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={2}
      />
    </mesh>
  );
}
