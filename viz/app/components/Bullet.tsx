"use client";

import type { BulletState } from "../hooks/useMatch";

interface BulletProps {
  bullet: BulletState;
}

const OWNER_COLORS = ["#2563eb", "#dc2626"];

export default function Bullet({ bullet }: BulletProps) {
  const color = OWNER_COLORS[bullet.owner] ?? "#6b7280";

  return (
    <group position={[bullet.x, bullet.y, 1]}>
      <mesh>
        <circleGeometry args={[3, 12]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </group>
  );
}
