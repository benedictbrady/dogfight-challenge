"use client";

import { useMemo } from "react";
import { Canvas, extend } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import Arena from "./Arena";
import Fighter from "./Fighter";
import Bullet from "./Bullet";
import HitEffect from "./HitEffect";
import type { Frame } from "../hooks/useMatch";

extend({
  Mesh: THREE.Mesh,
  Group: THREE.Group,
  AmbientLight: THREE.AmbientLight,
  DirectionalLight: THREE.DirectionalLight,
  Color: THREE.Color,
  BoxGeometry: THREE.BoxGeometry,
  SphereGeometry: THREE.SphereGeometry,
  CircleGeometry: THREE.CircleGeometry,
  PlaneGeometry: THREE.PlaneGeometry,
  BufferGeometry: THREE.BufferGeometry,
  ShapeGeometry: THREE.ShapeGeometry,
  MeshStandardMaterial: THREE.MeshStandardMaterial,
  MeshBasicMaterial: THREE.MeshBasicMaterial,
  LineBasicMaterial: THREE.LineBasicMaterial,
  LineSegments: THREE.LineSegments,
});

interface SceneProps {
  frame: Frame | null;
  cameraMode: "free" | "chase0" | "chase1";
  trailFrames: Frame[];
}

function SceneContent({ frame, trailFrames }: SceneProps) {
  const trails = useMemo(() => {
    const t0: THREE.Vector3[] = [];
    const t1: THREE.Vector3[] = [];
    for (const f of trailFrames) {
      if (f.fighters[0]) {
        t0.push(new THREE.Vector3(f.fighters[0].x, f.fighters[0].y, 0));
      }
      if (f.fighters[1]) {
        t1.push(new THREE.Vector3(f.fighters[1].x, f.fighters[1].y, 0));
      }
    }
    return [t0, t1];
  }, [trailFrames]);

  return (
    <>
      {/* Warm lighting — like a hazy afternoon */}
      <ambientLight intensity={0.7} color="#f5e6c8" />
      <directionalLight position={[0, 300, 500]} intensity={0.4} color="#fff8e0" />

      {/* Background — matches sky gradient top edge */}
      <color attach="background" args={["#61a6e0"]} />

      {/* Pan/zoom only — target center of arena so the full play area fills the screen */}
      <OrbitControls
        enableRotate={false}
        enableDamping
        dampingFactor={0.15}
        maxDistance={2000}
        minDistance={100}
        target={[0, 300, 0]}
      />

      <Arena />

      {/* P0 — Allied tan/olive biplane */}
      {frame && frame.fighters[0] && (
        <Fighter
          state={frame.fighters[0]}
          color="#c4a050"
          trail={trails[0]}
        />
      )}
      {/* P1 — Red Baron */}
      {frame && frame.fighters[1] && (
        <Fighter
          state={frame.fighters[1]}
          color="#b83030"
          trail={trails[1]}
        />
      )}

      {frame &&
        frame.bullets.map((b, i) => <Bullet key={i} bullet={b} />)}

      {frame &&
        frame.hits.map((h, i) => (
          <HitEffect key={`hit-${frame.tick}-${i}`} hit={h} />
        ))}
    </>
  );
}

export default function Scene({ frame, cameraMode, trailFrames }: SceneProps) {
  return (
    <Canvas
      orthographic
      camera={{
        position: [0, 300, 500],
        zoom: 1.1,
        near: 1,
        far: 1000,
        up: [0, 1, 0],
      }}
      style={{ width: "100%", height: "100%" }}
    >
      <SceneContent
        frame={frame}
        cameraMode={cameraMode}
        trailFrames={trailFrames}
      />
    </Canvas>
  );
}
