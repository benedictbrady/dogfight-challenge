"use client";

import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import Arena from "./Arena";
import Fighter from "./Fighter";
import Bullet from "./Bullet";
import HitEffect from "./HitEffect";
import type { Frame } from "../hooks/useMatch";

interface SceneProps {
  frame: Frame | null;
  trailFrames: Frame[];
}

function SceneContent({ frame, trailFrames }: SceneProps) {
  const trails = useMemo(() => {
    const t0: THREE.Vector3[] = [];
    const t1: THREE.Vector3[] = [];
    for (const f of trailFrames) {
      t0.push(new THREE.Vector3(f.fighters[0].x, f.fighters[0].y, 0));
      t1.push(new THREE.Vector3(f.fighters[1].x, f.fighters[1].y, 0));
    }
    return [t0, t1];
  }, [trailFrames]);

  return (
    <>
      <color attach="background" args={["#f8f8f8"]} />

      <Arena />

      {frame && (
        <Fighter
          state={frame.fighters[0]}
          color="#2563eb"
          label="P0"
          trail={trails[0]}
        />
      )}
      {frame && (
        <Fighter
          state={frame.fighters[1]}
          color="#dc2626"
          label="P1"
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

export default function Scene({ frame, trailFrames }: SceneProps) {
  return (
    <Canvas
      orthographic
      onCreated={({ camera }) => {
        camera.lookAt(0, 300, 0);
        camera.updateProjectionMatrix();
      }}
      camera={{
        position: [0, 300, 500],
        zoom: 0.95,
        near: 1,
        far: 1000,
        up: [0, 1, 0],
      }}
      style={{ width: "100%", height: "100%" }}
    >
      <SceneContent
        frame={frame}
        trailFrames={trailFrames}
      />
    </Canvas>
  );
}
