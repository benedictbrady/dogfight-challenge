"use client";

import { useMemo, useRef } from "react";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";

const ARENA_RADIUS = 500;
const MAX_ALTITUDE = 600;
const WIDE = 4000; // how far scenery extends horizontally

// ── Sky gradient shader ──────────────────────────────────────────────
const skyVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const skyFragmentShader = `
  varying vec2 vUv;
  void main() {
    // top = deeper blue, bottom = warm haze near horizon
    vec3 top    = vec3(0.38, 0.65, 0.88);   // #61a6e0
    vec3 mid    = vec3(0.66, 0.82, 0.93);   // #a8d1ee
    vec3 bottom = vec3(0.88, 0.90, 0.82);   // #e0e6d1 warm haze
    float t = vUv.y;
    vec3 col = mix(bottom, mid, smoothstep(0.0, 0.4, t));
    col = mix(col, top, smoothstep(0.4, 1.0, t));
    gl_FragColor = vec4(col, 1.0);
  }
`;

// ── Helpers ──────────────────────────────────────────────────────────

/** Build a mountain ridge silhouette as a THREE.Shape */
function makeMountainShape(
  peaks: { x: number; y: number }[],
  baseY: number,
  xMin: number,
  xMax: number,
): THREE.Shape {
  const shape = new THREE.Shape();
  shape.moveTo(xMin, baseY);
  // walk across peaks with slight curves
  for (let i = 0; i < peaks.length; i++) {
    const p = peaks[i];
    if (i === 0) {
      shape.lineTo(p.x, p.y);
    } else {
      const prev = peaks[i - 1];
      const cpx = (prev.x + p.x) / 2;
      shape.quadraticCurveTo(cpx, Math.max(prev.y, p.y) * 1.05, p.x, p.y);
    }
  }
  shape.lineTo(xMax, baseY);
  shape.lineTo(xMin, baseY);
  return shape;
}

/** Seeded pseudo-random for deterministic clouds/mountains */
function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

/** Build a soft cloud blob shape */
function makeCloudShape(
  cx: number,
  cy: number,
  w: number,
  h: number,
  rand: () => number,
): THREE.Shape {
  const shape = new THREE.Shape();
  const bumps = 5 + Math.floor(rand() * 4);
  for (let i = 0; i <= bumps; i++) {
    const t = i / bumps;
    const angle = t * Math.PI;
    const bx = cx - w / 2 + t * w;
    const by = cy + Math.sin(angle) * h * (0.6 + rand() * 0.4);
    if (i === 0) {
      shape.moveTo(bx, cy);
      shape.lineTo(bx, by);
    } else {
      const prevT = (i - 1) / bumps;
      const prevX = cx - w / 2 + prevT * w;
      const cpx = (prevX + bx) / 2;
      const cpy = by + h * 0.3 * rand();
      shape.quadraticCurveTo(cpx, cpy, bx, by);
    }
  }
  shape.lineTo(cx + w / 2, cy);
  shape.closePath();
  return shape;
}

// ── Component ────────────────────────────────────────────────────────

export default function Arena() {
  const cloudsRef = useRef<THREE.Group>(null);

  // Gentle cloud drift
  useFrame((_, delta) => {
    if (cloudsRef.current) {
      cloudsRef.current.position.x += delta * 8; // slow drift right
      if (cloudsRef.current.position.x > 200) {
        cloudsRef.current.position.x = -200;
      }
    }
  });

  // Altitude reference lines
  const altLines = useMemo(() => {
    const lines: [number, number, number][][] = [];
    for (let alt = 100; alt <= 500; alt += 100) {
      lines.push([
        [-ARENA_RADIUS, alt, -0.1],
        [ARENA_RADIUS, alt, -0.1],
      ]);
    }
    return lines;
  }, []);

  // Far mountains — tall, faded blue-grey
  const farMountains = useMemo(() => {
    const rand = seededRandom(42);
    const peaks: { x: number; y: number }[] = [];
    for (let x = -WIDE; x <= WIDE; x += 120 + rand() * 80) {
      const isPeak = rand() > 0.4;
      peaks.push({ x, y: isPeak ? 40 + rand() * 100 : 10 + rand() * 30 });
    }
    return makeMountainShape(peaks, -10, -WIDE, WIDE);
  }, []);

  // Near mountains — shorter, darker
  const nearMountains = useMemo(() => {
    const rand = seededRandom(99);
    const peaks: { x: number; y: number }[] = [];
    for (let x = -WIDE; x <= WIDE; x += 80 + rand() * 60) {
      const isPeak = rand() > 0.5;
      peaks.push({ x, y: isPeak ? 20 + rand() * 55 : 5 + rand() * 20 });
    }
    return makeMountainShape(peaks, -10, -WIDE, WIDE);
  }, []);

  // Clouds
  const clouds = useMemo(() => {
    const rand = seededRandom(77);
    const result: { shape: THREE.Shape; opacity: number; z: number }[] = [];
    for (let i = 0; i < 14; i++) {
      const cx = -WIDE * 0.6 + rand() * WIDE * 1.2;
      const cy = 250 + rand() * 340;
      const w = 80 + rand() * 160;
      const h = 20 + rand() * 35;
      result.push({
        shape: makeCloudShape(cx, cy, w, h, rand),
        opacity: 0.25 + rand() * 0.35,
        z: -2 - rand() * 2,
      });
    }
    return result;
  }, []);

  // Ground layers geometry
  const groundLayers = useMemo(() => {
    const rand = seededRandom(55);
    // Rolling hills on top of ground
    const hillShape = new THREE.Shape();
    hillShape.moveTo(-WIDE, -30);
    for (let x = -WIDE; x <= WIDE; x += 60 + rand() * 40) {
      const h = rand() * 12;
      hillShape.lineTo(x, h);
    }
    hillShape.lineTo(WIDE, -30);
    hillShape.lineTo(-WIDE, -30);
    return hillShape;
  }, []);

  return (
    <group>
      {/* ── Sky gradient ── */}
      <mesh position={[0, MAX_ALTITUDE / 2, -8]}>
        <planeGeometry args={[WIDE * 2, MAX_ALTITUDE + 200]} />
        <shaderMaterial
          vertexShader={skyVertexShader}
          fragmentShader={skyFragmentShader}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* ── Far mountains — misty blue-grey ── */}
      <mesh position={[0, 0, -6]}>
        <shapeGeometry args={[farMountains]} />
        <meshBasicMaterial
          color="#8a9daa"
          transparent
          opacity={0.35}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* ── Near mountains — olive/slate ── */}
      <mesh position={[0, 0, -5]}>
        <shapeGeometry args={[nearMountains]} />
        <meshBasicMaterial
          color="#5c6e55"
          transparent
          opacity={0.45}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* ── Clouds ── */}
      <group ref={cloudsRef}>
        {clouds.map((c, i) => (
          <mesh key={`cloud-${i}`} position={[0, 0, c.z]}>
            <shapeGeometry args={[c.shape]} />
            <meshBasicMaterial
              color="#ffffff"
              transparent
              opacity={c.opacity}
              side={THREE.DoubleSide}
              depthWrite={false}
            />
          </mesh>
        ))}
      </group>

      {/* ── Ground: rolling grass hills ── */}
      <mesh position={[0, 0, -3]}>
        <shapeGeometry args={[groundLayers]} />
        <meshBasicMaterial color="#4a7a2e" side={THREE.DoubleSide} />
      </mesh>

      {/* ── Ground: main grass strip ── */}
      <mesh position={[0, -6, -2.5]}>
        <planeGeometry args={[WIDE * 2, 12]} />
        <meshBasicMaterial color="#3d6b24" side={THREE.DoubleSide} />
      </mesh>

      {/* ── Ground: dark earth below grass ── */}
      <mesh position={[0, -25, -2.5]}>
        <planeGeometry args={[WIDE * 2, 30]} />
        <meshBasicMaterial color="#5a4a30" side={THREE.DoubleSide} />
      </mesh>

      {/* ── Below-ground fill — earthy brown ── */}
      <mesh position={[0, -2000, -3]}>
        <planeGeometry args={[WIDE * 2, 4000]} />
        <meshBasicMaterial color="#6b5d45" side={THREE.DoubleSide} />
      </mesh>

      {/* ── Altitude reference lines ── */}
      {altLines.map((pts, i) => (
        <Line
          key={`alt-${i}`}
          points={pts}
          color="#90c0d8"
          transparent
          opacity={0.08}
          lineWidth={1}
        />
      ))}

      {/* ── Arena side boundaries ── */}
      <Line
        points={[
          [-ARENA_RADIUS, 0, 0.1],
          [-ARENA_RADIUS, MAX_ALTITUDE, 0.1],
        ]}
        color="#7090a8"
        transparent
        opacity={0.2}
        lineWidth={1}
      />
      <Line
        points={[
          [ARENA_RADIUS, 0, 0.1],
          [ARENA_RADIUS, MAX_ALTITUDE, 0.1],
        ]}
        color="#7090a8"
        transparent
        opacity={0.2}
        lineWidth={1}
      />

      {/* ── Ceiling line ── */}
      <Line
        points={[
          [-ARENA_RADIUS, MAX_ALTITUDE, 0.1],
          [ARENA_RADIUS, MAX_ALTITUDE, 0.1],
        ]}
        color="#7090a8"
        transparent
        opacity={0.12}
        lineWidth={1}
      />
    </group>
  );
}
