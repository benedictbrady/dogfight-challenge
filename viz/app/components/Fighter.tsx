"use client";

import { useMemo } from "react";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import type { FighterState } from "../hooks/useMatch";

interface FighterProps {
  state: FighterState;
  color: string;
  trail: THREE.Vector3[];
}

export default function Fighter({ state, color, trail }: FighterProps) {
  // When facing left, mirror horizontally and adjust rotation so wings stay on top
  const facingLeft = Math.cos(state.yaw) < 0;
  const adjustedYaw = facingLeft ? Math.PI - state.yaw : state.yaw;
  const scaleX = facingLeft ? -1 : 1;
  const rotation = useMemo(
    (): [number, number, number] => [0, 0, adjustedYaw],
    [adjustedYaw]
  );

  // Trail points — convert to relative coords (subtract current fighter position)
  const trailPoints =
    trail.length >= 2
      ? trail.map(
          (v): [number, number, number] => [v.x - state.x, v.y - state.y, 0]
        )
      : null;

  // --- Shape definitions ---
  const fuselage = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-13.4, 0);
    s.quadraticCurveTo(-12.8, 1.2, -10.2, 1.8);
    s.lineTo(-2.5, 2.2);
    s.quadraticCurveTo(4.8, 2.0, 8.7, 1.2);
    s.quadraticCurveTo(11.7, 0.4, 12.6, 0);
    s.quadraticCurveTo(11.7, -0.4, 8.7, -1.2);
    s.quadraticCurveTo(4.8, -2.0, -2.5, -2.2);
    s.lineTo(-10.2, -1.8);
    s.quadraticCurveTo(-12.8, -1.2, -13.4, 0);
    s.closePath();
    return s;
  }, []);

  const fuselageOutline = useMemo(() => {
    const s = new THREE.Shape();
    const o = 0.5;
    s.moveTo(-13.4 - o, 0);
    s.quadraticCurveTo(-12.8 - o, 1.2 + o, -10.2, 1.8 + o);
    s.lineTo(-2.5, 2.2 + o);
    s.quadraticCurveTo(4.8 + o, 2.0 + o, 8.9 + o, 1.3 + o);
    s.quadraticCurveTo(12.2 + o, 0.45 + o, 13.2 + o, 0);
    s.quadraticCurveTo(12.2 + o, -0.45 - o, 8.9 + o, -1.3 - o);
    s.quadraticCurveTo(4.8 + o, -2.0 - o, -2.5, -2.2 - o);
    s.lineTo(-10.2, -1.8 - o);
    s.quadraticCurveTo(-12.8 - o, -1.2 - o, -13.4 - o, 0);
    s.closePath();
    return s;
  }, []);

  const cockpit = useMemo(() => {
    const s = new THREE.Shape();
    s.absellipse(0, 0, 1.3, 0.62, 0, Math.PI * 2, false, 0);
    s.closePath();
    return s;
  }, []);

  const topWing = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-3.6, 0);
    s.quadraticCurveTo(-3.9, 0.05, -4.0, 0.25);
    s.lineTo(-4.0, 0.55);
    s.quadraticCurveTo(-3.9, 0.8, -3.6, 0.88);
    s.lineTo(4.8, 0.88);
    s.quadraticCurveTo(5.1, 0.8, 5.2, 0.55);
    s.lineTo(5.2, 0.25);
    s.quadraticCurveTo(5.1, 0.05, 4.8, 0);
    s.closePath();
    return s;
  }, []);

  const bottomWing = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-3.0, 0);
    s.quadraticCurveTo(-3.3, 0.04, -3.4, 0.22);
    s.lineTo(-3.4, 0.46);
    s.quadraticCurveTo(-3.3, 0.66, -3.0, 0.72);
    s.lineTo(4.2, 0.72);
    s.quadraticCurveTo(4.5, 0.66, 4.6, 0.46);
    s.lineTo(4.6, 0.22);
    s.quadraticCurveTo(4.5, 0.04, 4.2, 0);
    s.closePath();
    return s;
  }, []);

  const tailPlane = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-11.8, 0.1);
    s.quadraticCurveTo(-14.3, 0.2, -15.1, 1.2);
    s.lineTo(-15.1, 2.0);
    s.quadraticCurveTo(-14.3, 3.0, -11.8, 3.1);
    s.lineTo(-9.4, 3.1);
    s.quadraticCurveTo(-9.1, 1.7, -9.4, 0.1);
    s.closePath();
    return s;
  }, []);

  const rudder = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-12.2, 2.5);
    s.quadraticCurveTo(-12.6, 4.8, -10.7, 5.7);
    s.lineTo(-9.8, 5.4);
    s.quadraticCurveTo(-9.9, 3.9, -9.5, 2.5);
    s.closePath();
    return s;
  }, []);

  const nosePanel = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(7.4, -1.8);
    s.lineTo(9.7, -1.8);
    s.lineTo(9.7, 1.8);
    s.lineTo(7.4, 1.8);
    s.closePath();
    return s;
  }, []);

  const tailBand = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-8.5, -1.8);
    s.lineTo(-7.0, -1.8);
    s.lineTo(-7.0, 1.8);
    s.lineTo(-8.5, 1.8);
    s.closePath();
    return s;
  }, []);

  // --- Color tiers ---
  const fuselageColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.multiplyScalar(0.72);
    return `#${c.getHexString()}`;
  }, [color]);

  const wingColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.multiplyScalar(0.92);
    return `#${c.getHexString()}`;
  }, [color]);

  const wingShadowColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.multiplyScalar(0.5);
    return `#${c.getHexString()}`;
  }, [color]);

  const panelColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.multiplyScalar(0.4);
    return `#${c.getHexString()}`;
  }, [color]);

  const inkColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.multiplyScalar(0.22);
    return `#${c.getHexString()}`;
  }, [color]);

  const outlineColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.multiplyScalar(0.2);
    return `#${c.getHexString()}`;
  }, [color]);

  const accentColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.lerp(new THREE.Color("#e6c26a"), 0.35);
    return `#${c.getHexString()}`;
  }, [color]);

  if (!state.alive) {
    // Wreckage — small X with smoke
    return (
      <group position={[state.x, state.y, 1]} scale={[1.75, 1.75, 1]}>
        <Line
          points={[
            [-5, -5, 0],
            [5, 5, 0],
          ]}
          color="#3a2a1a"
          lineWidth={2}
          opacity={0.6}
          transparent
        />
        <Line
          points={[
            [-5, 5, 0],
            [5, -5, 0],
          ]}
          color="#3a2a1a"
          lineWidth={2}
          opacity={0.6}
          transparent
        />
        <mesh>
          <circleGeometry args={[4, 12]} />
          <meshBasicMaterial color="#1a1008" transparent opacity={0.4} />
        </mesh>
      </group>
    );
  }

  return (
    <group position={[state.x, state.y, 1]}>
      {/* Plane body — scaleX only applies here */}
      <group scale={[1.75 * scaleX, 1.75, 1]}>
        <group rotation={rotation}>
          {/* Lower wing */}
          <mesh position={[0.5, -1.55, 0.08]}>
            <shapeGeometry args={[bottomWing]} />
            <meshBasicMaterial color={wingShadowColor} side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[0.62, -1.42, 0.1]}>
            <shapeGeometry args={[bottomWing]} />
            <meshBasicMaterial color={wingColor} side={THREE.DoubleSide} />
          </mesh>

          {/* Fuselage body */}
          <mesh position={[0, 0, 0.14]}>
            <shapeGeometry args={[fuselageOutline]} />
            <meshBasicMaterial color={outlineColor} side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[0, 0, 0.16]}>
            <shapeGeometry args={[fuselage]} />
            <meshBasicMaterial color={fuselageColor} side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[0, 0, 0.18]}>
            <shapeGeometry args={[nosePanel]} />
            <meshBasicMaterial color={panelColor} side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[0, 0, 0.2]}>
            <shapeGeometry args={[tailBand]} />
            <meshBasicMaterial color={accentColor} side={THREE.DoubleSide} />
          </mesh>
          <Line
            points={[
              [-11.8, 0, 0.22],
              [9.2, 0, 0.22],
            ]}
            color={inkColor}
            lineWidth={1.4}
          />

          {/* Tail section */}
          <mesh position={[0, -0.3, 0.24]}>
            <shapeGeometry args={[tailPlane]} />
            <meshBasicMaterial color={wingColor} side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[0, -0.15, 0.26]}>
            <shapeGeometry args={[rudder]} />
            <meshBasicMaterial color={panelColor} side={THREE.DoubleSide} />
          </mesh>

          {/* Upper wing and cabane struts */}
          <mesh position={[0.25, 1.35, 0.3]}>
            <shapeGeometry args={[topWing]} />
            <meshBasicMaterial color={wingShadowColor} side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[0.38, 1.48, 0.32]}>
            <shapeGeometry args={[topWing]} />
            <meshBasicMaterial color={wingColor} side={THREE.DoubleSide} />
          </mesh>
          <Line
            points={[
              [-0.95, -0.95, 0.24],
              [-0.95, 1.9, 0.24],
            ]}
            color="#5c4a32"
            lineWidth={2}
          />
          <Line
            points={[
              [2.25, -0.95, 0.24],
              [2.25, 1.9, 0.24],
            ]}
            color="#5c4a32"
            lineWidth={2}
          />
          <Line
            points={[
              [-0.95, -0.95, 0.25],
              [2.25, 1.9, 0.25],
            ]}
            color="#8e7f66"
            lineWidth={0.6}
            transparent
            opacity={0.6}
          />
          <Line
            points={[
              [2.25, -0.95, 0.25],
              [-0.95, 1.9, 0.25],
            ]}
            color="#8e7f66"
            lineWidth={0.6}
            transparent
            opacity={0.6}
          />

          {/* Canopy and pilot */}
          <mesh position={[0.8, 0.58, 0.34]}>
            <shapeGeometry args={[cockpit]} />
            <meshBasicMaterial color="#24180f" side={THREE.DoubleSide} />
          </mesh>

          {/* Landing gear */}
          <Line
            points={[
              [0.65, -2.35, 0.15],
              [3.0, -5.95, 0.15],
            ]}
            color="#5c4a32"
            lineWidth={1.5}
          />
          <Line
            points={[
              [-0.95, -2.35, 0.15],
              [1.2, -5.95, 0.15],
            ]}
            color="#5c4a32"
            lineWidth={1.5}
          />
          <Line
            points={[
              [1.2, -5.95, 0.16],
              [3.0, -5.95, 0.16],
            ]}
            color="#5c4a32"
            lineWidth={1.6}
          />
          <mesh position={[3.0, -5.95, 0.17]}>
            <circleGeometry args={[1.02, 16]} />
            <meshBasicMaterial color="#2d1f12" side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[1.2, -5.95, 0.17]}>
            <circleGeometry args={[1.02, 16]} />
            <meshBasicMaterial color="#2d1f12" side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[3.0, -5.95, 0.18]}>
            <circleGeometry args={[0.7, 14]} />
            <meshBasicMaterial color="#5c4a32" side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[1.2, -5.95, 0.18]}>
            <circleGeometry args={[0.7, 14]} />
            <meshBasicMaterial color="#5c4a32" side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[3.0, -5.95, 0.19]}>
            <circleGeometry args={[0.32, 10]} />
            <meshBasicMaterial color="#8f7652" side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[1.2, -5.95, 0.19]}>
            <circleGeometry args={[0.32, 10]} />
            <meshBasicMaterial color="#8f7652" side={THREE.DoubleSide} />
          </mesh>

          {/* Nose and prop */}
          <mesh position={[11.1, 0, 0.4]}>
            <ringGeometry args={[0.9, 1.4, 16]} />
            <meshBasicMaterial color="#8f7652" side={THREE.DoubleSide} />
          </mesh>
          <group position={[11.55, 0, 0.42]} rotation={[0, 0, Math.PI / 8]}>
            <mesh>
              <planeGeometry args={[0.65, 6.3]} />
              <meshBasicMaterial
                color="#4a3a20"
                side={THREE.DoubleSide}
                transparent
                opacity={0.85}
              />
            </mesh>
            <mesh rotation={[0, 0, Math.PI / 2]}>
              <planeGeometry args={[0.65, 6.3]} />
              <meshBasicMaterial
                color="#4a3a20"
                side={THREE.DoubleSide}
                transparent
                opacity={0.85}
              />
            </mesh>
          </group>
          <mesh position={[11.55, 0, 0.43]}>
            <circleGeometry args={[0.58, 12]} />
            <meshBasicMaterial color="#4a3a20" side={THREE.DoubleSide} />
          </mesh>
        </group>
      </group>

      {/* Smoke trail — relative coords, no transform */}
      {trailPoints && (
        <Line
          points={trailPoints}
          color="#c8c0a8"
          transparent
          opacity={0.3}
          lineWidth={2}
        />
      )}
    </group>
  );
}
