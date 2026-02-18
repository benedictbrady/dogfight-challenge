"use client";

import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { Text, Line } from "@react-three/drei";
import type { FighterState } from "../hooks/useMatch";

interface FighterProps {
  state: FighterState;
  color: string;
  label: string;
  trail: THREE.Vector3[];
}

export default function Fighter({ state, color, label, trail }: FighterProps) {
  const groupRef = useRef<THREE.Group>(null);
  const propRef = useRef<THREE.Group>(null);

  // When facing left, mirror horizontally and adjust rotation so wings stay on top
  const facingLeft = Math.cos(state.yaw) < 0;
  const adjustedYaw = facingLeft ? Math.PI - state.yaw : state.yaw;
  const scaleX = facingLeft ? -1 : 1;
  const rotation = useMemo(
    (): [number, number, number] => [0, 0, adjustedYaw],
    [adjustedYaw]
  );

  // Spin the propeller
  useFrame((_, delta) => {
    if (propRef.current && state.alive) {
      propRef.current.rotation.z += 25 * delta;
    }
  });

  // Trail points — convert to relative coords (subtract current fighter position)
  const trailPoints =
    trail.length >= 2
      ? trail.map(
          (v): [number, number, number] => [v.x - state.x, v.y - state.y, 0]
        )
      : null;

  // --- Shape definitions ---

  // Fuselage — aerodynamic body: rounded nose, widest at cockpit, tapering to tail
  const fuselage = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(9, 0);
    s.quadraticCurveTo(9.5, 1.8, 6, 2.0);
    s.lineTo(2, 2.2);
    s.lineTo(-4, 1.8);
    s.lineTo(-7, 1.2);
    s.quadraticCurveTo(-9.5, 0.6, -10, 0.3);
    s.lineTo(-10, -0.3);
    s.quadraticCurveTo(-9.5, -0.6, -7, -1.2);
    s.lineTo(-4, -1.8);
    s.lineTo(2, -2.2);
    s.lineTo(6, -2.0);
    s.quadraticCurveTo(9.5, -1.8, 9, 0);
    return s;
  }, []);

  // Fuselage outline — slightly larger for stroke effect
  const fuselageOutline = useMemo(() => {
    const s = new THREE.Shape();
    const o = 0.4;
    s.moveTo(9 + o, 0);
    s.quadraticCurveTo(9.5 + o, 1.8 + o, 6, 2.0 + o);
    s.lineTo(2, 2.2 + o);
    s.lineTo(-4, 1.8 + o);
    s.lineTo(-7, 1.2 + o);
    s.quadraticCurveTo(-9.5 - o, 0.6 + o, -10 - o, 0.3 + o);
    s.lineTo(-10 - o, -0.3 - o);
    s.quadraticCurveTo(-9.5 - o, -0.6 - o, -7, -1.2 - o);
    s.lineTo(-4, -1.8 - o);
    s.lineTo(2, -2.2 - o);
    s.lineTo(6, -2.0 - o);
    s.quadraticCurveTo(9.5 + o, -1.8 - o, 9 + o, 0);
    return s;
  }, []);

  // Engine cowling — darker overlay on nose
  const cowling = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(9, 0);
    s.quadraticCurveTo(9.5, 1.8, 6, 2.0);
    s.lineTo(5, 2.0);
    s.lineTo(5, -2.0);
    s.lineTo(6, -2.0);
    s.quadraticCurveTo(9.5, -1.8, 9, 0);
    return s;
  }, []);

  // Upper wing — wider, airfoil shape with rounded tips
  const upperWing = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-5, 0);
    s.quadraticCurveTo(-5, 0.6, -4.5, 0.8);
    s.lineTo(6, 1.2);
    s.quadraticCurveTo(7, 1.2, 7, 0.6);
    s.lineTo(7, 0);
    s.quadraticCurveTo(7, -0.3, 6, -0.3);
    s.lineTo(-4.5, -0.3);
    s.quadraticCurveTo(-5, -0.3, -5, 0);
    return s;
  }, []);

  // Lower wing — slightly shorter
  const lowerWing = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-4, 0);
    s.quadraticCurveTo(-4, 0.5, -3.5, 0.7);
    s.lineTo(5, 1.0);
    s.quadraticCurveTo(6, 1.0, 6, 0.5);
    s.lineTo(6, 0);
    s.quadraticCurveTo(6, -0.3, 5, -0.3);
    s.lineTo(-3.5, -0.3);
    s.quadraticCurveTo(-4, -0.3, -4, 0);
    return s;
  }, []);

  // Tail fin — curved vertical rudder
  const tailFin = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-8, 1.0);
    s.quadraticCurveTo(-9, 3.5, -10.5, 4.2);
    s.lineTo(-11, 4.0);
    s.quadraticCurveTo(-10.5, 2.5, -10, 1.5);
    s.lineTo(-9, 0.5);
    s.closePath();
    return s;
  }, []);

  // Horizontal stabilizer — tail airfoil
  const tailStab = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(-8, 0.2);
    s.lineTo(-12, 0.8);
    s.quadraticCurveTo(-12.5, 0.6, -12.5, 0.4);
    s.lineTo(-12, 0);
    s.lineTo(-8, -0.3);
    s.closePath();
    return s;
  }, []);

  // Windscreen — angled cockpit glass
  const windscreen = useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(2, 2.2);
    s.lineTo(3.5, 4.0);
    s.lineTo(1.5, 4.0);
    s.lineTo(0.5, 2.2);
    s.closePath();
    return s;
  }, []);

  // --- Color tiers ---
  const fuselageColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.multiplyScalar(0.75);
    return `#${c.getHexString()}`;
  }, [color]);

  const darkColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.multiplyScalar(0.5);
    return `#${c.getHexString()}`;
  }, [color]);

  const cowlingColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.multiplyScalar(0.45);
    return `#${c.getHexString()}`;
  }, [color]);

  const outlineColor = useMemo(() => {
    const c = new THREE.Color(color);
    c.multiplyScalar(0.25);
    return `#${c.getHexString()}`;
  }, [color]);

  const hpSegments = [0, 1, 2];

  if (!state.alive) {
    // Wreckage — small X with smoke
    return (
      <group position={[state.x, state.y, 1]} scale={[3.5, 3.5, 1]}>
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
      <group scale={[3.5 * scaleX, 3.5, 1]}>
        <group ref={groupRef} rotation={rotation}>
          {/* Landing gear */}
          {/* Gear struts */}
          <Line
            points={[
              [2, -2.2, 0.10],
              [3.5, -5.5, 0.10],
            ]}
            color="#5c4a32"
            lineWidth={1.5}
          />
          <Line
            points={[
              [0, -2.2, 0.10],
              [3.5, -5.5, 0.10],
            ]}
            color="#5c4a32"
            lineWidth={1.5}
          />
          <Line
            points={[
              [2, -2.2, 0.10],
              [0.5, -5.5, 0.10],
            ]}
            color="#5c4a32"
            lineWidth={1.5}
          />
          <Line
            points={[
              [0, -2.2, 0.10],
              [0.5, -5.5, 0.10],
            ]}
            color="#5c4a32"
            lineWidth={1.5}
          />
          {/* Axle */}
          <Line
            points={[
              [0.5, -5.5, 0.11],
              [3.5, -5.5, 0.11],
            ]}
            color="#5c4a32"
            lineWidth={1.5}
          />
          {/* Wheels */}
          <mesh position={[3.5, -5.5, 0.12]}>
            <circleGeometry args={[1.0, 10]} />
            <meshBasicMaterial color="#2a2015" side={THREE.DoubleSide} />
          </mesh>
          <mesh position={[0.5, -5.5, 0.12]}>
            <circleGeometry args={[1.0, 10]} />
            <meshBasicMaterial color="#2a2015" side={THREE.DoubleSide} />
          </mesh>

          {/* Lower wing */}
          <mesh position={[0, -3.0, 0.15]}>
            <shapeGeometry args={[lowerWing]} />
            <meshBasicMaterial color={color} side={THREE.DoubleSide} />
          </mesh>

          {/* Fuselage outline (behind body) */}
          <mesh position={[0, 0, 0.18]}>
            <shapeGeometry args={[fuselageOutline]} />
            <meshBasicMaterial color={outlineColor} side={THREE.DoubleSide} />
          </mesh>

          {/* Fuselage body */}
          <mesh position={[0, 0, 0.20]}>
            <shapeGeometry args={[fuselage]} />
            <meshBasicMaterial color={fuselageColor} side={THREE.DoubleSide} />
          </mesh>

          {/* Engine cowling */}
          <mesh position={[0, 0, 0.22]}>
            <shapeGeometry args={[cowling]} />
            <meshBasicMaterial color={cowlingColor} side={THREE.DoubleSide} />
          </mesh>

          {/* Horizontal stabilizer */}
          <mesh position={[0, 0, 0.24]}>
            <shapeGeometry args={[tailStab]} />
            <meshBasicMaterial
              color={color}
              side={THREE.DoubleSide}
              transparent
              opacity={0.85}
            />
          </mesh>

          {/* Cross-wire rigging (X pattern between struts) */}
          <Line
            points={[
              [-2, -3.0, 0.24],
              [2, 4.5, 0.24],
            ]}
            color="#8a7a60"
            lineWidth={0.5}
            transparent
            opacity={0.5}
          />
          <Line
            points={[
              [2, -3.0, 0.24],
              [-2, 4.5, 0.24],
            ]}
            color="#8a7a60"
            lineWidth={0.5}
            transparent
            opacity={0.5}
          />

          {/* Interplane struts */}
          <Line
            points={[
              [2.5, -3.0, 0.25],
              [2.5, 4.5, 0.25],
            ]}
            color="#5c4a32"
            lineWidth={2}
          />
          <Line
            points={[
              [-2.5, -3.0, 0.25],
              [-2.5, 4.5, 0.25],
            ]}
            color="#5c4a32"
            lineWidth={2}
          />

          {/* Upper wing */}
          <mesh position={[0, 3.5, 0.28]}>
            <shapeGeometry args={[upperWing]} />
            <meshBasicMaterial color={color} side={THREE.DoubleSide} />
          </mesh>

          {/* Tail fin (vertical rudder) */}
          <mesh position={[0, 0, 0.35]}>
            <shapeGeometry args={[tailFin]} />
            <meshBasicMaterial color={darkColor} side={THREE.DoubleSide} />
          </mesh>

          {/* Windscreen */}
          <mesh position={[0, 0, 0.40]}>
            <shapeGeometry args={[windscreen]} />
            <meshBasicMaterial
              color="#2a1a0a"
              side={THREE.DoubleSide}
              transparent
              opacity={0.6}
            />
          </mesh>

          {/* Pilot head */}
          <mesh position={[1.5, 3.8, 0.42]}>
            <circleGeometry args={[1.0, 8]} />
            <meshBasicMaterial color="#2a1a0a" side={THREE.DoubleSide} />
          </mesh>

          {/* Propeller */}
          <group ref={propRef} position={[9.5, 0, 0.45]}>
            {/* Blade 1 */}
            <mesh>
              <planeGeometry args={[0.8, 5.5]} />
              <meshBasicMaterial
                color="#4a3a20"
                side={THREE.DoubleSide}
                transparent
                opacity={0.85}
              />
            </mesh>
            {/* Blade 2 */}
            <mesh rotation={[0, 0, Math.PI / 2]}>
              <planeGeometry args={[0.8, 5.5]} />
              <meshBasicMaterial
                color="#4a3a20"
                side={THREE.DoubleSide}
                transparent
                opacity={0.85}
              />
            </mesh>
          </group>
          {/* Spinner hub */}
          <mesh position={[9.5, 0, 0.46]}>
            <circleGeometry args={[0.7, 8]} />
            <meshBasicMaterial color="#4a3a20" side={THREE.DoubleSide} />
          </mesh>
        </group>
      </group>

      {/* UI group — never flipped */}
      <group>
        {/* HP pips */}
        <group position={[0, 28, 0.5]}>
          {hpSegments.map((i) => (
            <mesh key={i} position={[(i - 1) * 5, 0, 0]}>
              <circleGeometry args={[1.8, 6]} />
              <meshBasicMaterial
                color={i < state.hp ? "#8b0000" : "#2a2015"}
                transparent
                opacity={i < state.hp ? 0.9 : 0.35}
              />
            </mesh>
          ))}
        </group>

        {/* Player label */}
        <Text
          position={[0, 38, 0.5]}
          fontSize={14}
          color="#2a2015"
          anchorX="center"
          anchorY="middle"
          renderOrder={999}
        >
          {label}
        </Text>
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
