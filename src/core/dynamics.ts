import { State, PlantParams, DEFAULT_PLANT_PARAMS } from './types.js';
import { inv3 } from './math.js';

export interface PrecomputedDynamics {
  d1: number;
  d2: number;
  d3: number;
  d4: number;
  d5: number;
  d6: number;
  f1: number;
  f2: number;
  I1: number;
  I2: number;
}

/**
 * Precomputes constant mass matrix terms from plant physical parameters:
 * - d1: Total mass (M + m1 + m2)
 * - d2: First moment of link 1 about cart
 * - d3: First moment of link 2 about link 1
 * - d4: Moment of inertia of link 1
 * - d5: Cross-coupling inertia between link 1 and link 2
 * - d6: Moment of inertia of link 2
 * - f1, f2: Gravitational torque coefficients
 */
export function precomputeDynamics(p: PlantParams = DEFAULT_PLANT_PARAMS): PrecomputedDynamics {
  const { g, M, m1, m2, L1, L2 } = p;
  return {
    d1: M + m1 + m2,
    d2: (0.5 * m1 + m2) * L1,
    d3: 0.5 * m2 * L2,
    d4: ((1 / 3) * m1 + m2) * L1 * L1,
    d5: 0.5 * m2 * L1 * L2,
    d6: (1 / 3) * m2 * L2 * L2,
    f1: (0.5 * m1 + m2) * g * L1,
    f2: 0.5 * m2 * g * L2,
    I1: (m1 * L1 * L1) / 12,
    I2: (m2 * L2 * L2) / 12,
  };
}

const defaultPrecomputed = precomputeDynamics(DEFAULT_PLANT_PARAMS);

/**
 * Computes continuous-time state derivatives ds/dt = [v, a0, w1, a1, w2, a2].
 *
 * System equations of motion:
 *   M(q) * q̈ + C(q, q̇) * q̇ + G(q) + D * q̇ = [u, 0, 0]^T
 *
 * Where:
 * - q = [x, θ1, θ2]^T (generalized coordinates)
 * - s = [x, v, θ1, ω1, θ2, ω2]^T (6D state vector)
 * - u = horizontal actuator force on cart (N)
 * - Mm = 3x3 symmetric mass matrix M(q)
 * - h1, h2, h3 = non-inertial forces (Coriolis + gravity + damping)
 * - Mi = inv(Mm)
 * - a0, a1, a2 = accelerations [ẍ, θ̈1, θ̈2]^T
 *
 * @param s Current state [x, v, th1, w1, th2, w2]
 * @param u Horizontal force on cart (N)
 * @param p Optional plant parameters
 * @param pre Optional precomputed terms
 */
export function dynamics(
  s: State,
  u: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = defaultPrecomputed
): State {
  const [, v, th1, w1, th2, w2] = s;
  const c1 = Math.cos(th1);
  const c2 = Math.cos(th2);
  const c12 = Math.cos(th1 - th2);
  const s1 = Math.sin(th1);
  const s2 = Math.sin(th2);
  const s12 = Math.sin(th1 - th2);

  // Generalized Mass Matrix M(q) (3x3)
  const Mm = [
    [pre.d1, pre.d2 * c1, pre.d3 * c2],
    [pre.d2 * c1, pre.d4, pre.d5 * c12],
    [pre.d3 * c2, pre.d5 * c12, pre.d6],
  ];

  // Non-inertial forces: Coriolis + gravity + viscous damping terms
  const h1 = -pre.d2 * w1 * w1 * s1 - pre.d3 * w2 * w2 * s2 + p.cartDamp * v;
  const h2 = pre.d5 * w2 * w2 * s12 - pre.f1 * s1 + p.jointDamp * w1;
  const h3 = -pre.d5 * w1 * w1 * s12 - pre.f2 * s2 + p.jointDamp * w2;

  // Invert 3x3 mass matrix and solve for accelerations: q̈ = M^-1 * (τ - h)
  const Mi = inv3(Mm);
  const r0 = u - h1;
  const r1 = -h2;
  const r2 = -h3;

  const a0 = Mi[0][0] * r0 + Mi[0][1] * r1 + Mi[0][2] * r2;
  const a1 = Mi[1][0] * r0 + Mi[1][1] * r1 + Mi[1][2] * r2;
  const a2 = Mi[2][0] * r0 + Mi[2][1] * r1 + Mi[2][2] * r2;

  return [v, a0, w1, a1, w2, a2];
}

/**
 * 4th-order Runge-Kutta numerical integration step:
 *   k1 = f(s, u)
 *   k2 = f(s + 0.5*dt*k1, u)
 *   k3 = f(s + 0.5*dt*k2, u)
 *   k4 = f(s + dt*k3, u)
 *   s_{t+1} = s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
 *
 * @param s Current state
 * @param u Control force (constant over dt)
 * @param dt Timestep duration in seconds
 * @param p Optional plant parameters
 */
export function rk4(
  s: State,
  u: number,
  dt: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = defaultPrecomputed
): State {
  const k1 = dynamics(s, u, p, pre);

  const s2: State = [
    s[0] + 0.5 * dt * k1[0],
    s[1] + 0.5 * dt * k1[1],
    s[2] + 0.5 * dt * k1[2],
    s[3] + 0.5 * dt * k1[3],
    s[4] + 0.5 * dt * k1[4],
    s[5] + 0.5 * dt * k1[5],
  ];
  const k2 = dynamics(s2, u, p, pre);

  const s3: State = [
    s[0] + 0.5 * dt * k2[0],
    s[1] + 0.5 * dt * k2[1],
    s[2] + 0.5 * dt * k2[2],
    s[3] + 0.5 * dt * k2[3],
    s[4] + 0.5 * dt * k2[4],
    s[5] + 0.5 * dt * k2[5],
  ];
  const k3 = dynamics(s3, u, p, pre);

  const s4: State = [
    s[0] + dt * k3[0],
    s[1] + dt * k3[1],
    s[2] + dt * k3[2],
    s[3] + dt * k3[3],
    s[4] + dt * k3[4],
    s[5] + dt * k3[5],
  ];
  const k4 = dynamics(s4, u, p, pre);

  return [
    s[0] + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
    s[1] + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    s[2] + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
    s[3] + (dt / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
    s[4] + (dt / 6) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]),
    s[5] + (dt / 6) * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]),
  ];
}

/**
 * Computes the total mechanical energy (Kinetic + Potential) of the system in Joules.
 * Upright equilibrium has maximum potential energy (+19.62 J with default parameters).
 * Hanging equilibrium has minimum potential energy (-19.62 J with default parameters).
 */
export function totalEnergy(s: State, p: PlantParams = DEFAULT_PLANT_PARAMS): number {
  const [, v, th1, w1, th2, w2] = s;
  const { m1, m2, L1, L2, g, M } = p;

  const vx1 = v + 0.5 * L1 * w1 * Math.cos(th1);
  const vy1 = 0.5 * L1 * w1 * Math.sin(th1);
  const vx2 = v + L1 * w1 * Math.cos(th1) + 0.5 * L2 * w2 * Math.cos(th2);
  const vy2 = L1 * w1 * Math.sin(th1) + 0.5 * L2 * w2 * Math.sin(th2);

  const I1 = (m1 * L1 * L1) / 12;
  const I2 = (m2 * L2 * L2) / 12;

  // Kinetic energy: Cart + Link 1 (linear + rot) + Link 2 (linear + rot)
  const T =
    0.5 * M * v * v +
    0.5 * m1 * (vx1 * vx1 + vy1 * vy1) +
    0.5 * I1 * w1 * w1 +
    0.5 * m2 * (vx2 * vx2 + vy2 * vy2) +
    0.5 * I2 * w2 * w2;

  // Potential energy (datum at hinge on cart)
  const V =
    m1 * g * (0.5 * L1 * Math.cos(th1)) + m2 * g * (L1 * Math.cos(th1) + 0.5 * L2 * Math.cos(th2));

  return T + V;
}
