import { State, PlantParams, DEFAULT_PLANT_PARAMS } from './types.js';

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
const precomputedCache = new WeakMap<PlantParams, PrecomputedDynamics>();

/**
 * Precomputed terms for `p`, cached by object identity. Do not mutate a PlantParams after use;
 * create a new object instead.
 */
export function getPrecomputed(p: PlantParams): PrecomputedDynamics {
  if (p === DEFAULT_PLANT_PARAMS) return defaultPrecomputed;
  let pre = precomputedCache.get(p);
  if (!pre) {
    pre = precomputeDynamics(p);
    precomputedCache.set(p, pre);
  }
  return pre;
}

type Vec = ArrayLike<number>;
type Out = number[] | Float64Array;

/** Inverse of a symmetric 3x3 as six scalars [i00, i01, i02, i11, i12, i22], in a scratch buffer. */
const SYM = new Float64Array(6);
function symInv3(m00: number, m01: number, m02: number, m11: number, m12: number, m22: number) {
  const det =
    m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02) + m02 * (m01 * m12 - m11 * m02);
  const i = 1 / det;
  SYM[0] = (m11 * m22 - m12 * m12) * i;
  SYM[1] = (m02 * m12 - m01 * m22) * i;
  SYM[2] = (m01 * m12 - m02 * m11) * i;
  SYM[3] = (m00 * m22 - m02 * m02) * i;
  SYM[4] = (m01 * m02 - m00 * m12) * i;
  SYM[5] = (m00 * m11 - m01 * m01) * i;
}

/**
 * Accelerations [ẍ, θ̈1, θ̈2] from M(q) q̈ = [u, 0, 0] - C(q, q̇) q̇ - G(q) - D q̇, written into `out`.
 * `pre` is looked up from `p` when omitted.
 */
export function accelerationsInto(
  out: Out,
  s: Vec,
  u: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = getPrecomputed(p)
): void {
  const v = s[1];
  const th1 = s[2];
  const w1 = s[3];
  const th2 = s[4];
  const w2 = s[5];
  const c1 = Math.cos(th1);
  const c2 = Math.cos(th2);
  const c12 = Math.cos(th1 - th2);
  const s1 = Math.sin(th1);
  const s2 = Math.sin(th2);
  const s12 = Math.sin(th1 - th2);

  // Non-inertial forces: Coriolis + gravity + viscous damping
  const h1 = -pre.d2 * w1 * w1 * s1 - pre.d3 * w2 * w2 * s2 + p.cartDamp * v;
  const h2 = pre.d5 * w2 * w2 * s12 - pre.f1 * s1 + p.jointDamp * w1;
  const h3 = -pre.d5 * w1 * w1 * s12 - pre.f2 * s2 + p.jointDamp * w2;

  symInv3(pre.d1, pre.d2 * c1, pre.d3 * c2, pre.d4, pre.d5 * c12, pre.d6);
  const r0 = u - h1;
  const r1 = -h2;
  const r2 = -h3;

  out[0] = SYM[0] * r0 + SYM[1] * r1 + SYM[2] * r2;
  out[1] = SYM[1] * r0 + SYM[3] * r1 + SYM[4] * r2;
  out[2] = SYM[2] * r0 + SYM[4] * r1 + SYM[5] * r2;
}

const ACC = new Float64Array(3);

/**
 * Continuous-time state derivatives ds/dt = [v, a0, w1, a1, w2, a2], written into `out`.
 * Allocation-free counterpart of {@link dynamics}.
 */
export function dynamicsInto(
  out: Out,
  s: Vec,
  u: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = getPrecomputed(p)
): void {
  accelerationsInto(ACC, s, u, p, pre);
  out[0] = s[1];
  out[1] = ACC[0];
  out[2] = s[3];
  out[3] = ACC[1];
  out[4] = s[5];
  out[5] = ACC[2];
}

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
 * - M = 3x3 symmetric mass matrix M(q)
 * - a0, a1, a2 = accelerations [ẍ, θ̈1, θ̈2]^T
 *
 * @param s Current state [x, v, th1, w1, th2, w2]
 * @param u Horizontal force on cart (N)
 * @param p Optional plant parameters
 * @param pre Optional precomputed terms (derived from `p` if omitted)
 */
export function dynamics(
  s: State,
  u: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = getPrecomputed(p)
): State {
  const out: State = [0, 0, 0, 0, 0, 0];
  dynamicsInto(out, s, u, p, pre);
  return out;
}

const K1 = new Float64Array(6);
const K2 = new Float64Array(6);
const K3 = new Float64Array(6);
const K4 = new Float64Array(6);
const STAGE = new Float64Array(6);

/**
 * 4th-order Runge-Kutta step, written into `out` (which may be the same array as `s`):
 *   k1 = f(s, u)
 *   k2 = f(s + 0.5*dt*k1, u)
 *   k3 = f(s + 0.5*dt*k2, u)
 *   k4 = f(s + dt*k3, u)
 *   s_{t+1} = s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
 */
export function rk4Into(
  out: Out,
  s: Vec,
  u: number,
  dt: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = getPrecomputed(p)
): void {
  dynamicsInto(K1, s, u, p, pre);
  for (let i = 0; i < 6; i++) STAGE[i] = s[i] + 0.5 * dt * K1[i];
  dynamicsInto(K2, STAGE, u, p, pre);
  for (let i = 0; i < 6; i++) STAGE[i] = s[i] + 0.5 * dt * K2[i];
  dynamicsInto(K3, STAGE, u, p, pre);
  for (let i = 0; i < 6; i++) STAGE[i] = s[i] + dt * K3[i];
  dynamicsInto(K4, STAGE, u, p, pre);
  const h6 = dt / 6;
  for (let i = 0; i < 6; i++) out[i] = s[i] + h6 * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]);
}

/**
 * 4th-order Runge-Kutta integration step (see {@link rk4Into} for the scheme).
 *
 * @param s Current state
 * @param u Control force (constant over dt)
 * @param dt Timestep duration in seconds
 * @param p Optional plant parameters
 * @param pre Optional precomputed terms (derived from `p` if omitted)
 */
export function rk4(
  s: State,
  u: number,
  dt: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = getPrecomputed(p)
): State {
  const out: State = [0, 0, 0, 0, 0, 0];
  rk4Into(out, s, u, dt, p, pre);
  return out;
}

/**
 * Analytic derivatives of the accelerations with respect to every quantity that affects them
 * (v, θ1, ω1, θ2, ω2 and the force u; the cart position x drops out of the dynamics entirely),
 * written into a caller-owned Float64Array(18) as six consecutive 3-vectors:
 *
 *   [ da/dv | da/dθ1 | da/dω1 | da/dθ2 | da/dω2 | da/du ]
 *
 * Each 3-vector holds the derivative of [ẍ, θ̈1, θ̈2]. If `aOut` is given, the accelerations
 * themselves are written there too, since they fall out of the same computation.
 *
 * Differentiating M q̈ = r gives dq̈ = M⁻¹(dr − dM q̈). Reusing the q̈ already computed avoids forming
 * d(M⁻¹) as full 3x3 matrices.
 */
export function accelJacobianInto(
  J: Float64Array,
  s: Vec,
  u: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = getPrecomputed(p),
  aOut?: Float64Array
): void {
  const { d2, d3, d5, f1, f2 } = pre;
  const { cartDamp, jointDamp } = p;
  const v = s[1];
  const th1 = s[2];
  const w1 = s[3];
  const th2 = s[4];
  const w2 = s[5];
  const c1 = Math.cos(th1);
  const s1 = Math.sin(th1);
  const c2 = Math.cos(th2);
  const s2 = Math.sin(th2);
  const c12 = Math.cos(th1 - th2);
  const s12 = Math.sin(th1 - th2);

  symInv3(pre.d1, d2 * c1, d3 * c2, pre.d4, d5 * c12, pre.d6);

  const h1 = -d2 * w1 * w1 * s1 - d3 * w2 * w2 * s2 + cartDamp * v;
  const h2 = d5 * w2 * w2 * s12 - f1 * s1 + jointDamp * w1;
  const h3 = -d5 * w1 * w1 * s12 - f2 * s2 + jointDamp * w2;
  const r0 = u - h1;
  const r1 = -h2;
  const r2 = -h3;

  const a0 = SYM[0] * r0 + SYM[1] * r1 + SYM[2] * r2;
  const a1 = SYM[1] * r0 + SYM[3] * r1 + SYM[4] * r2;
  const a2 = SYM[2] * r0 + SYM[4] * r1 + SYM[5] * r2;
  if (aOut) {
    aOut[0] = a0;
    aOut[1] = a1;
    aOut[2] = a2;
  }

  // M has no velocity dependence, so only dr/dz matters for v, ω1 and ω2.
  putMiTimes(J, 15, 1, 0, 0); // da/du
  J[0] = -cartDamp * J[15]; // da/dv
  J[1] = -cartDamp * J[16];
  J[2] = -cartDamp * J[17];
  putMiTimes(J, 6, 2 * d2 * w1 * s1, -jointDamp, 2 * d5 * w1 * s12); // da/dω1
  putMiTimes(J, 12, 2 * d3 * w2 * s2, -2 * d5 * w2 * s12, -jointDamp); // da/dω2

  // θ1 and θ2 move M as well as r: add -(dM/dθ) q̈, using dM/dθ's sparsity.
  putMiTimes(
    J,
    3, // da/dθ1
    d2 * w1 * w1 * c1 + d2 * s1 * a1,
    -d5 * w2 * w2 * c12 + f1 * c1 + d2 * s1 * a0 + d5 * s12 * a2,
    d5 * w1 * w1 * c12 + d5 * s12 * a1
  );
  putMiTimes(
    J,
    9, // da/dθ2
    d3 * w2 * w2 * c2 + d3 * s2 * a2,
    d5 * w2 * w2 * c12 - d5 * s12 * a2,
    -d5 * w1 * w1 * c12 + f2 * c2 + d3 * s2 * a0 - d5 * s12 * a1
  );
}

/** J[off..off+2] = M⁻¹ [x0, x1, x2], reading M⁻¹ from the SYM scratch buffer. */
function putMiTimes(J: Float64Array, off: number, x0: number, x1: number, x2: number): void {
  J[off] = SYM[0] * x0 + SYM[1] * x1 + SYM[2] * x2;
  J[off + 1] = SYM[1] * x0 + SYM[3] * x1 + SYM[4] * x2;
  J[off + 2] = SYM[2] * x0 + SYM[4] * x1 + SYM[5] * x2;
}

const AJ = new Float64Array(18);
const AA = new Float64Array(3);

/**
 * Analytic continuous-time Jacobians of ds/dt = f(s, u):
 * A = ∂f/∂s (6x6, row-major) and B = ∂f/∂u (length 6), written into caller-owned arrays.
 * If `fOut` is given, f(s, u) itself is written there too.
 */
export function dynamicsJacobianInto(
  A: Float64Array,
  B: Float64Array,
  s: Vec,
  u: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = getPrecomputed(p),
  fOut?: Float64Array
): void {
  accelJacobianInto(AJ, s, u, p, pre, AA);
  A.fill(0);
  B.fill(0);

  // Position rows are kinematic: dx/dt = v, dθ1/dt = ω1, dθ2/dt = ω2.
  A[0 * 6 + 1] = 1;
  A[2 * 6 + 3] = 1;
  A[4 * 6 + 5] = 1;

  // Acceleration rows (1, 3, 5), columns v, θ1, ω1, θ2, ω2 = 1..5.
  for (let r = 0; r < 3; r++) {
    const row = (2 * r + 1) * 6;
    A[row + 1] = AJ[r]; // da/dv
    A[row + 2] = AJ[3 + r]; // da/dθ1
    A[row + 3] = AJ[6 + r]; // da/dω1
    A[row + 4] = AJ[9 + r]; // da/dθ2
    A[row + 5] = AJ[12 + r]; // da/dω2
    B[2 * r + 1] = AJ[15 + r]; // da/du
  }

  if (fOut) {
    fOut[0] = s[1];
    fOut[1] = AA[0];
    fOut[2] = s[3];
    fOut[3] = AA[1];
    fOut[4] = s[5];
    fOut[5] = AA[2];
  }
}

const AC = new Float64Array(36);
const BC = new Float64Array(6);
const KF = new Float64Array(6);
const RSTAGE = new Float64Array(6);
const DK: Float64Array[] = [0, 1, 2, 3].map(() => new Float64Array(36));
const DKU: Float64Array[] = [0, 1, 2, 3].map(() => new Float64Array(6));
const MTMP = new Float64Array(36);
const VTMP = new Float64Array(6);
/** Stage i+1 of RK4 is evaluated at s + STAGE_FRAC[i] * dt * K_i. */
const STAGE_FRAC = [0.5, 0.5, 1.0];

/** out = a * b for 6x6 row-major matrices (out must not alias a or b). */
function matMul6(out: Float64Array, a: Float64Array, b: Float64Array): void {
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) {
      let acc = 0;
      for (let k = 0; k < 6; k++) acc += a[i * 6 + k] * b[k * 6 + j];
      out[i * 6 + j] = acc;
    }
  }
}

/**
 * Exact Jacobians of one RK4 step s' = rk4(s, u, dt):
 *   fx = ∂s'/∂s (6x6, row-major),  fu = ∂s'/∂u (length 6),
 * written into caller-owned arrays.
 *
 * The RK4 step is a composition of four evaluations of f, so its derivative follows from the chain
 * rule through the stages (K_i are the stage slopes, A_i and B_i the continuous Jacobians at each stage):
 *
 *   dK1 = A1
 *   dK2 = A2 (I + dt/2 dK1)      dK2u = A2 (dt/2 dK1u) + B2
 *   dK3 = A3 (I + dt/2 dK2)      dK3u = A3 (dt/2 dK2u) + B3
 *   dK4 = A4 (I + dt   dK3)      dK4u = A4 (dt   dK3u) + B4
 *   fx  = I + dt/6 (dK1 + 2 dK2 + 2 dK3 + dK4)
 *   fu  =     dt/6 (dK1u + 2 dK2u + 2 dK3u + dK4u)
 *
 * This differentiates the integrator, not the underlying ODE, so it matches a finite difference of
 * `rk4` to roundoff.
 */
export function rk4JacobianInto(
  fx: Float64Array,
  fu: Float64Array,
  s: Vec,
  u: number,
  dt: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = getPrecomputed(p)
): void {
  dynamicsJacobianInto(AC, BC, s, u, p, pre, KF);
  DK[0].set(AC);
  DKU[0].set(BC);

  for (let st = 1; st < 4; st++) {
    const frac = STAGE_FRAC[st - 1] * dt;
    const prevK = DK[st - 1];
    const prevKu = DKU[st - 1];

    // Stage state s + frac * K_{st}, where K_{st} = f at the previous stage (still in KF).
    for (let i = 0; i < 6; i++) RSTAGE[i] = s[i] + frac * KF[i];
    dynamicsJacobianInto(AC, BC, RSTAGE, u, p, pre, KF);

    for (let i = 0; i < 36; i++) MTMP[i] = frac * prevK[i];
    for (let i = 0; i < 6; i++) MTMP[i * 6 + i] += 1;
    matMul6(DK[st], AC, MTMP);

    for (let i = 0; i < 6; i++) VTMP[i] = frac * prevKu[i];
    for (let i = 0; i < 6; i++) {
      let acc = BC[i];
      for (let k = 0; k < 6; k++) acc += AC[i * 6 + k] * VTMP[k];
      DKU[st][i] = acc;
    }
  }

  const h6 = dt / 6;
  for (let i = 0; i < 36; i++) {
    fx[i] = h6 * (DK[0][i] + 2 * DK[1][i] + 2 * DK[2][i] + DK[3][i]);
  }
  for (let i = 0; i < 6; i++) fx[i * 6 + i] += 1;
  for (let i = 0; i < 6; i++) {
    fu[i] = h6 * (DKU[0][i] + 2 * DKU[1][i] + 2 * DKU[2][i] + DKU[3][i]);
  }
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
