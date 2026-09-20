import { describe, it, expect } from 'vitest';
import {
  accelJacobianInto,
  accelerationsInto,
  dynamicsJacobianInto,
  linearizeDiscrete,
  linearizeDiscreteFD,
  linearizeContinuous,
  linearizeContinuousFD,
  DEFAULT_PLANT_PARAMS,
  PlantParams,
  State,
} from '../src/index.js';

const CUSTOM: PlantParams = {
  ...DEFAULT_PLANT_PARAMS,
  M: 2.0,
  m1: 1.5,
  m2: 0.7,
  L1: 0.8,
  L2: 1.1,
};

/** Deterministic random states covering full angle range, fast spins, and large forces. */
function randomCases(count: number) {
  let seed = 1234;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
  return Array.from({ length: count }, () => ({
    s: [
      (rand() - 0.5) * 4,
      (rand() - 0.5) * 6,
      rand() * 2 * Math.PI,
      (rand() - 0.5) * 8,
      rand() * 2 * Math.PI,
      (rand() - 0.5) * 8,
    ] as State,
    u: (rand() - 0.5) * 400,
  }));
}

describe('Analytic Jacobians agree with finite differences', () => {
  for (const [label, plant] of [
    ['default plant', DEFAULT_PLANT_PARAMS],
    ['custom plant', CUSTOM],
  ] as const) {
    it(`RK4 step Jacobian (${label}), 300 random states`, () => {
      let worst = 0;
      for (const { s, u } of randomCases(300)) {
        const a = linearizeDiscrete(s, u, 0.02, plant);
        const f = linearizeDiscreteFD(s, u, 0.02, 1e-6, plant);
        for (let i = 0; i < 6; i++) {
          for (let j = 0; j < 6; j++) worst = Math.max(worst, Math.abs(a.fx[i][j] - f.fx[i][j]));
          worst = Math.max(worst, Math.abs(a.fu[i][0] - f.fu[i][0]));
        }
      }
      // The finite-difference reference is good to ~1e-9; a wrong derivative would be O(1).
      expect(worst).toBeLessThan(1e-6);
    });

    it(`continuous-time A and B (${label}), 300 random states`, () => {
      let worst = 0;
      for (const { s, u } of randomCases(300)) {
        const a = linearizeContinuous(s, u, plant);
        const f = linearizeContinuousFD(s, u, 1e-6, plant);
        for (let i = 0; i < 6; i++) {
          for (let j = 0; j < 6; j++) worst = Math.max(worst, Math.abs(a.A[i][j] - f.A[i][j]));
          worst = Math.max(worst, Math.abs(a.B[i][0] - f.B[i][0]));
        }
      }
      expect(worst).toBeLessThan(1e-5);
    });
  }

  it('acceleration Jacobian columns match finite differences one at a time', () => {
    const J = new Float64Array(18);
    const eps = 1e-6;
    const a0 = new Float64Array(3);
    const ap = new Float64Array(3);
    const am = new Float64Array(3);
    const inputs = [1, 2, 3, 4, 5]; // v, th1, w1, th2, w2 (indices into the state)

    for (const { s, u } of randomCases(50)) {
      accelJacobianInto(J, s, u, DEFAULT_PLANT_PARAMS, undefined, a0);
      accelerationsInto(ap, s, u);
      for (let r = 0; r < 3; r++) expect(a0[r]).toBeCloseTo(ap[r], 12);

      inputs.forEach((idx, col) => {
        const sp = [...s] as State;
        const sm = [...s] as State;
        sp[idx] += eps;
        sm[idx] -= eps;
        accelerationsInto(ap, sp, u);
        accelerationsInto(am, sm, u);
        for (let r = 0; r < 3; r++) {
          const fd = (ap[r] - am[r]) / (2 * eps);
          expect(Math.abs(J[3 * col + r] - fd)).toBeLessThan(1e-5 * Math.max(1, Math.abs(fd)));
        }
      });

      accelerationsInto(ap, s, u + eps);
      accelerationsInto(am, s, u - eps);
      for (let r = 0; r < 3; r++) {
        expect(J[15 + r]).toBeCloseTo((ap[r] - am[r]) / (2 * eps), 6);
      }
    }
  });

  it('the position rows of the continuous Jacobian are purely kinematic', () => {
    const A = new Float64Array(36);
    const B = new Float64Array(6);
    dynamicsJacobianInto(A, B, [0.3, -1, 2.4, 1.7, -2, 0.6], 12);
    // dx/dt = v, dθ1/dt = ω1, dθ2/dt = ω2, and the cart position x never appears in the dynamics.
    expect(A[0 * 6 + 1]).toBe(1);
    expect(A[2 * 6 + 3]).toBe(1);
    expect(A[4 * 6 + 5]).toBe(1);
    for (let r = 0; r < 6; r++) expect(A[r * 6 + 0]).toBe(0);
    expect(B[0]).toBe(0);
    expect(B[2]).toBe(0);
    expect(B[4]).toBe(0);
  });
});
