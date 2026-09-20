import { describe, it, expect } from 'vitest';
import {
  computeBalanceLQR,
  computeBrakeLQR,
  solveCARE,
  linearizeContinuous,
  DEFAULT_Q_BALANCE,
  DEFAULT_R_BALANCE,
  STATE_DIM,
  STATE_UPRIGHT,
} from '../src/index.js';

/** Frobenius norm of A^T P + P A - P B R^-1 B^T P + Q. */
function careResidual(A: number[][], B: number[][], Q: number[][], R: number, P: number[][]) {
  const n = STATE_DIM;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let r = Q[i][j];
      for (let k = 0; k < n; k++) r += A[k][i] * P[k][j] + P[i][k] * A[k][j];
      let pbi = 0;
      let pbj = 0;
      for (let k = 0; k < n; k++) {
        pbi += P[i][k] * B[k][0];
        pbj += P[j][k] * B[k][0];
      }
      r -= (pbi * pbj) / R;
      sum += r * r;
    }
  }
  return Math.sqrt(sum);
}

const diag = (d: number[]) => d.map((v, i) => d.map((_, j) => (i === j ? v : 0)));

describe('CARE solver (matrix sign function)', () => {
  it('reproduces the balance gains shipped on davidnash.dev', () => {
    // Gains published on davidnash.dev (Q = diag(8, 6, 150, 12, 150, 12), R = 0.1).
    const { K } = computeBalanceLQR();
    const shipped = [8.9443, 20.0675, -295.9858, 12.1302, 456.6956, 87.1185];
    K.forEach((k, i) => expect(k).toBeCloseTo(shipped[i], 3));
  });

  it('reproduces the brake gains from the site', () => {
    const { K } = computeBrakeLQR();
    const shipped = [22.3607, 12.4484, 8.0041, -2.0341, -5.0894, -1.0726];
    K.forEach((k, i) => expect(k).toBeCloseTo(shipped[i], 3));
  });

  it('satisfies the Riccati equation to near machine precision', () => {
    for (const compute of [computeBalanceLQR, computeBrakeLQR]) {
      const { A, B, P } = compute();
      const Q = diag(
        compute === computeBalanceLQR ? DEFAULT_Q_BALANCE : [50, 1, 0.5, 0.5, 0.5, 0.5]
      );
      expect(careResidual(A, B, Q, 0.1, P)).toBeLessThan(1e-7);
    }
  });

  it('returns a symmetric, positive definite P', () => {
    const { P } = computeBalanceLQR();
    for (let i = 0; i < STATE_DIM; i++) {
      for (let j = 0; j < STATE_DIM; j++) expect(P[i][j]).toBeCloseTo(P[j][i], 9);
    }
    // Positive definite: x^T P x > 0 for a spread of directions.
    for (let trial = 0; trial < 20; trial++) {
      const x = Array.from({ length: STATE_DIM }, (_, i) => Math.sin(1.7 * trial + 2.3 * i + 0.4));
      let q = 0;
      for (let i = 0; i < STATE_DIM; i++)
        for (let j = 0; j < STATE_DIM; j++) q += x[i] * P[i][j] * x[j];
      expect(q).toBeGreaterThan(0);
    }
  });

  it('is consistent with K = R^-1 B^T P', () => {
    const { A, B } = linearizeContinuous(STATE_UPRIGHT);
    const Q = diag(DEFAULT_Q_BALANCE);
    const { P, K } = solveCARE(A, B, Q, DEFAULT_R_BALANCE);
    for (let j = 0; j < STATE_DIM; j++) {
      let bp = 0;
      for (let k = 0; k < STATE_DIM; k++) bp += B[k][0] * P[k][j];
      expect(K[j]).toBeCloseTo(bp / DEFAULT_R_BALANCE, 9);
    }
  });

  it('throws instead of returning garbage when no stabilizing solution exists', () => {
    // Undriven pure rotation: the Hamiltonian's eigenvalues sit on the imaginary axis.
    const A = Array.from({ length: STATE_DIM }, () => new Array(STATE_DIM).fill(0));
    for (let b = 0; b < STATE_DIM; b += 2) {
      A[b][b + 1] = 1;
      A[b + 1][b] = -1;
    }
    const B = Array.from({ length: STATE_DIM }, () => [0]);
    const Q = diag(new Array(STATE_DIM).fill(0));
    expect(() => solveCARE(A, B, Q, 1)).toThrow();
  });
});
