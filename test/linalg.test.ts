import { describe, it, expect } from 'vitest';
import { invert, solveLeastSquares, frobenius } from '../src/core/linalg.js';

/** Deterministic pseudo-random matrix so failures are reproducible. */
function matrix(rows: number, cols: number, seed: number) {
  const a = new Float64Array(rows * cols);
  let s = seed;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) % 4294967296;
    a[i] = s / 4294967296 - 0.5;
  }
  return a;
}

describe('linalg helpers', () => {
  it('inverts a dense matrix: A * A^-1 = I', () => {
    const n = 12;
    const A = matrix(n, n, 7);
    for (let i = 0; i < n; i++) A[i * n + i] += 3; // keep it well conditioned
    const Ai = invert(A, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += A[i * n + k] * Ai[k * n + j];
        expect(s).toBeCloseTo(i === j ? 1 : 0, 10);
      }
    }
  });

  it('needs row pivoting for a zero on the diagonal', () => {
    const A = Float64Array.from([0, 1, 1, 0]);
    const Ai = invert(A, 2);
    expect(Array.from(Ai)).toEqual([0, 1, 1, 0]);
  });

  it('throws on a singular matrix', () => {
    expect(() => invert(Float64Array.from([1, 2, 2, 4]), 2)).toThrow(/singular/i);
  });

  it('recovers the exact solution of a consistent overdetermined system', () => {
    const m = 12;
    const n = 6;
    const k = 3;
    const A = matrix(m, n, 11);
    const X = matrix(n, k, 23);
    const B = new Float64Array(m * k);
    for (let i = 0; i < m; i++)
      for (let c = 0; c < k; c++)
        for (let j = 0; j < n; j++) B[i * k + c] += A[i * n + j] * X[j * k + c];

    const sol = solveLeastSquares(A, m, n, B, k);
    for (let i = 0; i < n * k; i++) expect(sol[i]).toBeCloseTo(X[i], 10);
  });

  it('minimizes the residual of an inconsistent system (normal equations hold)', () => {
    const m = 10;
    const n = 4;
    const A = matrix(m, n, 5);
    const B = matrix(m, 1, 9);
    const x = solveLeastSquares(A, m, n, B, 1);

    // Residual must be orthogonal to the columns of A: A^T (A x - b) = 0.
    const r = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      r[i] = -B[i];
      for (let j = 0; j < n; j++) r[i] += A[i * n + j] * x[j];
    }
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < m; i++) s += A[i * n + j] * r[i];
      expect(Math.abs(s)).toBeLessThan(1e-12);
    }
    expect(frobenius(r)).toBeGreaterThan(1e-3); // genuinely inconsistent
  });

  it('throws on a rank-deficient matrix', () => {
    const A = Float64Array.from([1, 1, 2, 2, 3, 3]); // two identical columns
    expect(() => solveLeastSquares(A, 3, 2, Float64Array.from([1, 2, 3]), 1)).toThrow(/rank/i);
  });
});
