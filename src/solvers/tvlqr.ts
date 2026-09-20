import { State, STATE_DIM, PlantParams, DEFAULT_PLANT_PARAMS } from '../core/types.js';
import { getPrecomputed } from '../core/dynamics.js';
import { linearizeDiscreteInto } from './ilqr.js';

export interface TVLQROptions {
  /** Substeps per knot interval to prevent discretization distortion, default 5 */
  substeps?: number;
  /** Plant parameters */
  plant?: PlantParams;
}

const NX = STATE_DIM;

/**
 * Computes time-varying LQR feedback gain schedule K(t) along a nominal state-action trajectory (xs, us).
 *
 * Mathematical Formulation:
 * - Solves the Discrete Algebraic Riccati Equation (DARE) backward recursion along the nominal trajectory.
 * - S = R_h + B^T * P * B (scalar control curvature for NU=1)
 * - K_t = S^-1 * B^T * P * A (1x6 time-varying gain vector at timestep t)
 * - P_{t-1} = Q_h + A^T * P * A - A^T * P * B * K_t
 *
 * Terminal Handover:
 * Seeding the backward Riccati pass with the infinite-horizon balance Riccati matrix P (from CARE) ensures
 * seamless, continuous handover from swing-up trajectory tracking to stationary upright stabilization.
 *
 * @param xs Nominal state sequence, length N + 1
 * @param us Nominal control sequence, length N
 * @param dt Knot spacing in seconds
 * @param Q 6x6 stage state penalty matrix (or diagonal 6-vector)
 * @param R Stage control penalty scalar
 * @param P_terminal 6x6 terminal Riccati matrix from upright LQR
 * @param options TVLQR options
 */
export function computeTVLQR(
  xs: State[],
  us: number[],
  dt: number,
  Q: number[] | number[][],
  R: number,
  P_terminal: number[][],
  options: TVLQROptions = {}
): number[][] {
  const { substeps = 5, plant = DEFAULT_PLANT_PARAMS } = options;
  const pre = getPrecomputed(plant);
  const N = us.length;
  const h = dt / substeps;

  // Discrete stage cost, scaled for the sub-stepped recursion: Qh = Q * h (6x6 row-major), Rh = R * h.
  const Qh = new Float64Array(NX * NX);
  if (Array.isArray(Q[0])) {
    for (let i = 0; i < NX; i++) {
      for (let j = 0; j < NX; j++) Qh[i * NX + j] = (Q as number[][])[i][j] * h;
    }
  } else {
    for (let i = 0; i < NX; i++) Qh[i * NX + i] = (Q as number[])[i] * h;
  }
  const Rh = R * h;

  // P and its scratch, all flat and allocated once.
  let P = new Float64Array(NX * NX);
  let Pnew = new Float64Array(NX * NX);
  for (let i = 0; i < NX; i++) {
    for (let j = 0; j < NX; j++) P[i * NX + j] = P_terminal[i][j];
  }
  const A = new Float64Array(NX * NX);
  const B = new Float64Array(NX);
  const PA = new Float64Array(NX * NX);
  const PB = new Float64Array(NX);
  const K = new Float64Array(NX);
  const ATPB = new Float64Array(NX);

  const Ks: number[][] = new Array(N);

  for (let t = N - 1; t >= 0; t--) {
    for (let s = substeps - 1; s >= 0; s--) {
      linearizeDiscreteInto(A, B, xs[t], us[t], h, plant, pre);

      // PA = P * A (6x6),  PB = P * B (6x1)
      for (let i = 0; i < NX; i++) {
        for (let j = 0; j < NX; j++) {
          let acc = 0;
          for (let k = 0; k < NX; k++) acc += P[i * NX + k] * A[k * NX + j];
          PA[i * NX + j] = acc;
        }
        let acc = 0;
        for (let k = 0; k < NX; k++) acc += P[i * NX + k] * B[k];
        PB[i] = acc;
      }

      // S = Rh + B^T * P * B (scalar control curvature)
      let S = Rh;
      for (let k = 0; k < NX; k++) S += B[k] * PB[k];

      // Time-varying feedback gain: K = S^-1 * B^T * P * A (1x6)
      for (let j = 0; j < NX; j++) {
        let acc = 0;
        for (let k = 0; k < NX; k++) acc += B[k] * PA[k * NX + j];
        K[j] = acc / S;
      }

      if (s === 0) Ks[t] = Array.from(K);

      // Riccati update: P_prev = Qh + A^T * P * A - A^T * P * B * K
      for (let i = 0; i < NX; i++) {
        let acc = 0;
        for (let k = 0; k < NX; k++) acc += A[k * NX + i] * PB[k];
        ATPB[i] = acc;
      }
      for (let i = 0; i < NX; i++) {
        for (let j = 0; j < NX; j++) {
          let atpa = 0;
          for (let k = 0; k < NX; k++) atpa += A[k * NX + i] * PA[k * NX + j];
          Pnew[i * NX + j] = Qh[i * NX + j] + atpa - ATPB[i] * K[j];
        }
      }

      // Symmetrize P to prevent numerical drift
      for (let i = 0; i < NX; i++) {
        for (let j = i; j < NX; j++) {
          const val = 0.5 * (Pnew[i * NX + j] + Pnew[j * NX + i]);
          Pnew[i * NX + j] = val;
          Pnew[j * NX + i] = val;
        }
      }

      [P, Pnew] = [Pnew, P];
    }
  }

  return Ks;
}
