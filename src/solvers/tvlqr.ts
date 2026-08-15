import { State, STATE_DIM, PlantParams, DEFAULT_PLANT_PARAMS } from '../core/types.js';
import { linearizeDiscrete } from './ilqr.js';
import { zeros, zeros2, cloneMatrix, transpose, matMul } from '../core/math.js';

export interface TVLQROptions {
  /** Substeps per knot interval to prevent discretization distortion, default 5 */
  substeps?: number;
  /** Plant parameters */
  plant?: PlantParams;
}

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
  const N = us.length;
  const h = dt / substeps;

  // Build 6x6 Q matrix
  const Qmat = zeros2(STATE_DIM, STATE_DIM);
  if (Array.isArray(Q[0])) {
    for (let i = 0; i < STATE_DIM; i++) {
      for (let j = 0; j < STATE_DIM; j++) {
        Qmat[i][j] = (Q as number[][])[i][j];
      }
    }
  } else {
    for (let i = 0; i < STATE_DIM; i++) {
      Qmat[i][i] = (Q as number[])[i];
    }
  }

  // Discrete cost scaling for sub-stepped recursion
  const Qh = zeros2(STATE_DIM, STATE_DIM);
  for (let i = 0; i < STATE_DIM; i++) {
    for (let j = 0; j < STATE_DIM; j++) {
      Qh[i][j] = Qmat[i][j] * h;
    }
  }
  const Rh = R * h;

  let P = cloneMatrix(P_terminal);
  const Ks: number[][] = Array.from({ length: N }, () => zeros(STATE_DIM));

  for (let t = N - 1; t >= 0; t--) {
    const x_t = xs[t];
    const u_t = us[t];

    for (let s = substeps - 1; s >= 0; s--) {
      const { fx: A, fu: B } = linearizeDiscrete(x_t, u_t, h, 1e-6, plant);
      const AT = transpose(A);
      const BT = transpose(B);

      // PA = P * A (6x6)
      const PA = matMul(P, A);
      // PB = P * B (6x1)
      const PB = matMul(P, B);

      // S = Rh + B^T * P * B (scalar control curvature)
      let S = Rh;
      for (let k = 0; k < STATE_DIM; k++) {
        S += BT[0][k] * PB[k][0];
      }

      // Time-varying feedback gain: K = S^-1 * B^T * P * A (1x6)
      const K = zeros(STATE_DIM);
      for (let j = 0; j < STATE_DIM; j++) {
        let acc = 0;
        for (let k = 0; k < STATE_DIM; k++) {
          acc += BT[0][k] * PA[k][j];
        }
        K[j] = acc / S;
      }

      if (s === 0) {
        Ks[t] = K.slice();
      }

      // Riccati update: P_prev = Qh + A^T * P * A - A^T * P * B * K
      const AT_PA = matMul(AT, PA);
      const AT_PB = matMul(AT, PB);

      const P_new = zeros2(STATE_DIM, STATE_DIM);
      for (let i = 0; i < STATE_DIM; i++) {
        for (let j = 0; j < STATE_DIM; j++) {
          P_new[i][j] = Qh[i][j] + AT_PA[i][j] - AT_PB[i][0] * K[j];
        }
      }

      // Symmetrize P to prevent numerical drift
      for (let i = 0; i < STATE_DIM; i++) {
        for (let j = i; j < STATE_DIM; j++) {
          const val = 0.5 * (P_new[i][j] + P_new[j][i]);
          P_new[i][j] = val;
          P_new[j][i] = val;
        }
      }

      P = P_new;
    }
  }

  return Ks;
}
