import {
  State,
  STATE_DIM,
  STATE_UPRIGHT,
  STATE_HANGING,
  LQRResult,
  PlantParams,
  DEFAULT_PLANT_PARAMS,
} from '../core/types.js';
import { linearizeContinuous, solveCARE } from '../solvers/riccati.js';
import { zeros2, wrapPi } from '../core/math.js';

export const DEFAULT_Q_BALANCE = [8, 6, 150, 12, 150, 12];
export const DEFAULT_R_BALANCE = 0.1;

export const DEFAULT_Q_BRAKE = [50, 1, 0.5, 0.5, 0.5, 0.5];
export const DEFAULT_R_BRAKE = 0.1;

/**
 * Computes infinite-horizon continuous-time LQR gains for balancing at the upright equilibrium.
 */
export function computeBalanceLQR(
  Q_diag: number[] = DEFAULT_Q_BALANCE,
  R: number = DEFAULT_R_BALANCE,
  p: PlantParams = DEFAULT_PLANT_PARAMS
): LQRResult {
  const { A, B } = linearizeContinuous(STATE_UPRIGHT, 0, 1e-6, p);
  const Q = zeros2(STATE_DIM, STATE_DIM);
  for (let i = 0; i < STATE_DIM; i++) {
    Q[i][i] = Q_diag[i];
  }
  const { P, K } = solveCARE(A, B, Q, R);
  return { K, P, A, B };
}

/**
 * Computes infinite-horizon continuous-time LQR gains for braking at the hanging equilibrium.
 * Linearizing about hanging and penalizing cart displacement allows the system to shed energy
 * and park cleanly at the origin without manual mode switches.
 */
export function computeBrakeLQR(
  Q_diag: number[] = DEFAULT_Q_BRAKE,
  R: number = DEFAULT_R_BRAKE,
  p: PlantParams = DEFAULT_PLANT_PARAMS
): LQRResult {
  const { A, B } = linearizeContinuous(STATE_HANGING, 0, 1e-6, p);
  const Q = zeros2(STATE_DIM, STATE_DIM);
  for (let i = 0; i < STATE_DIM; i++) {
    Q[i][i] = Q_diag[i];
  }
  const { P, K } = solveCARE(A, B, Q, R);
  return { K, P, A, B };
}

/**
 * Evaluates state-feedback control law: u = -K * (s - s_target) with proper angle wrapping.
 */
export function evaluateLQR(s: State, K: number[], s_target: State = STATE_UPRIGHT): number {
  let u = 0;
  for (let i = 0; i < STATE_DIM; i++) {
    let err = s[i] - s_target[i];
    if (i === 2 || i === 4) {
      err = wrapPi(err);
    }
    u -= K[i] * err;
  }
  return u;
}
