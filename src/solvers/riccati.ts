import { State, STATE_DIM, CONTROL_DIM, PlantParams, DEFAULT_PLANT_PARAMS } from '../core/types.js';
import { dynamics } from '../core/dynamics.js';
import { zeros2, transpose, matMul, matAdd, matScale } from '../core/math.js';

export interface Linearization {
  /** Continuous state matrix A = ∂f/∂x (6x6) */
  A: number[][];
  /** Continuous control matrix B = ∂f/∂u (6x1) */
  B: number[][];
}

/**
 * Finite-difference continuous-time linearization about an equilibrium or arbitrary state (s, u).
 * Returns continuous-time A = ∂f/∂x (6x6) and B = ∂f/∂u (6x1).
 */
export function linearizeContinuous(
  s: State,
  u: number = 0,
  eps: number = 1e-6,
  p: PlantParams = DEFAULT_PLANT_PARAMS
): Linearization {
  const A = zeros2(STATE_DIM, STATE_DIM);
  const B = zeros2(STATE_DIM, CONTROL_DIM);

  for (let j = 0; j < STATE_DIM; j++) {
    const sp: State = [...s] as State;
    const sm: State = [...s] as State;
    sp[j] += eps;
    sm[j] -= eps;
    const a = dynamics(sp, u, p);
    const b = dynamics(sm, u, p);
    for (let i = 0; i < STATE_DIM; i++) {
      A[i][j] = (a[i] - b[i]) / (2 * eps);
    }
  }

  const a = dynamics(s, u + eps, p);
  const b = dynamics(s, u - eps, p);
  for (let i = 0; i < STATE_DIM; i++) {
    B[i][0] = (a[i] - b[i]) / (2 * eps);
  }

  return { A, B };
}

export interface CareOptions {
  /** Numerical integration step size in seconds (h), default 2e-4 */
  h?: number;
  /** Maximum integration steps, default 400000 */
  maxSteps?: number;
  /** Convergence tolerance on dP/dt, default 1e-11 */
  tol?: number;
}

/**
 * Solves the Continuous Algebraic Riccati Equation (CARE):
 *   A^T * P + P * A - P * B * R^-1 * B^T * P + Q = 0
 *
 * Method: Integrates the differential Riccati matrix equation backwards in time
 *   dP/dt = -(A^T * P + P * A + Q - P * B * R^-1 * B^T * P)
 * until dP/dt reaches steady state (P converges).
 *
 * Variables:
 * - A (6x6): Continuous-time state transition matrix
 * - B (6x1): Continuous-time control input matrix
 * - Q (6x6): Quadratic state error penalty matrix
 * - R (scalar): Quadratic control effort penalty scalar
 * - P (6x6): Steady-state Riccati cost-to-go matrix (V(x) = 0.5 * x^T * P * x)
 * - K (1x6): Optimal linear feedback gain vector (u = -K * x)
 */
export function solveCARE(
  A: number[][],
  B: number[][],
  Q: number[][],
  R: number,
  options: CareOptions = {}
): { P: number[][]; K: number[] } {
  const { h = 2e-4, maxSteps = 400000, tol = 1e-11 } = options;

  const BT = transpose(B);
  let P = zeros2(STATE_DIM, STATE_DIM);

  // Riccati ODE: dP/dt = -(A^T*P + P*A + Q - P*B*R^-1*B^T*P)
  const f = (pMat: number[][]): number[][] => {
    const PB = matMul(pMat, B);
    const PBRBP = matScale(matMul(PB, matMul(BT, pMat)), 1 / R);
    const AT_P = matMul(transpose(A), pMat);
    const P_A = matMul(pMat, A);
    const sum = matAdd(matAdd(AT_P, P_A), matAdd(Q, matScale(PBRBP, -1)));
    return matScale(sum, -1);
  };

  const step = -h; // integrate backward in time
  for (let i = 0; i < maxSteps; i++) {
    const k1 = f(P);
    const k2 = f(matAdd(P, matScale(k1, step / 2)));
    const k3 = f(matAdd(P, matScale(k2, step / 2)));
    const k4 = f(matAdd(P, matScale(k3, step)));

    const delta = matScale(
      matAdd(matAdd(k1, matScale(k2, 2)), matAdd(matScale(k3, 2), k4)),
      step / 6
    );
    const Pn = matAdd(P, delta);

    if (i % 2000 === 0 && i > 0) {
      let maxDiff = 0;
      for (let r = 0; r < STATE_DIM; r++) {
        for (let c = 0; c < STATE_DIM; c++) {
          maxDiff = Math.max(maxDiff, Math.abs(Pn[r][c] - P[r][c]));
        }
      }
      if (maxDiff < tol) {
        P = Pn;
        break;
      }
    }
    P = Pn;
  }

  // Optimal feedback gain: K = R^-1 * B^T * P (1x6 vector)
  const K = matMul(BT, P)[0].map((v) => v / R);
  return { P, K };
}
