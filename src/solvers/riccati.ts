import { State, STATE_DIM, CONTROL_DIM, PlantParams, DEFAULT_PLANT_PARAMS } from '../core/types.js';
import { dynamics, dynamicsJacobianInto } from '../core/dynamics.js';
import { zeros2 } from '../core/math.js';
import { frobenius, invert, solveLeastSquares } from '../core/linalg.js';

export interface Linearization {
  /** Continuous state matrix A = ∂f/∂x (6x6) */
  A: number[][];
  /** Continuous control matrix B = ∂f/∂u (6x1) */
  B: number[][];
}

/** Analytic continuous-time Jacobians at (s, u): A = ∂f/∂x (6x6) and B = ∂f/∂u (6x1). */
export function linearizeContinuous(
  s: State,
  u: number = 0,
  p: PlantParams = DEFAULT_PLANT_PARAMS
): Linearization {
  const Aflat = new Float64Array(STATE_DIM * STATE_DIM);
  const Bflat = new Float64Array(STATE_DIM);
  dynamicsJacobianInto(Aflat, Bflat, s, u, p);

  const A = zeros2(STATE_DIM, STATE_DIM);
  const B = zeros2(STATE_DIM, CONTROL_DIM);
  for (let i = 0; i < STATE_DIM; i++) {
    for (let j = 0; j < STATE_DIM; j++) A[i][j] = Aflat[i * STATE_DIM + j];
    B[i][0] = Bflat[i];
  }
  return { A, B };
}

/** Central finite-difference version of {@link linearizeContinuous}, kept as a reference. */
export function linearizeContinuousFD(
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
  /** Maximum sign-function Newton iterations, default 100 */
  maxIter?: number;
  /** Relative change in the iterate at which the sign function counts as converged, default 1e-12 */
  tol?: number;
}

/**
 * Solves the Continuous Algebraic Riccati Equation (CARE):
 *   A^T * P + P * A - P * B * R^-1 * B^T * P + Q = 0
 *
 * Method: matrix sign function. The stable invariant subspace of the Hamiltonian
 *
 *   H = [  A   -B R^-1 B^T ]
 *       [ -Q       -A^T    ]
 *
 * determines the stabilizing P. sign(H) comes from the scaled Newton iteration
 * Z <- (c Z + (c Z)^-1) / 2, and with S = sign(H) split into 6x6 blocks, P solves
 * [ S12; S22 + I ] P = -[ S11 + I; S21 ] by least squares.
 *
 * Throws when no stabilizing solution exists, i.e. (A, B) is not stabilizable or (Q, A) is not
 * detectable.
 *
 * Variables:
 * - A (6x6): Continuous-time state transition matrix
 * - B (6x1): Continuous-time control input matrix
 * - Q (6x6): Quadratic state error penalty matrix
 * - R (scalar): Quadratic control effort penalty scalar
 * - P (6x6): Stabilizing Riccati cost-to-go matrix (V(x) = 0.5 * x^T * P * x)
 * - K (1x6): Optimal linear feedback gain vector (u = -K * x)
 */
export function solveCARE(
  A: number[][],
  B: number[][],
  Q: number[][],
  R: number,
  options: CareOptions = {}
): { P: number[][]; K: number[] } {
  const { maxIter = 100, tol = 1e-12 } = options;
  const n = STATE_DIM;
  const n2 = 2 * n;

  // Hamiltonian, row-major 12x12.
  const H = new Float64Array(n2 * n2);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      H[i * n2 + j] = A[i][j];
      H[i * n2 + n + j] = -(B[i][0] * B[j][0]) / R;
      H[(n + i) * n2 + j] = -Q[i][j];
      H[(n + i) * n2 + n + j] = -A[j][i];
    }
  }

  // Scaled Newton iteration for sign(H).
  let Z = H;
  let converged = false;
  for (let iter = 0; iter < maxIter; iter++) {
    const Zinv = invert(Z, n2);
    const c = Math.sqrt(frobenius(Zinv) / frobenius(Z));
    const Znext = new Float64Array(n2 * n2);
    let diff = 0;
    for (let i = 0; i < Znext.length; i++) {
      Znext[i] = 0.5 * (c * Z[i] + Zinv[i] / c);
      diff += (Znext[i] - Z[i]) ** 2;
    }
    Z = Znext;
    if (Math.sqrt(diff) <= tol * frobenius(Z)) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    throw new Error(
      'solveCARE: sign-function iteration did not converge. Is (A, B) stabilizable and (Q, A) detectable?'
    );
  }

  // Solve [S12; S22 + I] P = -[S11 + I; S21] in the least-squares sense.
  const M = new Float64Array(n2 * n);
  const rhs = new Float64Array(n2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      M[i * n + j] = Z[i * n2 + n + j]; // S12
      M[(n + i) * n + j] = Z[(n + i) * n2 + n + j] + (i === j ? 1 : 0); // S22 + I
      rhs[i * n + j] = -(Z[i * n2 + j] + (i === j ? 1 : 0)); // -(S11 + I)
      rhs[(n + i) * n + j] = -Z[(n + i) * n2 + j]; // -S21
    }
  }
  const sol = solveLeastSquares(M, n2, n, rhs, n);

  const P = zeros2(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) P[i][j] = 0.5 * (sol[i * n + j] + sol[j * n + i]);
  }

  // Verify the residual so an ill-posed problem throws instead of returning a bad P.
  const PB = P.map((row) => row.reduce((acc, v, k) => acc + v * B[k][0], 0));
  let resid = 0;
  let scale = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let atp = 0;
      let pa = 0;
      for (let k = 0; k < n; k++) {
        atp += A[k][i] * P[k][j];
        pa += P[i][k] * A[k][j];
      }
      const r = atp + pa - (PB[i] * PB[j]) / R + Q[i][j];
      resid += r * r;
      scale += (atp + pa) ** 2 + Q[i][j] ** 2;
    }
  }
  if (Math.sqrt(resid) > 1e-8 * Math.max(1, Math.sqrt(scale))) {
    throw new Error(
      `solveCARE: residual ${Math.sqrt(resid).toExponential(2)} is too large. Is (A, B) stabilizable and (Q, A) detectable?`
    );
  }

  // Optimal feedback gain: K = R^-1 * B^T * P (1x6 vector)
  const K = PB.map((v) => v / R);
  return { P, K };
}
