import {
  State,
  STATE_DIM,
  CONTROL_DIM,
  CostFunction,
  TrajectoryResult,
  PlantParams,
  DEFAULT_PLANT_PARAMS,
} from '../core/types.js';
import { rk4 } from '../core/dynamics.js';
import { zeros, zeros2 } from '../core/math.js';

export interface ILQROptions {
  /** Maximum number of optimization iterations, default 500 */
  maxIter?: number;
  /** Cost reduction convergence threshold, default 1e-7 */
  tol?: number;
  /** Initial Levenberg-Marquardt regularization parameter mu, default 1.0 */
  muInit?: number;
  /** Minimum regularization parameter, default 1e-6 */
  muMin?: number;
  /** Maximum regularization parameter before giving up, default 1e10 */
  muMax?: number;
  /** Regularization scale factor, default 2.0 */
  muFactor?: number;
  /** Backtracking line search step ratios (alphas) */
  alphas?: number[];
  /** Whether to log iteration progress */
  verbose?: boolean;
  /** Optional callback after each improved step */
  onIter?: (info: { iter: number; cost: number; xs: State[]; us: number[] }) => void;
  /** Optional custom plant parameters */
  plant?: PlantParams;
}

const DEFAULT_ALPHAS = [1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005];

/**
 * Finite-difference Jacobians of the discrete RK4 step at (s, u):
 * - fx = ∂f/∂x (6x6): Sensitivity of next state to current state
 * - fu = ∂f/∂u (6x1): Sensitivity of next state to control input
 */
export function linearizeDiscrete(
  s: State,
  u: number,
  dt: number,
  eps: number = 1e-6,
  p: PlantParams = DEFAULT_PLANT_PARAMS
): { fx: number[][]; fu: number[][] } {
  const fx = zeros2(STATE_DIM, STATE_DIM);
  const fu = zeros2(STATE_DIM, CONTROL_DIM);

  for (let j = 0; j < STATE_DIM; j++) {
    const sp = [...s] as State;
    const sm = [...s] as State;
    sp[j] += eps;
    sm[j] -= eps;
    const a = rk4(sp, u, dt, p);
    const b = rk4(sm, u, dt, p);
    for (let i = 0; i < STATE_DIM; i++) {
      fx[i][j] = (a[i] - b[i]) / (2 * eps);
    }
  }

  const a = rk4(s, u + eps, dt, p);
  const b = rk4(s, u - eps, dt, p);
  for (let i = 0; i < STATE_DIM; i++) {
    fu[i][0] = (a[i] - b[i]) / (2 * eps);
  }

  return { fx, fu };
}

/**
 * Rolls out nominal trajectory under control sequence us from initial state s0.
 */
export function rollout(
  s0: State,
  us: number[],
  dt: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS
): State[] {
  const N = us.length;
  const xs: State[] = [[...s0] as State];
  for (let t = 0; t < N; t++) {
    xs.push(rk4(xs[t], us[t], dt, p));
  }
  return xs;
}

/**
 * Evaluates the total trajectory cost J = sum_{t=0}^{N-1} l(x_t, u_t) + l_f(x_N).
 */
export function totalCost(xs: State[], us: number[], cost: CostFunction): number {
  let J = 0;
  const N = us.length;
  for (let t = 0; t < N; t++) {
    J += cost.run(xs[t], us[t], t);
  }
  J += cost.term(xs[N], N);
  return J;
}

/**
 * Iterative Linear Quadratic Regulator (iLQR / Gauss-Newton DDP).
 *
 * Solves the non-linear optimal control problem:
 *   min_{u_0...u_{N-1}} sum l(x_t, u_t) + l_f(x_N)
 *   s.t. x_{t+1} = f(x_t, u_t), x_0 = s0
 *
 * Notation Guide:
 * - Vx (6x1): Gradient of remaining cost w.r.t. state (∂V/∂x)
 * - Vxx (6x6): Curvature/Hessian of remaining cost w.r.t. state (∂²V/∂x²)
 * - Qx, Qu: Gradients of action-value Q-function
 * - Qxx, Qux, Quu: Curvature of action-value Q-function (Quu is scalar for 1 actuator)
 * - mu: Levenberg-Marquardt damping parameter ensuring positive curvature (Quu + mu > 0)
 * - k_t: Feedforward adjustment to nominal control force (scalar)
 * - K_t (1x6): Feedback gain matrix correcting real-time state deviations
 * - alpha: Backtracking line search step fraction
 */
export function ilqr(
  s0: State,
  usInit: number[],
  dt: number,
  cost: CostFunction,
  options: ILQROptions = {}
): TrajectoryResult {
  const {
    maxIter = 500,
    tol = 1e-7,
    muInit = 1.0,
    muMin = 1e-6,
    muMax = 1e10,
    muFactor = 2.0,
    alphas = DEFAULT_ALPHAS,
    verbose = false,
    onIter = null,
    plant = DEFAULT_PLANT_PARAMS,
  } = options;

  const N = usInit.length;
  let us = usInit.slice();
  let xs = rollout(s0, us, dt, plant);
  let J = totalCost(xs, us, cost);
  let mu = muInit;

  let Ks = Array.from({ length: N }, () => zeros(STATE_DIM));
  let ks = zeros(N);
  let converged = false;
  let iter = 0;

  for (iter = 0; iter < maxIter; iter++) {
    // ---- 1. Backward Pass (Compute Value function & Control Policy) ----
    // Initialize Value function derivatives at terminal knot N using terminal cost l_f(x_N)
    const term = cost.termDeriv(xs[N], N);
    const Vx = Array.from(term.lx); // Vx: ∂l_f/∂x (6x1)
    const Vxx = zeros2(STATE_DIM, STATE_DIM); // Vxx: ∂²l_f/∂x² (6x6)
    for (let i = 0; i < STATE_DIM; i++) {
      Vxx[i][i] = term.lxxDiag[i];
    }

    const KsNew = Array.from({ length: N }, () => zeros(STATE_DIM));
    const ksNew = zeros(N);
    let dV1 = 0; // 1st-order expected cost reduction
    let dV2 = 0; // 2nd-order expected cost reduction
    let passOk = true;

    for (let t = N - 1; t >= 0; t--) {
      // Linearize discrete dynamics at current knot: x_{t+1} ≈ fx * δx + fu * δu
      const { fx, fu } = linearizeDiscrete(xs[t], us[t], dt, 1e-6, plant);
      const stage = cost.runDeriv(xs[t], us[t], t);

      // Qx = lx + fx^T * Vx (State gradient of Q-function)
      const Qx = zeros(STATE_DIM);
      for (let i = 0; i < STATE_DIM; i++) {
        let acc = stage.lx[i];
        for (let k = 0; k < STATE_DIM; k++) {
          acc += fx[k][i] * Vx[k];
        }
        Qx[i] = acc;
      }

      // Qu = lu + fu^T * Vx (Control gradient of Q-function, scalar)
      let Qu = stage.lu;
      for (let k = 0; k < STATE_DIM; k++) {
        Qu += fu[k][0] * Vx[k];
      }

      // Intermediate matrix: Vxx * fx (6x6)
      const VxxFx = zeros2(STATE_DIM, STATE_DIM);
      for (let i = 0; i < STATE_DIM; i++) {
        for (let j = 0; j < STATE_DIM; j++) {
          let acc = 0;
          for (let k = 0; k < STATE_DIM; k++) {
            acc += Vxx[i][k] * fx[k][j];
          }
          VxxFx[i][j] = acc;
        }
      }

      // Intermediate vector: Vxx * fu (6x1)
      const VxxFu = zeros(STATE_DIM);
      for (let i = 0; i < STATE_DIM; i++) {
        let acc = 0;
        for (let k = 0; k < STATE_DIM; k++) {
          acc += Vxx[i][k] * fu[k][0];
        }
        VxxFu[i] = acc;
      }

      // Qxx = lxx + fx^T * Vxx * fx (State-state curvature of Q)
      const Qxx = zeros2(STATE_DIM, STATE_DIM);
      for (let i = 0; i < STATE_DIM; i++) {
        for (let j = 0; j < STATE_DIM; j++) {
          let acc = i === j ? stage.lxxDiag[i] : 0;
          for (let k = 0; k < STATE_DIM; k++) {
            acc += fx[k][i] * VxxFx[k][j];
          }
          Qxx[i][j] = acc;
        }
      }

      // Qux = fu^T * Vxx * fx (Cross-coupling curvature, 1x6)
      const Qux = zeros(STATE_DIM);
      for (let j = 0; j < STATE_DIM; j++) {
        let acc = 0;
        for (let k = 0; k < STATE_DIM; k++) {
          acc += fu[k][0] * VxxFx[k][j];
        }
        Qux[j] = acc;
      }

      // Quu = luu + fu^T * Vxx * fu (Control-control curvature, scalar)
      let Quu = stage.luu;
      for (let k = 0; k < STATE_DIM; k++) {
        Quu += fu[k][0] * VxxFu[k];
      }

      // Levenberg-Marquardt regularization: ensure strictly positive curvature
      const QuuReg = Quu + mu;
      if (QuuReg <= 1e-12) {
        passOk = false;
        break;
      }

      // Compute optimal feedforward (k_t) and feedback gain (K_t)
      // k_t = -Qu / (Quu + mu)
      // K_t = -Qux / (Quu + mu)
      const k_t = -Qu / QuuReg;
      const K_t = Qux.map((z) => -z / QuuReg);
      ksNew[t] = k_t;
      KsNew[t] = K_t;

      // Accumulate expected cost improvements
      dV1 += k_t * Qu;
      dV2 += 0.5 * k_t * k_t * Quu;

      // Propagate Value function backwards to previous knot:
      // Vx = Qx + K_t^T * Quu * k_t + K_t^T * Qu + Qux^T * k_t
      for (let i = 0; i < STATE_DIM; i++) {
        Vx[i] = Qx[i] + K_t[i] * Quu * k_t + K_t[i] * Qu + Qux[i] * k_t;
      }

      // Vxx = Qxx + K_t^T * Quu * K_t + K_t^T * Qux + Qux^T * K_t (symmetrized)
      const VxxNew = zeros2(STATE_DIM, STATE_DIM);
      for (let i = 0; i < STATE_DIM; i++) {
        for (let j = 0; j < STATE_DIM; j++) {
          VxxNew[i][j] = Qxx[i][j] + K_t[i] * Quu * K_t[j] + K_t[i] * Qux[j] + Qux[i] * K_t[j];
        }
      }
      for (let i = 0; i < STATE_DIM; i++) {
        for (let j = i; j < STATE_DIM; j++) {
          const val = 0.5 * (VxxNew[i][j] + VxxNew[j][i]);
          Vxx[i][j] = val;
          Vxx[j][i] = val;
        }
      }
    }

    if (!passOk) {
      // Step failed positive-definiteness: increase damping mu and retry backward pass
      mu = Math.min(muMax, mu * muFactor);
      if (mu >= muMax) break;
      continue;
    }

    // ---- 2. Forward Pass with Backtracking Line Search ----
    let improved = false;
    let bestJ = J;
    let bestXs = xs;
    let bestUs = us;

    for (const a of alphas) {
      const xsNew: State[] = [[...s0] as State];
      const usNew = zeros(N);
      let invalid = false;

      for (let t = 0; t < N; t++) {
        // Control update: u_new = u_nominal + α * k_t + K_t * (x_new - x_nominal)
        let du = a * ksNew[t];
        for (let i = 0; i < STATE_DIM; i++) {
          du += KsNew[t][i] * (xsNew[t][i] - xs[t][i]);
        }
        const u = us[t] + du;
        if (!Number.isFinite(u)) {
          invalid = true;
          break;
        }
        usNew[t] = u;
        const nextState = rk4(xsNew[t], u, dt, plant);
        if (!nextState.every(Number.isFinite)) {
          invalid = true;
          break;
        }
        xsNew.push(nextState);
      }

      if (invalid) continue;

      const Jnew = totalCost(xsNew, usNew, cost);
      const expected = -(a * dV1 + a * a * dV2);
      const ratio = expected > 0 ? (J - Jnew) / expected : J - Jnew;

      // Armijo condition check
      if (Jnew < J && (expected <= 0 || ratio > 1e-4)) {
        bestJ = Jnew;
        bestXs = xsNew;
        bestUs = usNew;
        improved = true;
        break;
      }
    }

    if (improved) {
      const dJ = J - bestJ;
      xs = bestXs;
      us = bestUs;
      J = bestJ;
      Ks = KsNew;
      ks = ksNew;

      if (onIter) onIter({ iter, cost: J, xs, us });
      mu = Math.max(muMin, mu / muFactor); // Decrease damping on success

      if (verbose && iter % 25 === 0) {
        console.log(`[iLQR] iter ${iter}: Cost=${J.toFixed(4)}, mu=${mu.toExponential(1)}`);
      }

      if (dJ < tol) {
        converged = true;
        break;
      }
    } else {
      // Step rejected: increase damping mu and retry
      mu = Math.min(muMax, mu * muFactor);
      if (mu >= muMax) break;
    }
  }

  return { xs, us, Ks, ks, cost: J, iters: iter, converged };
}
