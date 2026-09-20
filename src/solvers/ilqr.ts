import {
  State,
  STATE_DIM,
  CONTROL_DIM,
  CostFunction,
  TrajectoryResult,
  PlantParams,
  DEFAULT_PLANT_PARAMS,
} from '../core/types.js';
import {
  rk4,
  rk4Into,
  rk4JacobianInto,
  getPrecomputed,
  PrecomputedDynamics,
} from '../core/dynamics.js';
import { zeros2 } from '../core/math.js';

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
  /** Optional callback after each improved step. Receives copies, safe to keep. */
  onIter?: (info: { iter: number; cost: number; xs: State[]; us: number[] }) => void;
  /** Optional custom plant parameters */
  plant?: PlantParams;
  /**
   * How the backward pass linearizes the dynamics, default 'analytic'. 'finite-difference' is slower
   * and differs from the analytic Jacobian by ~1e-9, which can change which local optimum iLQR
   * finds. Use it to reproduce trajectories generated with it.
   */
  linearization?: 'analytic' | 'finite-difference';
}

const DEFAULT_ALPHAS = [1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005];

const NX = STATE_DIM;

/**
 * Jacobians of one RK4 step at (s, u), written into flat arrays: fx = ∂f/∂x (6x6 row-major) and
 * fu = ∂f/∂u (length 6). See {@link rk4JacobianInto}.
 */
export function linearizeDiscreteInto(
  fx: Float64Array,
  fu: Float64Array,
  s: ArrayLike<number>,
  u: number,
  dt: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  pre: PrecomputedDynamics = getPrecomputed(p)
): void {
  rk4JacobianInto(fx, fu, s, u, dt, p, pre);
}

/** {@link linearizeDiscreteInto} returning nested arrays: fx (6x6) and fu (6x1). */
export function linearizeDiscrete(
  s: State,
  u: number,
  dt: number,
  p: PlantParams = DEFAULT_PLANT_PARAMS
): { fx: number[][]; fu: number[][] } {
  const fxFlat = new Float64Array(NX * NX);
  const fuFlat = new Float64Array(NX);
  rk4JacobianInto(fxFlat, fuFlat, s, u, dt, p);

  const fx = zeros2(NX, NX);
  const fu = zeros2(NX, CONTROL_DIM);
  for (let i = 0; i < NX; i++) {
    for (let j = 0; j < NX; j++) fx[i][j] = fxFlat[i * NX + j];
    fu[i][0] = fuFlat[i];
  }
  return { fx, fu };
}

/**
 * Central finite-difference Jacobians of one RK4 step (14 RK4 steps per call). A reference to check
 * {@link linearizeDiscrete} against.
 */
export function linearizeDiscreteFD(
  s: State,
  u: number,
  dt: number,
  eps: number = 1e-6,
  p: PlantParams = DEFAULT_PLANT_PARAMS
): { fx: number[][]; fu: number[][] } {
  const fx = zeros2(NX, NX);
  const fu = zeros2(NX, CONTROL_DIM);

  for (let j = 0; j < NX; j++) {
    const sp = [...s] as State;
    const sm = [...s] as State;
    sp[j] += eps;
    sm[j] -= eps;
    const a = rk4(sp, u, dt, p);
    const b = rk4(sm, u, dt, p);
    for (let i = 0; i < NX; i++) {
      fx[i][j] = (a[i] - b[i]) / (2 * eps);
    }
  }

  const a = rk4(s, u + eps, dt, p);
  const b = rk4(s, u - eps, dt, p);
  for (let i = 0; i < NX; i++) {
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
  const pre = getPrecomputed(p);
  const xs: State[] = [[...s0] as State];
  for (let t = 0; t < N; t++) {
    const next: State = [0, 0, 0, 0, 0, 0];
    rk4Into(next, xs[t], us[t], dt, p, pre);
    xs.push(next);
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
    linearization = 'analytic',
  } = options;

  const useFiniteDifference = linearization === 'finite-difference';
  const pre = getPrecomputed(plant);
  const N = usInit.length;

  // Two trajectory buffers: the accepted one and the line-search candidate. They swap on success.
  const newTrajectory = () => Array.from({ length: N + 1 }, () => [0, 0, 0, 0, 0, 0] as State);
  let xs = newTrajectory();
  let xsNew = newTrajectory();
  let us = usInit.slice();
  let usNew = new Array<number>(N).fill(0);

  xs[0] = [...s0] as State;
  for (let t = 0; t < N; t++) rk4Into(xs[t + 1], xs[t], us[t], dt, plant, pre);
  let J = totalCost(xs, us, cost);
  let mu = muInit;

  // Accepted gains and the candidate gains from the current backward pass (flat, N x 6).
  let Ks = new Float64Array(N * NX);
  let ks = new Float64Array(N);
  let KsNew = new Float64Array(N * NX);
  let ksNew = new Float64Array(N);

  const fx = new Float64Array(NX * NX);
  const fu = new Float64Array(NX);
  const Vx = new Float64Array(NX);
  const Vxx = new Float64Array(NX * NX);
  const VxxNew = new Float64Array(NX * NX);
  const Qx = new Float64Array(NX);
  const Qxx = new Float64Array(NX * NX);
  const Qux = new Float64Array(NX);
  const VxxFx = new Float64Array(NX * NX);
  const VxxFu = new Float64Array(NX);
  const Kt = new Float64Array(NX);

  let converged = false;
  let iter = 0;

  for (iter = 0; iter < maxIter; iter++) {
    // ---- 1. Backward Pass (Compute Value function & Control Policy) ----
    // Initialize Value function derivatives at terminal knot N using terminal cost l_f(x_N)
    const term = cost.termDeriv(xs[N], N);
    Vxx.fill(0);
    for (let i = 0; i < NX; i++) {
      Vx[i] = term.lx[i];
      Vxx[i * NX + i] = term.lxxDiag[i];
    }

    let dV1 = 0; // 1st-order expected cost reduction
    let dV2 = 0; // 2nd-order expected cost reduction
    let passOk = true;

    for (let t = N - 1; t >= 0; t--) {
      // Linearize discrete dynamics at current knot: x_{t+1} ≈ fx * δx + fu * δu
      if (useFiniteDifference) {
        const fd = linearizeDiscreteFD(xs[t], us[t], dt, 1e-6, plant);
        for (let i = 0; i < NX; i++) {
          for (let j = 0; j < NX; j++) fx[i * NX + j] = fd.fx[i][j];
          fu[i] = fd.fu[i][0];
        }
      } else {
        rk4JacobianInto(fx, fu, xs[t], us[t], dt, plant, pre);
      }
      const stage = cost.runDeriv(xs[t], us[t], t);

      // Qx = lx + fx^T * Vx (State gradient of Q-function)
      for (let i = 0; i < NX; i++) {
        let acc = stage.lx[i];
        for (let k = 0; k < NX; k++) acc += fx[k * NX + i] * Vx[k];
        Qx[i] = acc;
      }

      // Qu = lu + fu^T * Vx (Control gradient of Q-function, scalar)
      let Qu = stage.lu;
      for (let k = 0; k < NX; k++) Qu += fu[k] * Vx[k];

      // Intermediate matrix: Vxx * fx (6x6)
      for (let i = 0; i < NX; i++) {
        for (let j = 0; j < NX; j++) {
          let acc = 0;
          for (let k = 0; k < NX; k++) acc += Vxx[i * NX + k] * fx[k * NX + j];
          VxxFx[i * NX + j] = acc;
        }
      }

      // Intermediate vector: Vxx * fu (6x1)
      for (let i = 0; i < NX; i++) {
        let acc = 0;
        for (let k = 0; k < NX; k++) acc += Vxx[i * NX + k] * fu[k];
        VxxFu[i] = acc;
      }

      // Qxx = lxx + fx^T * Vxx * fx (State-state curvature of Q)
      for (let i = 0; i < NX; i++) {
        for (let j = 0; j < NX; j++) {
          let acc = i === j ? stage.lxxDiag[i] : 0;
          for (let k = 0; k < NX; k++) acc += fx[k * NX + i] * VxxFx[k * NX + j];
          Qxx[i * NX + j] = acc;
        }
      }

      // Qux = fu^T * Vxx * fx (Cross-coupling curvature, 1x6)
      for (let j = 0; j < NX; j++) {
        let acc = 0;
        for (let k = 0; k < NX; k++) acc += fu[k] * VxxFx[k * NX + j];
        Qux[j] = acc;
      }

      // Quu = luu + fu^T * Vxx * fu (Control-control curvature, scalar)
      let Quu = stage.luu;
      for (let k = 0; k < NX; k++) Quu += fu[k] * VxxFu[k];

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
      ksNew[t] = k_t;
      for (let j = 0; j < NX; j++) {
        Kt[j] = -Qux[j] / QuuReg;
        KsNew[t * NX + j] = Kt[j];
      }

      // Accumulate expected cost improvements
      dV1 += k_t * Qu;
      dV2 += 0.5 * k_t * k_t * Quu;

      // Propagate Value function backwards to previous knot:
      // Vx = Qx + K_t^T * Quu * k_t + K_t^T * Qu + Qux^T * k_t
      for (let i = 0; i < NX; i++) {
        Vx[i] = Qx[i] + Kt[i] * Quu * k_t + Kt[i] * Qu + Qux[i] * k_t;
      }

      // Vxx = Qxx + K_t^T * Quu * K_t + K_t^T * Qux + Qux^T * K_t (symmetrized)
      for (let i = 0; i < NX; i++) {
        for (let j = 0; j < NX; j++) {
          VxxNew[i * NX + j] =
            Qxx[i * NX + j] + Kt[i] * Quu * Kt[j] + Kt[i] * Qux[j] + Qux[i] * Kt[j];
        }
      }
      for (let i = 0; i < NX; i++) {
        for (let j = i; j < NX; j++) {
          const val = 0.5 * (VxxNew[i * NX + j] + VxxNew[j * NX + i]);
          Vxx[i * NX + j] = val;
          Vxx[j * NX + i] = val;
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

    for (const a of alphas) {
      for (let i = 0; i < NX; i++) xsNew[0][i] = s0[i];
      let invalid = false;

      for (let t = 0; t < N; t++) {
        // Control update: u_new = u_nominal + α * k_t + K_t * (x_new - x_nominal)
        let du = a * ksNew[t];
        for (let i = 0; i < NX; i++) {
          du += KsNew[t * NX + i] * (xsNew[t][i] - xs[t][i]);
        }
        const u = us[t] + du;
        if (!Number.isFinite(u)) {
          invalid = true;
          break;
        }
        usNew[t] = u;
        rk4Into(xsNew[t + 1], xsNew[t], u, dt, plant, pre);
        const next = xsNew[t + 1];
        if (!(
          Number.isFinite(next[0]) &&
          Number.isFinite(next[1]) &&
          Number.isFinite(next[2]) &&
          Number.isFinite(next[3]) &&
          Number.isFinite(next[4]) &&
          Number.isFinite(next[5])
        )) {
          invalid = true;
          break;
        }
      }

      if (invalid) continue;

      const Jnew = totalCost(xsNew, usNew, cost);
      const expected = -(a * dV1 + a * a * dV2);
      const ratio = expected > 0 ? (J - Jnew) / expected : J - Jnew;

      // Armijo condition check
      if (Jnew < J && (expected <= 0 || ratio > 1e-4)) {
        bestJ = Jnew;
        improved = true;
        break;
      }
    }

    if (improved) {
      const dJ = J - bestJ;

      // Accept the candidate: swap buffers instead of copying.
      [xs, xsNew] = [xsNew, xs];
      [us, usNew] = [usNew, us];
      [Ks, KsNew] = [KsNew, Ks];
      [ks, ksNew] = [ksNew, ks];
      J = bestJ;

      if (onIter) {
        onIter({ iter, cost: J, xs: xs.map((x) => [...x] as State), us: us.slice() });
      }
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

  return {
    xs,
    us,
    Ks: Array.from({ length: N }, (_, t) => Array.from(Ks.subarray(t * NX, (t + 1) * NX))),
    ks: Array.from(ks),
    cost: J,
    iters: iter,
    converged,
  };
}
