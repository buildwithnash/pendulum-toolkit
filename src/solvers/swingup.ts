import {
  State,
  STATE_DIM,
  STATE_HANGING,
  CostFunction,
  TrajectoryResult,
  PlantParams,
  DEFAULT_PLANT_PARAMS,
} from '../core/types.js';
import { ilqr } from './ilqr.js';

/**
 * Terminal-weight continuation schedule for the swing-up solve.
 *
 * Asking for a perfect landing from the first iteration stalls in a local minimum where the
 * linkage flails without committing to a swing. Each stage tightens the terminal penalty starting
 * from the previous solution. Weights are per state: [x, v, θ1, ω1, θ2, ω2].
 */
export const DEFAULT_QF_STAGES: number[][] = [
  [20, 20, 200, 20, 200, 20],
  [60, 60, 800, 80, 800, 80],
  [200, 200, 3000, 300, 3000, 300],
  [600, 600, 12000, 1200, 12000, 1200],
];

/** Weights for the swing-up cost that are shared by every continuation stage. */
export interface SwingUpWeights {
  /** Control effort penalty, default 1e-4 */
  R: number;
  /** Running state penalty [x, v, θ1, ω1, θ2, ω2], default [0.02, 0.002, 0, 0, 0, 0] */
  Qrun: number[];
  /** Soft cart travel limit (m), default 2.5 */
  xLim: number;
  /** Soft actuator force limit (N), default 220 */
  uLim: number;
  /** Penalty weight past the travel limit, default 60 */
  wTrack: number;
  /** Penalty weight past the force limit, default 0.05 */
  wForce: number;
}

export const DEFAULT_SWINGUP_WEIGHTS: SwingUpWeights = {
  R: 1e-4,
  Qrun: [0.02, 0.002, 0, 0, 0, 0],
  xLim: 2.5,
  uLim: 220,
  wTrack: 60,
  wForce: 0.05,
};

const hinge = (z: number, lim: number) => Math.max(0, Math.abs(z) - lim) * Math.sign(z);

/**
 * Swing-up cost: a small running cost on cart position and speed, a control-effort penalty,
 * quadratic soft barriers on cart travel and force, and a heavy terminal cost `Qf` pulling the
 * state onto the upright equilibrium (all angles measured from upright, so the goal is the origin).
 *
 * The barriers are quadratic hinges, so the cost stays C¹, which is all Gauss-Newton needs.
 */
export function createSwingUpCost(
  Qf: number[],
  weights: Partial<SwingUpWeights> = {}
): CostFunction {
  const { R, Qrun, xLim, uLim, wTrack, wForce } = { ...DEFAULT_SWINGUP_WEIGHTS, ...weights };

  const lxRun = new Float64Array(STATE_DIM);
  const lxxRun = new Float64Array(STATE_DIM);
  const lxTerm = new Float64Array(STATE_DIM);
  const lxxTerm = new Float64Array(STATE_DIM);
  const retRun = { lx: lxRun, lxxDiag: lxxRun, lu: 0, luu: R };
  const retTerm = { lx: lxTerm, lxxDiag: lxxTerm };

  return {
    run(s: State, u: number): number {
      let c = 0.5 * R * u * u;
      for (let i = 0; i < STATE_DIM; i++) c += 0.5 * Qrun[i] * s[i] * s[i];
      c += 0.5 * wTrack * hinge(s[0], xLim) ** 2;
      c += 0.5 * wForce * hinge(u, uLim) ** 2;
      return c;
    },

    term(s: State): number {
      let c = 0;
      for (let i = 0; i < STATE_DIM; i++) c += 0.5 * Qf[i] * s[i] * s[i];
      return c;
    },

    runDeriv(s: State, u: number) {
      for (let i = 0; i < STATE_DIM; i++) {
        lxRun[i] = Qrun[i] * s[i];
        lxxRun[i] = Qrun[i];
      }
      const hx = hinge(s[0], xLim);
      lxRun[0] += wTrack * hx;
      if (hx !== 0) lxxRun[0] += wTrack;

      const hu = hinge(u, uLim);
      retRun.lu = R * u + wForce * hu;
      retRun.luu = R + (hu !== 0 ? wForce : 0);
      return retRun;
    },

    termDeriv(s: State) {
      for (let i = 0; i < STATE_DIM; i++) {
        lxTerm[i] = Qf[i] * s[i];
        lxxTerm[i] = Qf[i];
      }
      return retTerm;
    },
  };
}

/** Small deterministic PRNG (mulberry32), so a given seed always gives the same initial guess. */
export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface SwingUpOptions {
  /** Swing-up duration in seconds, default 4.0 */
  duration?: number;
  /** Knot spacing in seconds, default 0.02 */
  dt?: number;
  /** Seed for the random initial force guess, default 7 */
  seed?: number;
  /** Amplitude (N) of the random initial force guess, default 40 */
  initAmplitude?: number;
  /** Terminal-weight continuation schedule, default DEFAULT_QF_STAGES */
  stages?: number[][];
  /** Cost weights shared by every stage */
  weights?: Partial<SwingUpWeights>;
  /** Iteration cap per stage, default 1000 */
  maxIterPerStage?: number;
  /** Convergence tolerance on cost reduction per stage, default 1e-8 */
  tol?: number;
  /** Start state, default hanging still */
  start?: State;
  plant?: PlantParams;
  /** Passed to iLQR: 'analytic' (default) or 'finite-difference' (see ILQROptions.linearization) */
  linearization?: 'analytic' | 'finite-difference';
}

export interface SwingUpStageInfo {
  stage: number;
  iters: number;
  converged: boolean;
  cost: number;
}

export interface SwingUpResult extends TrajectoryResult {
  /** Per-stage iteration counts and costs (each stage is scored against its own terminal weight) */
  stages: SwingUpStageInfo[];
  /** Iterations summed over all stages */
  totalIters: number;
}

/**
 * Solves the swing-up from hanging to upright with multi-stage iLQR.
 *
 * The initial force guess is uniform random noise, and iLQR is a local optimizer, so different seeds
 * can settle into different local optima.
 */
export function solveSwingUp(options: SwingUpOptions = {}): SwingUpResult {
  const {
    duration = 4.0,
    dt = 0.02,
    seed = 7,
    initAmplitude = 40,
    stages = DEFAULT_QF_STAGES,
    weights = {},
    maxIterPerStage = 1000,
    tol = 1e-8,
    start = STATE_HANGING,
    plant = DEFAULT_PLANT_PARAMS,
    linearization = 'analytic',
  } = options;

  const N = Math.round(duration / dt);
  const rng = mulberry32(seed);
  let us = Array.from({ length: N }, () => (rng() * 2 - 1) * initAmplitude);

  let result: TrajectoryResult | null = null;
  const info: SwingUpStageInfo[] = [];
  let totalIters = 0;

  stages.forEach((Qf, i) => {
    const cost = createSwingUpCost(Qf, weights);
    result = ilqr(start, us, dt, cost, { maxIter: maxIterPerStage, tol, plant, linearization });
    us = result.us;
    totalIters += result.iters;
    info.push({
      stage: i + 1,
      iters: result.iters,
      converged: result.converged,
      cost: result.cost,
    });
  });

  if (!result) throw new Error('solveSwingUp requires at least one continuation stage');
  return { ...(result as TrajectoryResult), stages: info, totalIters };
}
