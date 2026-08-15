import {
  State,
  STATE_DIM,
  CostFunction,
  PlantParams,
  DEFAULT_PLANT_PARAMS,
} from '../core/types.js';
import { ilqr } from '../solvers/ilqr.js';
import { wrapPi } from '../core/math.js';

export interface TrajectoryInterpolation {
  /** Gets reference state at timestamp t */
  getState(t: number): State;
  /** Gets reference feedforward control at timestamp t */
  getControl(t: number): number;
}

export interface TrackingCostWeights {
  Q?: number[];
  R?: number;
  Q_terminal_scale?: number;
}

/**
 * Constructs a time-varying quadratic tracking cost function penalizing deviation
 * from a pre-computed nominal trajectory.
 */
export function createTrackingCost(
  trajectory: TrajectoryInterpolation,
  tStart: number,
  dt: number,
  weights: TrackingCostWeights = {}
): CostFunction {
  const { Q = [10, 1, 200, 10, 200, 10], R = 0.5, Q_terminal_scale = 10.0 } = weights;

  const lxRun = new Float64Array(STATE_DIM);
  const lxxRun = new Float64Array(STATE_DIM);
  const lxTerm = new Float64Array(STATE_DIM);
  const lxxTerm = new Float64Array(STATE_DIM);

  for (let i = 0; i < STATE_DIM; i++) {
    lxxRun[i] = Q[i];
    lxxTerm[i] = Q[i] * Q_terminal_scale;
  }

  const retRun = { lx: lxRun, lxxDiag: lxxRun, lu: 0, luu: R };
  const retTerm = { lx: lxTerm, lxxDiag: lxxTerm };

  return {
    run(s: State, u: number, t: number): number {
      const globalTime = tStart + t * dt;
      const ref = trajectory.getState(globalTime);
      const uRef = trajectory.getControl(globalTime);

      let c = 0.5 * R * Math.pow(u - uRef, 2);
      for (let i = 0; i < STATE_DIM; i++) {
        const err = i === 2 || i === 4 ? wrapPi(s[i] - ref[i]) : s[i] - ref[i];
        c += 0.5 * Q[i] * Math.pow(err, 2);
      }
      return c;
    },

    term(s: State, t: number): number {
      const globalTime = tStart + t * dt;
      const ref = trajectory.getState(globalTime);

      let c = 0;
      for (let i = 0; i < STATE_DIM; i++) {
        const err = i === 2 || i === 4 ? wrapPi(s[i] - ref[i]) : s[i] - ref[i];
        c += 0.5 * (Q[i] * Q_terminal_scale) * Math.pow(err, 2);
      }
      return c;
    },

    runDeriv(s: State, u: number, t: number) {
      const globalTime = tStart + t * dt;
      const ref = trajectory.getState(globalTime);
      const uRef = trajectory.getControl(globalTime);

      for (let i = 0; i < STATE_DIM; i++) {
        const err = i === 2 || i === 4 ? wrapPi(s[i] - ref[i]) : s[i] - ref[i];
        lxRun[i] = Q[i] * err;
      }
      retRun.lu = R * (u - uRef);
      return retRun;
    },

    termDeriv(s: State, t: number) {
      const globalTime = tStart + t * dt;
      const ref = trajectory.getState(globalTime);

      for (let i = 0; i < STATE_DIM; i++) {
        const err = i === 2 || i === 4 ? wrapPi(s[i] - ref[i]) : s[i] - ref[i];
        lxTerm[i] = Q[i] * Q_terminal_scale * err;
      }
      return retTerm;
    },
  };
}

/**
 * Real-Time Iteration (RTI) Nonlinear Model Predictive Controller for Trajectory Tracking.
 */
export class TrackingMPC {
  public horizonSteps: number;
  public dt: number;
  public us: number[];
  public predictedXs: State[] = [];
  public trajectory: TrajectoryInterpolation;
  public plant: PlantParams;

  constructor(
    trajectory: TrajectoryInterpolation,
    horizonSeconds: number = 0.5,
    dt: number = 0.02,
    plant: PlantParams = DEFAULT_PLANT_PARAMS
  ) {
    this.trajectory = trajectory;
    this.dt = dt;
    this.horizonSteps = Math.max(2, Math.round(horizonSeconds / dt));
    this.us = new Array(this.horizonSteps).fill(0);
    this.plant = plant;
  }

  setHorizon(horizonSeconds: number) {
    const newSteps = Math.max(2, Math.round(horizonSeconds / this.dt));
    if (newSteps !== this.horizonSteps) {
      const newUs = new Array(newSteps).fill(0);
      for (let i = 0; i < Math.min(newSteps, this.horizonSteps); i++) {
        newUs[i] = this.us[i];
      }
      this.us = newUs;
      this.horizonSteps = newSteps;
    }
  }

  /**
   * Computes the optimal control input for the current timestep via RTI (1-iteration warm-started iLQR).
   */
  computeControl(s: State, t: number): number {
    // 1. Warm-start by shifting control buffer
    for (let i = 0; i < this.horizonSteps - 1; i++) {
      this.us[i] = this.us[i + 1];
    }
    this.us[this.horizonSteps - 1] = this.trajectory.getControl(t + this.horizonSteps * this.dt);

    // 2. Build tracking cost for current window
    const cost = createTrackingCost(this.trajectory, t, this.dt);

    // 3. Single Gauss-Newton iteration
    const result = ilqr(s, this.us, this.dt, cost, {
      maxIter: 1,
      verbose: false,
      plant: this.plant,
    });

    this.us = result.us;
    this.predictedXs = result.xs;

    return this.us[0];
  }
}
