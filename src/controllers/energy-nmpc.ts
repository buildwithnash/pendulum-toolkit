import {
  State,
  STATE_DIM,
  CostFunction,
  PlantParams,
  DEFAULT_PLANT_PARAMS,
} from '../core/types.js';
import { ilqr } from '../solvers/ilqr.js';
import { totalEnergy } from '../core/dynamics.js';
import { wrapPi } from '../core/math.js';
import { computeBalanceLQR } from './lqr.js';

export interface EnergyNmpcOptions {
  R?: number;
  Q_cart?: number;
  Q_w?: number;
  W_energy?: number;
  W_bowl?: number;
  W_lqr?: number;
  gateAngle?: number;
  gateRate?: number;
  plant?: PlantParams;
}

/**
 * Creates an energy-based cost function for reference-free swing-up NMPC.
 *
 * It uses:
 * 1. Running cost on cart center, angular rate damping, and control effort.
 * 2. Energy terminal cost driving system energy to the upright target (+19.62 J).
 * 3. Periodic catch bowl term: (1 - cos θ1) + (1 - cos θ2) for smooth landing.
 * 4. Gated LQR cost-to-go term: 0.5 * x^T * P_balance * x, activated only near upright.
 */
export function createEnergyCost(
  p: PlantParams = DEFAULT_PLANT_PARAMS,
  P_balance?: number[][],
  options: EnergyNmpcOptions = {}
): CostFunction {
  const {
    R = 0.08,
    Q_cart = 5.0,
    Q_w = 0.3,
    W_energy = 150.0,
    W_bowl = 1500.0,
    W_lqr = 1.0,
    gateAngle = 0.08,
    gateRate = 2.0,
  } = options;

  const P_mat = P_balance ?? computeBalanceLQR(undefined, undefined, p).P;
  const targetEnergy = totalEnergy([0, 0, 0, 0, 0, 0], p); // +19.62 J for nominal plant

  const lxRun = new Float64Array(STATE_DIM);
  const lxxRun = new Float64Array(STATE_DIM);
  lxxRun[0] = Q_cart;
  lxxRun[3] = Q_w;
  lxxRun[5] = Q_w;

  const lxTerm = new Float64Array(STATE_DIM);
  const lxxTerm = new Float64Array(STATE_DIM);

  const retRun = { lx: lxRun, lxxDiag: lxxRun, lu: 0, luu: R };
  const retTerm = { lx: lxTerm, lxxDiag: lxxTerm };

  const sp: State = [0, 0, 0, 0, 0, 0];
  const sm: State = [0, 0, 0, 0, 0, 0];
  const err = new Float64Array(STATE_DIM);

  const terminalCost = (state: State): number => {
    // Energy error
    const e = totalEnergy(state, p);
    let c = 0.5 * W_energy * Math.pow(e - targetEnergy, 2);

    // Catch bowl: smooth periodic trigonometric form (1 - cos θ)
    c += W_bowl * (1 - Math.cos(state[2])) + W_bowl * (1 - Math.cos(state[4]));
    c += 0.5 * 20.0 * (state[0] * state[0]);
    c += 0.5 * (Q_w * 2) * (state[3] * state[3] + state[5] * state[5]);

    // Gated LQR cost-to-go
    const angErr = 1 - Math.cos(state[2]) + (1 - Math.cos(state[4]));
    const rateErr = state[3] * state[3] + state[5] * state[5];
    const gate = Math.exp(-angErr / gateAngle - rateErr / gateRate);

    if (gate > 1e-6) {
      err[0] = state[0];
      err[1] = state[1];
      err[2] = wrapPi(state[2]);
      err[3] = state[3];
      err[4] = wrapPi(state[4]);
      err[5] = state[5];

      let q = 0;
      for (let i = 0; i < STATE_DIM; i++) {
        const ei = err[i];
        if (ei === 0) continue;
        for (let j = 0; j < STATE_DIM; j++) {
          q += ei * P_mat[i][j] * err[j];
        }
      }
      c += W_lqr * gate * 0.5 * q;
    }

    return c;
  };

  return {
    run(s: State, u: number, _t: number): number {
      return 0.5 * Q_cart * s[0] * s[0] + 0.5 * Q_w * (s[3] * s[3] + s[5] * s[5]) + 0.5 * R * u * u;
    },

    term(s: State, _t: number): number {
      return terminalCost(s);
    },

    runDeriv(s: State, u: number, _t: number) {
      lxRun[0] = Q_cart * s[0];
      lxRun[3] = Q_w * s[3];
      lxRun[5] = Q_w * s[5];
      retRun.lu = R * u;
      return retRun;
    },

    termDeriv(s: State, _t: number) {
      const eps = 1e-5;
      const baseCost = terminalCost(s);

      for (let i = 0; i < STATE_DIM; i++) {
        for (let k = 0; k < STATE_DIM; k++) {
          sp[k] = s[k];
          sm[k] = s[k];
        }
        sp[i] += eps;
        sm[i] -= eps;

        const cp = terminalCost(sp);
        const cm = terminalCost(sm);

        lxTerm[i] = (cp - cm) / (2 * eps);
        lxxTerm[i] = (cp - 2 * baseCost + cm) / (eps * eps);
      }

      return retTerm;
    },
  };
}

/**
 * From-Scratch Energy NMPC.
 * Discovers the swing-up trajectory online in real-time without reference trajectories.
 */
export class EnergyNMPC {
  public horizonSteps: number;
  public dt: number;
  public us: number[];
  public predictedXs: State[] = [];
  public cost: CostFunction;
  public plant: PlantParams;
  public options: EnergyNmpcOptions;
  private lastT: number = -1;
  private shiftBuffer: number[] = [];

  constructor(
    horizonSeconds: number = 1.0,
    dt: number = 0.02,
    plant: PlantParams = DEFAULT_PLANT_PARAMS,
    options: EnergyNmpcOptions = {}
  ) {
    this.plant = plant;
    this.dt = dt;
    this.options = options;
    this.horizonSteps = Math.max(2, Math.round(horizonSeconds / dt));
    this.us = new Array(this.horizonSteps).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    this.shiftBuffer = new Array(this.horizonSteps).fill(0);
    this.cost = createEnergyCost(this.plant, undefined, options);
  }

  reset(): void {
    this.lastT = -1;
    this.us = new Array(this.horizonSteps).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    this.predictedXs = [];
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
      this.shiftBuffer = new Array(this.horizonSteps).fill(0);
      this.cost = createEnergyCost(this.plant, undefined, this.options);
      this.lastT = -1;
    }
  }

  /**
   * Computes control command by optimizing receding horizon.
   *
   * @param s - Current system state [x, v, θ1, ω1, θ2, ω2]
   * @param maxIter - Maximum iLQR optimization iterations (default 20)
   * @param tol - Convergence cost change tolerance (default 1e-3)
   * @param t - Optional simulation timestamp. When provided, warm-starting continuously interpolates
   *            control actions based on actual elapsed time (t - lastT), keeping the horizon synchronized
   *            at high control loop frequencies (e.g. 100 Hz, 200 Hz) without discarding future steps.
   */
  computeControl(s: State, maxIter: number = 20, tol: number = 1e-3, t?: number): number {
    if (t !== undefined) {
      if (this.lastT < 0 || t < this.lastT) {
        this.lastT = t;
      }
      const dtElapsed = t - this.lastT;
      this.lastT = t;

      const shiftKnots = dtElapsed / this.dt;
      if (shiftKnots > 0) {
        if (this.shiftBuffer.length !== this.horizonSteps) {
          this.shiftBuffer = new Array(this.horizonSteps).fill(0);
        }
        for (let i = 0; i < this.horizonSteps; i++) {
          const srcIdx = i + shiftKnots;
          const idx0 = Math.floor(srcIdx);
          const frac = srcIdx - idx0;
          const u0 = idx0 < this.horizonSteps ? this.us[idx0] : 0;
          const u1 = idx0 + 1 < this.horizonSteps ? this.us[idx0 + 1] : 0;
          this.shiftBuffer[i] = u0 * (1 - frac) + u1 * frac;
        }
        for (let i = 0; i < this.horizonSteps; i++) {
          this.us[i] = this.shiftBuffer[i];
        }
      }
    } else {
      // Warm-start: shift buffer left by 1 knot (assumes control interval == knot dt)
      for (let i = 0; i < this.horizonSteps - 1; i++) {
        this.us[i] = this.us[i + 1];
      }
      this.us[this.horizonSteps - 1] = 0;
    }

    const result = ilqr(s, this.us, this.dt, this.cost, {
      maxIter,
      tol,
      verbose: false,
      plant: this.plant,
    });

    this.us = result.us;
    this.predictedXs = result.xs;
    return this.us[0];
  }
}
