import { State, STATE_UPRIGHT, PlantParams, DEFAULT_PLANT_PARAMS } from '../core/types.js';
import { rk4 } from '../core/dynamics.js';
import { evaluateLQR } from '../controllers/lqr.js';

export interface BasinSweepOptions {
  /** Resolution per axis, default 41 */
  resolution?: number;
  /** Max angle range in radians (e.g. 0.6 = ~35 degrees), default 0.6 */
  maxAngle?: number;
  /** Simulation duration per point (seconds), default 3.0 */
  simDuration?: number;
  /** Simulation timestep (seconds), default 0.005 */
  dt?: number;
  /** Maximum cart force limit (N), default 150 */
  uMax?: number;
  /** Plant parameters */
  plant?: PlantParams;
}

export interface BasinPoint {
  theta1: number;
  theta2: number;
  stable: boolean;
  settleTime: number;
  maxCartDisplacement: number;
}

/**
 * Sweeps a 2D grid of initial link tilt angles (θ1, θ2) starting from rest
 * to evaluate closed-loop region-of-attraction (basin of attraction) under LQR.
 */
export function sweepBasin(K_balance: number[], options: BasinSweepOptions = {}): BasinPoint[] {
  const {
    resolution = 31,
    maxAngle = 0.55,
    simDuration = 3.0,
    dt = 0.005,
    uMax = 150.0,
    plant = DEFAULT_PLANT_PARAMS,
  } = options;

  const results: BasinPoint[] = [];
  const steps = Math.round(simDuration / dt);

  const angles: number[] = [];
  for (let i = 0; i < resolution; i++) {
    angles.push(-maxAngle + (2 * maxAngle * i) / (resolution - 1));
  }

  for (const th1 of angles) {
    for (const th2 of angles) {
      let s: State = [0, 0, th1, 0, th2, 0];
      let stable = true;
      let settleTime = simDuration;
      let maxCart = 0;

      for (let step = 0; step < steps; step++) {
        let u = evaluateLQR(s, K_balance, STATE_UPRIGHT);
        if (Math.abs(u) > uMax) {
          u = Math.sign(u) * uMax;
        }

        s = rk4(s, u, dt, plant);
        maxCart = Math.max(maxCart, Math.abs(s[0]));

        // Check if divergent (|th1| > pi/2 or |th2| > pi/2 or cart runaway)
        if (Math.abs(s[2]) > Math.PI / 2 || Math.abs(s[4]) > Math.PI / 2 || Math.abs(s[0]) > 5.0) {
          stable = false;
          settleTime = step * dt;
          break;
        }
      }

      // Check final state error
      if (stable) {
        const finalErr = Math.abs(s[0]) + Math.abs(s[2]) + Math.abs(s[4]);
        if (finalErr > 0.1) {
          stable = false;
        }
      }

      results.push({
        theta1: th1,
        theta2: th2,
        stable,
        settleTime,
        maxCartDisplacement: maxCart,
      });
    }
  }

  return results;
}
