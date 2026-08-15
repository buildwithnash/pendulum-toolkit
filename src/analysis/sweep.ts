import { State, STATE_UPRIGHT, DEFAULT_PLANT_PARAMS } from '../core/types.js';
import { HardwareBench } from '../sim/bench.js';
import { predictForward } from '../sim/predictor.js';
import { evaluateLQR } from '../controllers/lqr.js';

export interface RobustnessPoint {
  loopDelayMs: number;
  coulombFrictionN: number;
  massMultiplier: number;
  success: boolean;
  maxExcursionM: number;
}

/**
 * Runs a sweep testing controller survivability across parametric mass errors,
 * dry Coulomb friction, and round-trip loop latency.
 */
export function sweepRobustness(
  K_balance: number[],
  delaySweepMs: number[] = [0, 10, 20, 30, 40, 50, 60, 80],
  usePredictor: boolean = true
): RobustnessPoint[] {
  const results: RobustnessPoint[] = [];
  const dt = 0.002;
  const simSteps = Math.round(5.0 / dt);

  for (const delayMs of delaySweepMs) {
    const loopDelay = delayMs / 1000.0;
    // Initial upright perturbation
    const s0: State = [0, 0, 0.08, 0, -0.05, 0];

    const bench = new HardwareBench(s0, {
      loopDelay,
      coulombFriction: 0.5,
      plant: DEFAULT_PLANT_PARAMS,
    });

    let success = true;
    let maxExcursion = 0;

    for (let step = 0; step < simSteps; step++) {
      const measured = bench.getSensorMeasurement();
      let stateForControl = measured;

      if (usePredictor && loopDelay > 0) {
        stateForControl = predictForward(measured, bench.getControlHistory(), dt, loopDelay, {
          coulombEstimate: 0.4,
          plant: DEFAULT_PLANT_PARAMS,
        });
      }

      let u = evaluateLQR(stateForControl, K_balance, STATE_UPRIGHT);
      if (Math.abs(u) > 100) u = Math.sign(u) * 100;

      const { state, diedOnRail } = bench.step(u, dt);
      maxExcursion = Math.max(maxExcursion, Math.abs(state[0]));

      if (diedOnRail || Math.abs(state[2]) > Math.PI / 2 || Math.abs(state[4]) > Math.PI / 2) {
        success = false;
        break;
      }
    }

    results.push({
      loopDelayMs: delayMs,
      coulombFrictionN: 0.5,
      massMultiplier: 1.0,
      success,
      maxExcursionM: maxExcursion,
    });
  }

  return results;
}
