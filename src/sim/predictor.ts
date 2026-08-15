import { State, PlantParams, DEFAULT_PLANT_PARAMS } from '../core/types.js';
import { rk4 } from '../core/dynamics.js';

export interface PredictorOptions {
  /** Estimated Coulomb stiction to compensate (N), default 0 */
  coulombEstimate?: number;
  /** Plant parameters */
  plant?: PlantParams;
}

/**
 * Rolls nominal plant dynamics forward across transport latency / loop delay tau
 * applying the recent control history.
 *
 * Crucial insight: On a rail with dry friction, integrating without friction compensation
 * causes the velocity estimates to bias into the velocity feedback gains, making prediction
 * worse than no prediction. `predictForward` incorporates friction estimation directly into
 * the forward integration steps.
 *
 * @param measuredState Delayed state estimate [x, v, th1, w1, th2, w2]
 * @param controlHistory Ring buffer of past applied controls [u_{-K}, ..., u_{-1}]
 * @param dt Timestep of control history entries (seconds)
 * @param delaySeconds Total loop delay tau
 * @param options Predictor options
 */
export function predictForward(
  measuredState: State,
  controlHistory: number[],
  dt: number,
  delaySeconds: number,
  options: PredictorOptions = {}
): State {
  const { coulombEstimate = 0, plant = DEFAULT_PLANT_PARAMS } = options;
  const steps = Math.round(delaySeconds / dt);
  if (steps <= 0) return [...measuredState] as State;

  let s = [...measuredState] as State;
  const historyLen = controlHistory.length;

  for (let i = 0; i < steps; i++) {
    // Read corresponding control from history buffer
    const idx = historyLen - steps + i;
    let u = idx >= 0 && idx < historyLen ? controlHistory[idx] : 0;

    // Apply Coulomb friction correction if moving
    if (coulombEstimate > 0) {
      const v = s[1];
      if (Math.abs(v) > 1e-3) {
        u -= coulombEstimate * Math.sign(v);
      }
    }

    s = rk4(s, u, dt, plant);
  }

  return s;
}
