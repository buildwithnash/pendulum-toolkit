import { State, PlantParams, DEFAULT_PLANT_PARAMS } from '../core/types.js';
import { rk4 } from '../core/dynamics.js';

export interface HardwareBenchConfig {
  /** Plant parameters (allows simulating model mismatch) */
  plant?: PlantParams;
  /** Transport / computation loop delay in seconds (e.g. 0.04 for 40ms) */
  loopDelay?: number;
  /** Coulomb stiction force on cart rail (N), default 0 */
  coulombFriction?: number;
  /** Encoder bits for angle measurement quantization (e.g. 12 = 4096 counts/rev, 0 = ideal) */
  encoderBits?: number;
  /** Force disturbance noise standard deviation (N), default 0 */
  forceNoiseStd?: number;
  /** Cart rail travel limit (m), default 6.0 */
  railLimit?: number;
}

/**
 * Realistic hardware bench simulation engine.
 */
export class HardwareBench {
  public state: State;
  public config: Required<HardwareBenchConfig>;
  private controlHistory: number[] = [];
  private historyCapacity: number;
  public time: number = 0;

  constructor(initialState: State, config: HardwareBenchConfig = {}) {
    this.state = [...initialState] as State;
    this.config = {
      plant: config.plant ?? DEFAULT_PLANT_PARAMS,
      loopDelay: config.loopDelay ?? 0,
      coulombFriction: config.coulombFriction ?? 0,
      encoderBits: config.encoderBits ?? 0,
      forceNoiseStd: config.forceNoiseStd ?? 0,
      railLimit: config.railLimit ?? 6.0,
    };

    // Allocate control history for delay simulation
    this.historyCapacity = Math.max(10, Math.ceil((this.config.loopDelay + 0.1) / 0.001));
    this.controlHistory = new Array(this.historyCapacity).fill(0);
  }

  /**
   * Generates noisy/quantized sensor measurements from the true physical state.
   */
  getSensorMeasurement(): State {
    const s = [...this.state] as State;
    const { encoderBits } = this.config;

    if (encoderBits > 0) {
      const counts = Math.pow(2, encoderBits);
      const quant = (2 * Math.PI) / counts;
      s[2] = Math.round(s[2] / quant) * quant;
      s[4] = Math.round(s[4] / quant) * quant;
    }

    return s;
  }

  /**
   * Pushes a commanded control input into the transport pipeline and executes one physics step.
   *
   * @param commandedForce Commanded force u_cmd from controller (N)
   * @param dt Physics timestep duration (e.g. 0.001 for 1 kHz simulation)
   */
  step(
    commandedForce: number,
    dt: number
  ): { state: State; appliedForce: number; diedOnRail: boolean } {
    this.controlHistory.push(commandedForce);
    if (this.controlHistory.length > this.historyCapacity) {
      this.controlHistory.shift();
    }

    // Delayed force lookup
    const delaySteps = Math.round(this.config.loopDelay / dt);
    const histIdx = this.controlHistory.length - 1 - delaySteps;
    let appliedForce = histIdx >= 0 ? this.controlHistory[histIdx] : 0;

    // Add noise disturbance if configured
    if (this.config.forceNoiseStd > 0) {
      const u1 = Math.random();
      const u2 = Math.random();
      const randNorm = Math.sqrt(-2.0 * Math.log(u1 || 1e-9)) * Math.cos(2.0 * Math.PI * u2);
      appliedForce += randNorm * this.config.forceNoiseStd;
    }

    // Apply Coulomb dry friction opposing cart motion: -F * sign(v)
    let netForce = appliedForce;
    if (this.config.coulombFriction > 0) {
      const v = this.state[1];
      if (Math.abs(v) > 1e-4) {
        netForce -= this.config.coulombFriction * Math.sign(v);
      }
    }

    // Integrate physics
    this.state = rk4(this.state, netForce, dt, this.config.plant);
    this.time += dt;

    const diedOnRail = Math.abs(this.state[0]) > this.config.railLimit;

    return {
      state: [...this.state] as State,
      appliedForce,
      diedOnRail,
    };
  }

  /** Returns recent control history for state predictors */
  getControlHistory(): number[] {
    return this.controlHistory.slice();
  }
}
