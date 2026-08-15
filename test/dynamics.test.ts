import { describe, it, expect } from 'vitest';
import {
  dynamics,
  rk4,
  totalEnergy,
  STATE_UPRIGHT,
  STATE_HANGING,
  DEFAULT_PLANT_PARAMS,
  wrapPi,
  State,
} from '../src/index.js';

describe('Double Pendulum Dynamics & Math', () => {
  it('should have zero acceleration at upright equilibrium with zero force', () => {
    const dxdt = dynamics(STATE_UPRIGHT, 0);
    for (let i = 0; i < 6; i++) {
      expect(Math.abs(dxdt[i])).toBeLessThan(1e-10);
    }
  });

  it('should have zero acceleration at hanging equilibrium with zero force', () => {
    const dxdt = dynamics(STATE_HANGING, 0);
    for (let i = 0; i < 6; i++) {
      expect(Math.abs(dxdt[i])).toBeLessThan(1e-10);
    }
  });

  it('should compute nominal upright potential energy as +19.62 J', () => {
    const E = totalEnergy(STATE_UPRIGHT, DEFAULT_PLANT_PARAMS);
    expect(E).toBeCloseTo(19.62, 2);
  });

  it('should compute nominal hanging potential energy as -19.62 J', () => {
    const E = totalEnergy(STATE_HANGING, DEFAULT_PLANT_PARAMS);
    expect(E).toBeCloseTo(-19.62, 2);
  });

  it('should conserve mechanical energy in unforced undamped motion', () => {
    const undampedPlant = {
      ...DEFAULT_PLANT_PARAMS,
      jointDamp: 0,
      cartDamp: 0,
    };

    let s: State = [0, 0, 0.5, 0, -0.3, 0];
    const initialEnergy = totalEnergy(s, undampedPlant);
    const dt = 0.001;

    for (let i = 0; i < 1000; i++) {
      s = rk4(s, 0, dt, undampedPlant);
    }

    const finalEnergy = totalEnergy(s, undampedPlant);
    expect(Math.abs(finalEnergy - initialEnergy)).toBeLessThan(1e-3);
  });

  it('should properly wrap angles across branch cuts', () => {
    expect(wrapPi(Math.PI + 0.1)).toBeCloseTo(-Math.PI + 0.1, 5);
    expect(wrapPi(-Math.PI - 0.1)).toBeCloseTo(Math.PI - 0.1, 5);
    expect(wrapPi(4 * Math.PI)).toBeCloseTo(0, 5);
  });
});
