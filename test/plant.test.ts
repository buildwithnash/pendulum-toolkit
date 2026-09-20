import { describe, it, expect } from 'vitest';
import {
  dynamics,
  dynamicsInto,
  rk4,
  rk4Into,
  totalEnergy,
  getPrecomputed,
  precomputeDynamics,
  DEFAULT_PLANT_PARAMS,
  PlantParams,
  State,
} from '../src/index.js';

/** Non-default masses and lengths, no damping. */
const undampedCustom: PlantParams = {
  ...DEFAULT_PLANT_PARAMS,
  M: 2.0,
  m1: 1.5,
  m2: 0.7,
  L1: 0.8,
  L2: 1.1,
  jointDamp: 0,
  cartDamp: 0,
};

describe('Custom plant parameters', () => {
  it('conserves energy for a non-default plant (masses and lengths reach the dynamics)', () => {
    let s: State = [0, 0, 0.5, 0, -0.3, 0];
    const e0 = totalEnergy(s, undampedCustom);
    for (let i = 0; i < 1000; i++) s = rk4(s, 0, 0.001, undampedCustom);
    expect(Math.abs(totalEnergy(s, undampedCustom) - e0)).toBeLessThan(1e-3);
  });

  it('gives different accelerations for a heavier plant', () => {
    const s: State = [0, 0, 0.5, 0, -0.3, 0];
    const a = dynamics(s, 5, DEFAULT_PLANT_PARAMS);
    const b = dynamics(s, 5, undampedCustom);
    expect(Math.abs(a[1] - b[1])).toBeGreaterThan(0.1);
  });

  it('derives precomputed terms from the plant when they are not supplied', () => {
    const s: State = [0.1, 0.2, 0.5, 0.3, -0.3, -0.2];
    const implicit = dynamics(s, 3, undampedCustom);
    const explicit = dynamics(s, 3, undampedCustom, precomputeDynamics(undampedCustom));
    for (let i = 0; i < 6; i++) expect(implicit[i]).toBe(explicit[i]);
  });

  it('caches precomputed terms per plant object', () => {
    expect(getPrecomputed(undampedCustom)).toBe(getPrecomputed(undampedCustom));
    expect(getPrecomputed(DEFAULT_PLANT_PARAMS)).toBe(getPrecomputed(DEFAULT_PLANT_PARAMS));
    expect(getPrecomputed(undampedCustom)).not.toBe(getPrecomputed(DEFAULT_PLANT_PARAMS));
  });
});

describe('Allocation-free physics', () => {
  it('rk4Into matches rk4 exactly', () => {
    const s: State = [0.2, -0.4, 1.1, 0.7, -2.2, 1.3];
    const expected = rk4(s, 17, 0.01);
    const out = new Float64Array(6);
    rk4Into(out, s, 17, 0.01);
    for (let i = 0; i < 6; i++) expect(out[i]).toBe(expected[i]);
  });

  it('rk4Into can step in place', () => {
    const s: State = [0.2, -0.4, 1.1, 0.7, -2.2, 1.3];
    const expected = rk4(s, 17, 0.01);
    const inPlace = [...s];
    rk4Into(inPlace, inPlace, 17, 0.01);
    for (let i = 0; i < 6; i++) expect(inPlace[i]).toBe(expected[i]);
  });

  it('dynamicsInto matches dynamics exactly', () => {
    const s: State = [0.2, -0.4, 1.1, 0.7, -2.2, 1.3];
    const expected = dynamics(s, -8);
    const out = new Float64Array(6);
    dynamicsInto(out, s, -8);
    for (let i = 0; i < 6; i++) expect(out[i]).toBe(expected[i]);
  });
});
