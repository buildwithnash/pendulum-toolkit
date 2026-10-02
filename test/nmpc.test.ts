import { describe, it, expect } from 'vitest';
import { EnergyNMPC, TrackingMPC, TrajectoryInterpolation, State } from '../src/index.js';

describe('Real-Time Nonlinear MPC', () => {
  it('should compute control step fast enough for real-time loops', () => {
    const mockTrajectory: TrajectoryInterpolation = {
      getState: (_t: number): State => [0, 0, 0, 0, 0, 0],
      getControl: (_t: number): number => 0,
    };

    const trackingMPC = new TrackingMPC(mockTrajectory, 0.5, 0.02);
    const s: State = [0, 0, 0.02, 0, -0.01, 0];

    // Warmup call to allow V8 JIT compilation
    trackingMPC.computeControl(s, 0);

    const t0 = performance.now();
    const u = trackingMPC.computeControl(s, 0.01);
    const elapsed = performance.now() - t0;

    expect(typeof u).toBe('number');
    expect(Number.isFinite(u)).toBe(true);
    // Real-time tracking threshold accounting for noisy cloud CI runners
    expect(elapsed).toBeLessThan(40);
  });

  it('should compute valid control force from EnergyNMPC', () => {
    const nmpc = new EnergyNMPC(0.6, 0.02);
    const s: State = [0, 0, 0.1, 0, 0.1, 0];

    const u = nmpc.computeControl(s, 5, 1e-2);
    expect(Number.isFinite(u)).toBe(true);
    expect(nmpc.predictedXs.length).toBeGreaterThan(0);
  });

  it('should support time-interpolated continuous warm starting and reset', () => {
    const nmpc = new EnergyNMPC(0.8, 0.02);
    const s: State = [0, 0, Math.PI, 0, Math.PI, 0];

    // High-frequency control calls at 200 Hz (5ms step)
    const u0 = nmpc.computeControl(s, 5, 1e-2, 0.0);
    const u1 = nmpc.computeControl(s, 5, 1e-2, 0.005);
    const u2 = nmpc.computeControl(s, 5, 1e-2, 0.01);

    expect(Number.isFinite(u0)).toBe(true);
    expect(Number.isFinite(u1)).toBe(true);
    expect(Number.isFinite(u2)).toBe(true);

    nmpc.reset();
    expect(nmpc.predictedXs.length).toBe(0);
  });
});
