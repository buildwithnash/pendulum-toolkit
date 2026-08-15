import { describe, it, expect } from 'vitest';
import {
  computeBalanceLQR,
  computeBrakeLQR,
  STATE_UPRIGHT,
  STATE_HANGING,
  State,
  rk4,
  evaluateLQR,
} from '../src/index.js';

describe('CARE Riccati & LQR Solvers', () => {
  it('should compute stabilizing balance gains for upright equilibrium', () => {
    const balance = computeBalanceLQR();
    expect(balance.K).toHaveLength(6);
    expect(balance.P).toHaveLength(6);

    // K gains must be non-zero
    for (const k of balance.K) {
      expect(Math.abs(k)).toBeGreaterThan(0.1);
    }

    // Closed-loop simulation from small tilt must converge towards origin
    let s: State = [0, 0, 0.05, 0, -0.03, 0];
    const dt = 0.005;
    for (let i = 0; i < 800; i++) {
      const u = evaluateLQR(s, balance.K, STATE_UPRIGHT);
      s = rk4(s, u, dt);
    }

    // After 4.0s of stabilization, cart and angles should be within tight boundary of upright
    expect(Math.abs(s[0])).toBeLessThan(0.05);
    expect(Math.abs(s[2])).toBeLessThan(0.01);
    expect(Math.abs(s[4])).toBeLessThan(0.01);
  });

  it('should compute stabilizing brake gains for hanging equilibrium', () => {
    const brake = computeBrakeLQR();
    expect(brake.K).toHaveLength(6);

    // Closed-loop simulation towards hanging
    let s: State = [0.5, 0, Math.PI - 0.2, 0, Math.PI + 0.1, 0];
    const dt = 0.005;
    for (let i = 0; i < 1000; i++) {
      const u = evaluateLQR(s, brake.K, STATE_HANGING);
      s = rk4(s, u, dt);
    }

    // After 5.0s, cart displacement and velocity should be near zero
    expect(Math.abs(s[0])).toBeLessThan(0.05);
    expect(Math.abs(s[1])).toBeLessThan(0.05);
  });
});
