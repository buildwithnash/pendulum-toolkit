import { describe, it, expect } from 'vitest';
import { ilqr, CostFunction, State, STATE_DIM, rollout, totalCost } from '../src/index.js';

describe('iLQR Trajectory Optimizer', () => {
  it('should monotonically decrease cost and achieve terminal target using smooth trigonometric potential', () => {
    const DT = 0.02;
    const N = 100; // 2.0s test window

    // Smooth cost function using (1 - cos θ) for angles (θ1=s[2], θ2=s[4])
    const cost: CostFunction = {
      run(s: State, u: number) {
        return (
          0.5 * 0.1 * u * u +
          0.5 * s[0] * s[0] +
          10.0 * (1 - Math.cos(s[2])) +
          10.0 * (1 - Math.cos(s[4]))
        );
      },
      term(s: State) {
        return 100 * s[0] * s[0] + 500 * (1 - Math.cos(s[2])) + 500 * (1 - Math.cos(s[4]));
      },
      runDeriv(s: State, u: number) {
        const lx = new Float64Array(STATE_DIM);
        const lxx = new Float64Array(STATE_DIM);
        lx[0] = s[0];
        lxx[0] = 1;

        // Angle derivatives: d/dθ [Q*(1 - cos θ)] = Q*sin(θ)
        lx[2] = 10.0 * Math.sin(s[2]);
        lxx[2] = Math.max(0, 10.0 * Math.cos(s[2]));
        lx[4] = 10.0 * Math.sin(s[4]);
        lxx[4] = Math.max(0, 10.0 * Math.cos(s[4]));

        return { lx, lxxDiag: lxx, lu: 0.1 * u, luu: 0.1 };
      },
      termDeriv(s: State) {
        const lx = new Float64Array(STATE_DIM);
        const lxx = new Float64Array(STATE_DIM);
        lx[0] = 200 * s[0];
        lxx[0] = 200;

        lx[2] = 500 * Math.sin(s[2]);
        lxx[2] = Math.max(0, 500 * Math.cos(s[2]));
        lx[4] = 500 * Math.sin(s[4]);
        lxx[4] = Math.max(0, 500 * Math.cos(s[4]));

        return { lx, lxxDiag: lxx };
      },
    };

    const s0: State = [0, 0, 0.4, 0, 0.3, 0];
    const usInit = new Array(N).fill(0);

    const xsInit = rollout(s0, usInit, DT);
    const initialCost = totalCost(xsInit, usInit, cost);

    const result = ilqr(s0, usInit, DT, cost, { maxIter: 40, tol: 1e-4 });

    expect(result.iters).toBeGreaterThan(0);
    expect(result.cost).toBeLessThan(initialCost);
    expect(result.xs).toHaveLength(N + 1);
    expect(result.us).toHaveLength(N);
    expect(result.Ks).toHaveLength(N);

    // Final state error should be smaller than initial uncontrolled fall
    const finalErr =
      Math.abs(result.xs[N][0]) + (1 - Math.cos(result.xs[N][2])) + (1 - Math.cos(result.xs[N][4]));
    const unforcedErr =
      Math.abs(xsInit[N][0]) + (1 - Math.cos(xsInit[N][2])) + (1 - Math.cos(xsInit[N][4]));

    expect(finalErr).toBeLessThan(unforcedErr);
  });
});
