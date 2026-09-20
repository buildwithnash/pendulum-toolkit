import { describe, it, expect } from 'vitest';
import {
  solveSwingUp,
  computeBalanceLQR,
  computeTVLQR,
  evaluateLQR,
  rk4,
  totalEnergy,
  DEFAULT_Q_BALANCE,
  DEFAULT_R_BALANCE,
  STATE_HANGING,
  STATE_UPRIGHT,
  State,
} from '../src/index.js';

const DT = 0.02;

const plan = solveSwingUp({ duration: 4.0, dt: DT });
const N = plan.us.length;
const balance = computeBalanceLQR();
const Ks = computeTVLQR(plan.xs, plan.us, DT, DEFAULT_Q_BALANCE, DEFAULT_R_BALANCE, balance.P);

/** Tracks the plan with TVLQR for 4 s, then balances with LQR for 4 s. */
function closedLoop(start: State) {
  const h = 0.002;
  let s: State = [...start] as State;
  for (let t = 0; t < 4.0 - 1e-9; t += h) {
    const i = Math.min(N - 1, Math.floor(t / DT + 1e-9));
    let u = plan.us[i];
    for (let j = 0; j < 6; j++) u -= Ks[i][j] * (s[j] - plan.xs[i][j]);
    s = rk4(s, u, h);
  }
  const atHandover = [...s] as State;
  for (let k = 0; k < 2000; k++) s = rk4(s, evaluateLQR(s, balance.K, STATE_UPRIGHT), h);
  return { atHandover, settled: s };
}

describe('Swing-up solve (iLQR with terminal-weight continuation)', () => {
  it('converges every stage and lands on the upright equilibrium', () => {
    expect(plan.stages).toHaveLength(4);
    for (const stage of plan.stages) expect(stage.converged).toBe(true);

    const final = plan.xs[N];
    expect(Math.abs(final[2])).toBeLessThan(0.01);
    expect(Math.abs(final[4])).toBeLessThan(0.01);
    expect(Math.abs(final[3])).toBeLessThan(0.05);
    expect(Math.abs(final[5])).toBeLessThan(0.05);
    expect(Math.abs(totalEnergy(final) - 19.62)).toBeLessThan(0.05);
    expect(Math.abs(totalEnergy(plan.xs[0]) + 19.62)).toBeLessThan(0.01);
  });

  it('stays inside the actuator and rail limits', () => {
    const peakU = Math.max(...plan.us.map(Math.abs));
    const peakX = Math.max(...plan.xs.map((x) => Math.abs(x[0])));
    expect(peakU).toBeLessThan(60);
    expect(peakX).toBeLessThan(2.5);
  });

  it('starts from hanging', () => {
    for (let i = 0; i < 6; i++) expect(plan.xs[0][i]).toBeCloseTo(STATE_HANGING[i], 12);
  });

  it('reproduces the trajectory published on davidnash.dev with finite-difference linearization', () => {
    // The published trajectory (cost 1.896, 30.3 N peak, 0.92 m cart travel) used finite differences.
    const published = solveSwingUp({ linearization: 'finite-difference' });
    expect(published.cost).toBeCloseTo(1.896, 2);
    expect(Math.max(...published.us.map(Math.abs))).toBeCloseTo(30.3, 0);
    expect(Math.max(...published.xs.map((x) => Math.abs(x[0])))).toBeCloseTo(0.92, 2);
  });

  it('gives the same answer under both linearizations after the first iteration', () => {
    // Later iterations can diverge into a different local optimum; a single step must agree.
    const a = solveSwingUp({ stages: [[20, 20, 200, 20, 200, 20]], maxIterPerStage: 1 });
    const f = solveSwingUp({
      stages: [[20, 20, 200, 20, 200, 20]],
      maxIterPerStage: 1,
      linearization: 'finite-difference',
    });
    expect(Math.abs(a.cost - f.cost) / f.cost).toBeLessThan(1e-7);
    const worstDu = Math.max(...a.us.map((u, i) => Math.abs(u - f.us[i])));
    expect(worstDu).toBeLessThan(1e-4);
  });
});

describe('TVLQR handover', () => {
  it('ends on gains close to the balance controller', () => {
    // Same Q and R and a P-seeded backward pass, so the gains differ only by discretization.
    const last = Ks[N - 1];
    last.forEach((k, i) => {
      expect(Math.abs((k - balance.K[i]) / balance.K[i])).toBeLessThan(0.15);
      expect(Math.sign(k)).toBe(Math.sign(balance.K[i]));
    });
  });

  it('stands the pendulum up in closed loop, then holds it', () => {
    const { atHandover, settled } = closedLoop(STATE_HANGING);
    expect(Math.abs(atHandover[2])).toBeLessThan(0.03);
    expect(Math.abs(atHandover[4])).toBeLessThan(0.03);
    expect(Math.abs(settled[0])).toBeLessThan(0.01);
    expect(Math.abs(settled[2])).toBeLessThan(0.001);
    expect(Math.abs(settled[4])).toBeLessThan(0.001);
  });

  it('still reaches upright from slightly perturbed hanging poses', () => {
    const perturbed: State[] = [
      [0, 0, Math.PI + 0.05, 0, Math.PI, 0],
      [0, 0, Math.PI, 0, Math.PI - 0.07, 0],
    ];
    for (const start of perturbed) {
      const { settled } = closedLoop(start);
      expect(Math.abs(settled[2])).toBeLessThan(0.001);
      expect(Math.abs(settled[4])).toBeLessThan(0.001);
    }
  });
});
