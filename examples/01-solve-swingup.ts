import {
  solveSwingUp,
  computeTVLQR,
  computeBalanceLQR,
  evaluateLQR,
  rk4,
  totalEnergy,
  DEFAULT_Q_BALANCE,
  DEFAULT_R_BALANCE,
  STATE_UPRIGHT,
  State,
} from '../src/index.js';

console.log('=== Double Pendulum Swing-Up: iLQR + TVLQR -> LQR ===\n');

const DT = 0.02; // 20 ms knot spacing
const DURATION = 4.0; // seconds

// 1. Plan the swing-up offline in four stages that tighten the terminal penalty (DEFAULT_QF_STAGES).
console.log(`Solving swing-up: ${Math.round(DURATION / DT)} knots (${DURATION}s @ ${DT * 1000}ms)`);
const t0 = performance.now();
const plan = solveSwingUp({ duration: DURATION, dt: DT });
const solveMs = performance.now() - t0;

const N = plan.us.length;
const final = plan.xs[N];
console.log(`Solve completed in ${solveMs.toFixed(0)} ms (${plan.totalIters} iterations)`);
plan.stages.forEach((s) =>
  console.log(`  stage ${s.stage}: ${s.iters} iterations, converged: ${s.converged}`)
);
console.log(`Final cost: ${plan.cost.toFixed(3)}`);
console.log(
  `Energy: ${totalEnergy(plan.xs[0]).toFixed(2)} J -> ${totalEnergy(final).toFixed(2)} J (target +19.62 J)`
);
console.log(
  `Final angles: th1 = ${final[2].toFixed(4)} rad, th2 = ${final[4].toFixed(4)} rad (upright = 0)`
);
console.log(`Peak cart force: ${Math.max(...plan.us.map(Math.abs)).toFixed(1)} N`);
console.log(`Peak cart excursion: ${Math.max(...plan.xs.map((x) => Math.abs(x[0]))).toFixed(2)} m`);

// 2. Tracking gains K(t) along the plan. Seeding the backward pass with the balance controller's
//    own P, and using the same Q and R, makes the last gains land on the balance gains.
console.log('\nComputing TVLQR gain schedule K(t)...');
const balance = computeBalanceLQR();
const Ks = computeTVLQR(plan.xs, plan.us, DT, DEFAULT_Q_BALANCE, DEFAULT_R_BALANCE, balance.P);
const lastK = Ks[N - 1];
const fmt = (v: number[]) => v.map((x) => x.toFixed(2)).join(', ');
console.log(`  Final TVLQR gain: [${fmt(lastK)}]`);
console.log(`  Balance LQR gain: [${fmt(balance.K)}]`);
console.log(
  `  Difference:       [${lastK.map((v, i) => (((v - balance.K[i]) / balance.K[i]) * 100).toFixed(1) + '%').join(', ')}]`
);

// 3. Closed loop: track the plan with TVLQR for 4 s, then hand over to the balance LQR.
console.log('\nClosed-loop check (start from hanging, physics at 500 Hz):');
const h = 0.002;
let s: State = [...plan.xs[0]] as State;
for (let t = 0; t < DURATION - 1e-9; t += h) {
  const i = Math.min(N - 1, Math.floor(t / DT + 1e-9));
  let u = plan.us[i];
  for (let j = 0; j < 6; j++) u -= Ks[i][j] * (s[j] - plan.xs[i][j]);
  s = rk4(s, u, h);
}
console.log(`  t = 4.0 s (handover): th1 = ${s[2].toFixed(4)}, th2 = ${s[4].toFixed(4)} rad`);
for (let k = 0; k < Math.round(2.0 / h); k++) {
  s = rk4(s, evaluateLQR(s, balance.K, STATE_UPRIGHT), h);
}
console.log(
  `  t = 6.0 s (balancing): x = ${s[0].toFixed(4)} m, th1 = ${s[2].toFixed(5)}, th2 = ${s[4].toFixed(5)} rad`
);
