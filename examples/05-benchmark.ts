import os from 'node:os';
import {
  rk4,
  computeBalanceLQR,
  linearizeDiscrete,
  linearizeDiscreteInto,
  linearizeDiscreteFD,
  solveSwingUp,
  EnergyNMPC,
  TrackingMPC,
  TrajectoryInterpolation,
  State,
  DEFAULT_PLANT_PARAMS,
} from '../src/index.js';

/** Mean milliseconds per call of `fn` after `warmup` untimed calls (V8 needs them to optimize). */
function bench(fn: () => void, iters: number, warmup: number): number {
  for (let i = 0; i < warmup; i++) fn();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  return (performance.now() - t0) / iters;
}

const fmtTime = (ms: number) =>
  ms >= 1
    ? `${ms.toFixed(2)} ms`
    : ms >= 0.001
      ? `${(ms * 1000).toFixed(2)} us`
      : `${(ms * 1e6).toFixed(0)} ns`;
const fmtRate = (ms: number) => {
  const hz = 1000 / ms;
  return hz >= 1e6
    ? `${(hz / 1e6).toFixed(1)} MHz`
    : hz >= 1e3
      ? `${(hz / 1e3).toFixed(1)} kHz`
      : `${hz.toFixed(0)} Hz`;
};

console.log('===============================================================');
console.log('   Double Inverted Pendulum Toolkit — Performance Benchmark   ');
console.log('===============================================================');
console.log(`Machine: ${os.cpus()[0].model} | Node ${process.version}\n`);

const s0: State = [0, 0, 0.1, 0, -0.05, 0];
const sJac: State = [0.3, -1.1, 2.4, 1.7, -2.0, 0.6];
const rows: [string, string, number][] = [];

// 1. Physics
let sCurr = s0;
rows.push([
  'RK4 physics step (6D non-linear)',
  '100,000',
  bench(
    () => {
      sCurr = rk4(sCurr, 2.0, 0.002, DEFAULT_PLANT_PARAMS);
    },
    100_000,
    1000
  ),
]);

// 2. Linearization
const fxBuf = new Float64Array(36);
const fuBuf = new Float64Array(6);
rows.push([
  'RK4 step Jacobian, analytic (flat buffers)',
  '100,000',
  bench(() => linearizeDiscreteInto(fxBuf, fuBuf, sJac, 42, 0.02), 100_000, 2000),
]);
rows.push([
  'RK4 step Jacobian, analytic (nested arrays)',
  '100,000',
  bench(() => linearizeDiscrete(sJac, 42, 0.02), 100_000, 2000),
]);
rows.push([
  'RK4 step Jacobian, finite-difference',
  '20,000',
  bench(() => linearizeDiscreteFD(sJac, 42, 0.02), 20_000, 2000),
]);

// 3. Riccati
rows.push(['Continuous Riccati (CARE) solve', '1,000', bench(() => computeBalanceLQR(), 1000, 50)]);

// 4. Real-time controllers
const mockTraj: TrajectoryInterpolation = {
  getState: (_t: number): State => [0, 0, 0, 0, 0, 0],
  getControl: (_t: number): number => 0,
};
const trackingMPC = new TrackingMPC(mockTraj, 0.5, 0.02);
let tk = 0;
rows.push([
  'Tracking MPC (1-step RTI, 0.5 s horizon)',
  '2,000',
  bench(() => trackingMPC.computeControl(s0, tk++ * 0.02), 2000, 100),
]);

const energyNMPC = new EnergyNMPC(0.8, 0.02);
rows.push([
  'Energy NMPC (5 iterations, 0.8 s horizon)',
  '300',
  bench(() => energyNMPC.computeControl(s0, 5, 1e-2), 300, 30),
]);

// 5. Offline swing-up solve (4 s, 200 knots, 4 continuation stages)
rows.push(['Swing-up solve, analytic Jacobian', '5', bench(() => solveSwingUp(), 5, 1)]);
rows.push([
  'Swing-up solve, finite-difference Jacobian',
  '3',
  bench(() => solveSwingUp({ linearization: 'finite-difference' }), 3, 1),
]);

const w = Math.max(...rows.map((r) => r[0].length));
console.log(`| ${'Routine / Operation'.padEnd(w)} | Iterations | Mean Time    | Max Frequency |`);
console.log(`| :${'-'.repeat(w - 1)} | :--------- | :----------- | :------------ |`);
for (const [name, iters, ms] of rows) {
  console.log(
    `| ${name.padEnd(w)} | ${iters.padEnd(10)} | ${fmtTime(ms).padEnd(12)} | ${fmtRate(ms).padEnd(13)} |`
  );
}
console.log('\nAll benchmarks executed synchronously in pure single-threaded JavaScript/V8.');
console.log(
  'Numbers depend on the machine and Node version; run `pnpm run bench` to get your own.'
);
