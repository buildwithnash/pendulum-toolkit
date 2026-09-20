import {
  linearizeDiscrete,
  linearizeDiscreteInto,
  linearizeDiscreteFD,
  accelerationsInto,
  getPrecomputed,
  inv3,
  DEFAULT_PLANT_PARAMS,
  State,
} from '../src/index.js';

/**
 * Checks the analytic RK4 Jacobian that iLQR uses against finite differences, times both, and
 * compares heap-allocated arrays with flat scalars. Run with `pnpm run verify:jacobian`.
 */

function mulberry32(seed: number) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- 1. Correctness ---------------------------------------------------------------------------
const rand = mulberry32(1234);
const TRIALS = 500;
let worstFx = 0;
let worstFu = 0;
for (let n = 0; n < TRIALS; n++) {
  const s: State = [
    (rand() - 0.5) * 4,
    (rand() - 0.5) * 6,
    rand() * 2 * Math.PI,
    (rand() - 0.5) * 8,
    rand() * 2 * Math.PI,
    (rand() - 0.5) * 8,
  ];
  const u = (rand() - 0.5) * 400;
  const ana = linearizeDiscrete(s, u, 0.02);
  const num = linearizeDiscreteFD(s, u, 0.02);
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) worstFx = Math.max(worstFx, Math.abs(ana.fx[i][j] - num.fx[i][j]));
    worstFu = Math.max(worstFu, Math.abs(ana.fu[i][0] - num.fu[i][0]));
  }
}
console.log(`Correctness over ${TRIALS} random states (RK4 step, dt = 0.02):`);
console.log(`  max |fx_numeric - fx_analytic| = ${worstFx.toExponential(2)}`);
console.log(`  max |fu_numeric - fu_analytic| = ${worstFu.toExponential(2)}`);
console.log('  (expect ~1e-9: finite-difference truncation error, not disagreement)');

// --- 2. Speed ---------------------------------------------------------------------------------
const sBench: State = [0.3, -1.1, 2.4, 1.7, -2.0, 0.6];
const uBench = 42;
const dt = 0.02;
const fx = new Float64Array(36);
const fu = new Float64Array(6);

const time = (fn: () => void, calls: number) => {
  for (let i = 0; i < Math.min(5000, calls); i++) fn(); // warm up the JIT
  const t0 = performance.now();
  for (let i = 0; i < calls; i++) fn();
  return ((performance.now() - t0) * 1000) / calls; // microseconds per call
};

const CALLS = 100_000;
const fdUs = time(() => linearizeDiscreteFD(sBench, uBench, dt), CALLS);
const nestedUs = time(() => linearizeDiscrete(sBench, uBench, dt), CALLS);
const flatUs = time(() => linearizeDiscreteInto(fx, fu, sBench, uBench, dt), CALLS);

console.log(`\nSpeed over ${CALLS.toLocaleString()} calls:`);
console.log(`  finite-difference:          ${fdUs.toFixed(3)} us/call`);
console.log(
  `  analytic (nested arrays):   ${nestedUs.toFixed(3)} us/call  (${(fdUs / nestedUs).toFixed(2)}x)`
);
console.log(
  `  analytic (flat buffers):    ${flatUs.toFixed(3)} us/call  (${(fdUs / flatUs).toFixed(2)}x)`
);

// --- 3. Allocation vs arithmetic --------------------------------------------------------------
// Same arithmetic in both: one builds nested heap arrays, the other writes scalars into a
// caller-owned buffer. Any gap is allocation and GC.
const pre = getPrecomputed(DEFAULT_PLANT_PARAMS);
const { cartDamp, jointDamp } = DEFAULT_PLANT_PARAMS;

function accelAllocating(s: State, force: number): number[] {
  const [, v, th1, w1, th2, w2] = s;
  const Mm = [
    [pre.d1, pre.d2 * Math.cos(th1), pre.d3 * Math.cos(th2)],
    [pre.d2 * Math.cos(th1), pre.d4, pre.d5 * Math.cos(th1 - th2)],
    [pre.d3 * Math.cos(th2), pre.d5 * Math.cos(th1 - th2), pre.d6],
  ];
  const h1 = -pre.d2 * w1 * w1 * Math.sin(th1) - pre.d3 * w2 * w2 * Math.sin(th2) + cartDamp * v;
  const h2 = pre.d5 * w2 * w2 * Math.sin(th1 - th2) - pre.f1 * Math.sin(th1) + jointDamp * w1;
  const h3 = -pre.d5 * w1 * w1 * Math.sin(th1 - th2) - pre.f2 * Math.sin(th2) + jointDamp * w2;
  const Mi = inv3(Mm);
  const r = [force - h1, -h2, -h3];
  return [
    Mi[0][0] * r[0] + Mi[0][1] * r[1] + Mi[0][2] * r[2],
    Mi[1][0] * r[0] + Mi[1][1] * r[1] + Mi[1][2] * r[2],
    Mi[2][0] * r[0] + Mi[2][1] * r[1] + Mi[2][2] * r[2],
  ];
}

const scratch = new Float64Array(3);
const reference = accelAllocating(sBench, uBench);
accelerationsInto(scratch, sBench, uBench);
const agree = Math.max(...[0, 1, 2].map((i) => Math.abs(reference[i] - scratch[i])));

const EVALS = 3_000_000;
for (let i = 0; i < 20_000; i++) {
  accelAllocating(sBench, uBench);
  accelerationsInto(scratch, sBench, uBench);
}
let t = performance.now();
for (let i = 0; i < EVALS; i++) accelAllocating(sBench, uBench);
const allocMs = performance.now() - t;
t = performance.now();
for (let i = 0; i < EVALS; i++) accelerationsInto(scratch, sBench, uBench);
const scalarMs = performance.now() - t;

console.log(`\nAllocation vs arithmetic, ${EVALS.toLocaleString()} acceleration evaluations:`);
console.log(`  agree to:     ${agree.toExponential(2)}`);
console.log(`  heap arrays:  ${((allocMs * 1e6) / EVALS).toFixed(0)} ns/call`);
console.log(`  flat scalars: ${((scalarMs * 1e6) / EVALS).toFixed(0)} ns/call`);
console.log(
  `  speedup: ${(allocMs / scalarMs).toFixed(2)}x  (same arithmetic, so this is allocator and GC)`
);
