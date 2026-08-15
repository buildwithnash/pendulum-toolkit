import {
  rk4,
  computeBalanceLQR,
  EnergyNMPC,
  TrackingMPC,
  TrajectoryInterpolation,
  STATE_UPRIGHT,
  State,
  DEFAULT_PLANT_PARAMS,
} from '../src/index.js';

console.log('===============================================================');
console.log('   Double Inverted Pendulum Toolkit — Performance Benchmark   ');
console.log('===============================================================\n');

// 1. Physics RK4 Step
const s0: State = [0, 0, 0.1, 0, -0.05, 0];
const rk4Warmup = 1000;
for (let i = 0; i < rk4Warmup; i++) rk4(s0, 5.0, 0.002, DEFAULT_PLANT_PARAMS);

const RK4_ITERS = 100_000;
const tRk4Start = performance.now();
let sCurr = s0;
for (let i = 0; i < RK4_ITERS; i++) {
  sCurr = rk4(sCurr, 2.0, 0.002, DEFAULT_PLANT_PARAMS);
}
const rk4TotalMs = performance.now() - tRk4Start;
const rk4Microseconds = (rk4TotalMs / RK4_ITERS) * 1000;

// 2. CARE Riccati Solver
const careWarmup = 10;
for (let i = 0; i < careWarmup; i++) computeBalanceLQR();

const CARE_ITERS = 100;
const tCareStart = performance.now();
for (let i = 0; i < CARE_ITERS; i++) {
  computeBalanceLQR();
}
const careTotalMs = performance.now() - tCareStart;
const careMs = careTotalMs / CARE_ITERS;

// 3. Real-Time Iteration (RTI) Tracking MPC Step
const mockTraj: TrajectoryInterpolation = {
  getState: (_t: number): State => [0, 0, 0, 0, 0, 0],
  getControl: (_t: number): number => 0,
};
const trackingMPC = new TrackingMPC(mockTraj, 0.5, 0.02);

const rtiWarmup = 50;
for (let i = 0; i < rtiWarmup; i++) trackingMPC.computeControl(s0, 0);

const RTI_ITERS = 1000;
const tRtiStart = performance.now();
for (let i = 0; i < RTI_ITERS; i++) {
  trackingMPC.computeControl(s0, i * 0.02);
}
const rtiTotalMs = performance.now() - tRtiStart;
const rtiMs = rtiTotalMs / RTI_ITERS;

// 4. Online Energy NMPC Step
const energyNMPC = new EnergyNMPC(0.8, 0.02);
const nmpcWarmup = 10;
for (let i = 0; i < nmpcWarmup; i++) energyNMPC.computeControl(s0, 5, 1e-2);

const NMPC_ITERS = 200;
const tNmpcStart = performance.now();
for (let i = 0; i < NMPC_ITERS; i++) {
  energyNMPC.computeControl(s0, 5, 1e-2);
}
const nmpcTotalMs = performance.now() - tNmpcStart;
const nmpcMs = nmpcTotalMs / NMPC_ITERS;

// Print Benchmark Results Table
console.log('| Routine / Operation                 | Iterations | Mean Time         | Max Frequency       |');
console.log('| :---------------------------------- | :--------- | :---------------- | :------------------ |');
console.log(`| RK4 Physics Step (6D non-linear)    | 100,000    | ${rk4Microseconds.toFixed(2).padStart(6, ' ')} µs/step    | ${(1000 / (rk4Microseconds / 1000) / 1000).toFixed(0)} kHz            |`);
console.log(`| Continuous Riccati (CARE) Solve     | 100        | ${careMs.toFixed(2).padStart(6, ' ')} ms/solve   | ${(1000 / careMs).toFixed(0)} Hz              |`);
console.log(`| Tracking MPC (RTI 1-Step Gauss-Newt)| 1,000      | ${rtiMs.toFixed(3).padStart(6, ' ')} ms/solve   | ${(1000 / rtiMs).toFixed(0)} Hz              |`);
console.log(`| Online Energy NMPC (5 iterations)   | 200        | ${nmpcMs.toFixed(2).padStart(6, ' ')} ms/solve   | ${(1000 / nmpcMs).toFixed(0)} Hz              |`);
console.log('\nAll benchmarks executed synchronously in pure single-threaded JavaScript/V8.');
