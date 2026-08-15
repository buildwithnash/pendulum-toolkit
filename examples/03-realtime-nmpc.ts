import { EnergyNMPC, rk4, totalEnergy, STATE_HANGING, State } from '../src/index.js';

console.log('=== Real-Time Nonlinear Model Predictive Control (NMPC) ===\n');

// 1. From-Scratch Energy NMPC
console.log('1. Simulating Online Swing-Up Discovery with From-Scratch Energy NMPC:');
console.log('   (No precomputed reference trajectory — discovering swing-up online)\n');

const nmpc = new EnergyNMPC(1.0, 0.02); // 1.0s horizon @ 20ms steps (50 knot points)
let s: State = [...STATE_HANGING] as State;
const dt = 0.02;
const simDuration = 4.0;
const steps = Math.round(simDuration / dt);

const solveTimes: number[] = [];

for (let step = 0; step < steps; step++) {
  const t0 = performance.now();
  const u = nmpc.computeControl(s, 15, 1e-3);
  const solveMs = performance.now() - t0;
  solveTimes.push(solveMs);

  s = rk4(s, u, dt);

  if (step % 25 === 0 || step === steps - 1) {
    const t = (step * dt).toFixed(2);
    const E = totalEnergy(s).toFixed(2);
    const x = s[0].toFixed(2);
    const th1 = ((s[2] * 180) / Math.PI).toFixed(1);
    const th2 = ((s[4] * 180) / Math.PI).toFixed(1);
    console.log(
      `   t=${t}s | solve=${solveMs.toFixed(1)}ms | u=${u.toFixed(1)}N | x=${x}m | th1=${th1}° | th2=${th2}° | E=${E} J`
    );
  }
}

const avgSolve = solveTimes.reduce((a, b) => a + b, 0) / solveTimes.length;
const maxSolve = Math.max(...solveTimes);

console.log(`\nPerformance Summary:`);
console.log(`   Average NMPC solve time: ${avgSolve.toFixed(2)} ms`);
console.log(`   Peak NMPC solve time:    ${maxSolve.toFixed(2)} ms`);
console.log(`   Final Energy:            ${totalEnergy(s).toFixed(2)} J (target: +19.62 J)`);
