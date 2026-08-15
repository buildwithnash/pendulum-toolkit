import {
  computeBalanceLQR,
  computeBrakeLQR,
  evaluateLQR,
  rk4,
  totalEnergy,
  sweepBasin,
  STATE_UPRIGHT,
  STATE_HANGING,
  State,
} from '../src/index.js';

console.log('=== LQR Balance & Hanging Brake Demonstration ===\n');

// 1. Upright Balance LQR
const balance = computeBalanceLQR();
console.log('1. Upright Balance Controller (CARE Solution):');
console.log(`   Feedback Gain K = [${balance.K.map((v) => v.toFixed(2)).join(', ')}]`);

// Simulate upright balance under an initial perturbation
let s: State = [0, 0, 0.15, 0, -0.1, 0]; // ~8.5° tilt on link 1, ~5.7° tilt on link 2
console.log(`\nSimulating 3.0 seconds of balance from perturbed initial state:`);
console.log(
  `   Initial State: [x=${s[0].toFixed(2)}m, th1=${((s[2] * 180) / Math.PI).toFixed(1)}°, th2=${((s[4] * 180) / Math.PI).toFixed(1)}°]`
);

const dt = 0.005;
const steps = Math.round(3.0 / dt);
for (let i = 0; i < steps; i++) {
  const u = evaluateLQR(s, balance.K, STATE_UPRIGHT);
  s = rk4(s, u, dt);
}
console.log(
  `   Settled State: [x=${s[0].toFixed(4)}m, th1=${((s[2] * 180) / Math.PI).toFixed(4)}°, th2=${((s[4] * 180) / Math.PI).toFixed(4)}°]`
);
console.log(`   Final Energy:  ${totalEnergy(s).toFixed(2)} J (Upright nominal: 19.62 J)`);

// 2. Hanging Brake LQR
const brake = computeBrakeLQR();
console.log('\n2. Hanging Brake Controller:');
console.log(`   Feedback Gain K = [${brake.K.map((v) => v.toFixed(2)).join(', ')}]`);

// Simulate fall from near upright down into hanging brake
let sFall: State = [0, 0, 0.5, 0, 0.8, 0]; // Large angle falling over
console.log(`\nSimulating active brake catch from falling state:`);
console.log(`   Initial Energy: ${totalEnergy(sFall).toFixed(2)} J`);

for (let i = 0; i < Math.round(5.0 / dt); i++) {
  const u = evaluateLQR(sFall, brake.K, STATE_HANGING);
  sFall = rk4(sFall, u, dt);
}
console.log(
  `   Braked State:  [x=${sFall[0].toFixed(4)}m, v=${sFall[1].toFixed(4)}m/s, th1=${((sFall[2] * 180) / Math.PI).toFixed(1)}°, th2=${((sFall[4] * 180) / Math.PI).toFixed(1)}°]`
);
console.log(`   Final Energy:  ${totalEnergy(sFall).toFixed(2)} J (Hanging nominal: -19.62 J)`);

// 3. Basin of Attraction
console.log('\n3. Evaluating Basin of Attraction Grid (31x31 points)...');
const basin = sweepBasin(balance.K, { resolution: 31, maxAngle: 0.5 });
const stablePoints = basin.filter((p) => p.stable).length;
console.log(
  `   Basin coverage: ${stablePoints} / ${basin.length} initial condition pairs stabilized (${((stablePoints / basin.length) * 100).toFixed(1)}%)`
);
