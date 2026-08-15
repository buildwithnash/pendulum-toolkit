import {
  HardwareBench,
  predictForward,
  computeBalanceLQR,
  evaluateLQR,
  STATE_UPRIGHT,
  State,
} from '../src/index.js';

console.log('=== Hardware Bench Realism: Transport Delay & Coulomb Stiction ===\n');

const balance = computeBalanceLQR();
const DT = 0.002; // 2 ms physics simulation step (500 Hz)
const DELAY_SECONDS = 0.04; // 40 ms transport/loop delay
const STICTION_N = 0.8; // 0.8 N dry Coulomb friction

function runSimulation(usePredictor: boolean) {
  const s0: State = [0, 0, 0.1, 0, -0.06, 0]; // Perturbed start
  const bench = new HardwareBench(s0, {
    loopDelay: DELAY_SECONDS,
    coulombFriction: STICTION_N,
    encoderBits: 14, // 16384 counts/rev encoder
  });

  const duration = 4.0;
  const steps = Math.round(duration / DT);
  let fell = false;

  for (let i = 0; i < steps; i++) {
    const rawState = bench.getSensorMeasurement();
    let controlState = rawState;

    if (usePredictor) {
      controlState = predictForward(rawState, bench.getControlHistory(), DT, DELAY_SECONDS, {
        coulombEstimate: STICTION_N * 0.85,
      });
    }

    let u = evaluateLQR(controlState, balance.K, STATE_UPRIGHT);
    if (Math.abs(u) > 100) u = Math.sign(u) * 100;

    const { state, diedOnRail } = bench.step(u, DT);

    if (diedOnRail || Math.abs(state[2]) > Math.PI / 2 || Math.abs(state[4]) > Math.PI / 2) {
      fell = true;
      console.log(`   [FAIL] Pendulum lost stability at t=${(i * DT).toFixed(2)}s`);
      break;
    }
  }

  if (!fell) {
    const final = bench.state;
    console.log(
      `   [SUCCESS] Settled at t=4.0s! (Cart x=${final[0].toFixed(3)}m, th1=${((final[2] * 180) / Math.PI).toFixed(2)}°, th2=${((final[4] * 180) / Math.PI).toFixed(2)}°)`
    );
  }
}

console.log('1. Test WITHOUT State Predictor (40ms loop delay + 0.8N stiction):');
runSimulation(false);

console.log('\n2. Test WITH Model-Based Forward Predictor:');
runSimulation(true);
