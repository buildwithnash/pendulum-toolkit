import {
  ilqr,
  computeTVLQR,
  computeBalanceLQR,
  totalEnergy,
  STATE_HANGING,
  CostFunction,
  State,
  STATE_DIM,
} from '../src/index.js';

console.log('=== Double Pendulum Swing-Up Trajectory Optimization (iLQR + TVLQR) ===\n');

const DT = 0.02; // 20 ms
const DURATION = 4.0; // 4.0 seconds
const N = Math.round(DURATION / DT);

const R = 0.05;
const Qrun = [0.2, 0.1, 1.0, 0.2, 1.0, 0.2];
const Qf = [500, 100, 3000, 400, 3000, 400];
const xLim = 1.0;
const uLim = 30.0;
const wTrack = 200.0;
const wForce = 10.0;

const hinge = (z: number, lim: number) => Math.max(0, Math.abs(z) - lim) * Math.sign(z);

/**
 * Cost formulation using (1 - cos θ) for angles (θ1=s[2], θ2=s[4])
 * to guarantee C^∞ smoothness everywhere without branch cut discontinuities.
 */
const cost: CostFunction = {
  run(s: State, u: number): number {
    let c = 0.5 * R * u * u;
    // Cart position and velocities (quadratic)
    c += 0.5 * Qrun[0] * s[0] ** 2;
    c += 0.5 * Qrun[1] * s[1] ** 2;
    c += 0.5 * Qrun[3] * s[3] ** 2;
    c += 0.5 * Qrun[5] * s[5] ** 2;

    // Angles: periodic (1 - cos θ), Taylor series matches 0.5 * θ^2 near 0
    c += Qrun[2] * (1 - Math.cos(s[2]));
    c += Qrun[4] * (1 - Math.cos(s[4]));

    // Soft hinge barrier constraints
    c += 0.5 * wTrack * hinge(s[0], xLim) ** 2;
    c += 0.5 * wForce * hinge(u, uLim) ** 2;
    return c;
  },

  term(s: State): number {
    let c = 0;
    c += 0.5 * Qf[0] * s[0] ** 2;
    c += 0.5 * Qf[1] * s[1] ** 2;
    c += 0.5 * Qf[3] * s[3] ** 2;
    c += 0.5 * Qf[5] * s[5] ** 2;

    // Terminal angles: heavy (1 - cos θ) potential ensuring smooth upright catch
    c += Qf[2] * (1 - Math.cos(s[2]));
    c += Qf[4] * (1 - Math.cos(s[4]));
    return c;
  },

  runDeriv(s: State, u: number) {
    const lx = new Float64Array(STATE_DIM);
    const lxx = new Float64Array(STATE_DIM);

    // Cart position and velocities
    lx[0] = Qrun[0] * s[0];
    lxx[0] = Qrun[0];
    lx[1] = Qrun[1];
    lxx[1] = Qrun[1];
    lx[3] = Qrun[3];
    lxx[3] = Qrun[3];
    lx[5] = Qrun[5];
    lxx[5] = Qrun[5];

    // Angles: d/dθ [Q*(1 - cos θ)] = Q*sin(θ), d²/dθ² = Q*cos(θ)
    lx[2] = Qrun[2] * Math.sin(s[2]);
    lxx[2] = Math.max(0, Qrun[2] * Math.cos(s[2]));
    lx[4] = Qrun[4] * Math.sin(s[4]);
    lxx[4] = Math.max(0, Qrun[4] * Math.cos(s[4]));

    // Barrier derivatives
    const hx = hinge(s[0], xLim);
    lx[0] += wTrack * hx;
    if (hx !== 0) lxx[0] += wTrack;

    const hu = hinge(u, uLim);
    const lu = R * u + wForce * hu;
    const luu = R + (hu !== 0 ? wForce : 0);
    return { lx, lxxDiag: lxx, lu, luu };
  },

  termDeriv(s: State) {
    const lx = new Float64Array(STATE_DIM);
    const lxx = new Float64Array(STATE_DIM);

    lx[0] = Qf[0] * s[0];
    lxx[0] = Qf[0];
    lx[1] = Qf[1];
    lxx[1] = Qf[1];
    lx[3] = Qf[3];
    lxx[3] = Qf[3];
    lx[5] = Qf[5];
    lxx[5] = Qf[5];

    // Terminal angles
    lx[2] = Qf[2] * Math.sin(s[2]);
    lxx[2] = Math.max(0, Qf[2] * Math.cos(s[2]));
    lx[4] = Qf[4] * Math.sin(s[4]);
    lxx[4] = Math.max(0, Qf[4] * Math.cos(s[4]));

    return { lx, lxxDiag: lxx };
  },
};

// Initial control sequence guess: small sinusoidal pump
const usInit = new Array(N).fill(0).map((_, i) => 8.0 * Math.sin((2 * Math.PI * i * DT) / 1.5));

console.log(`Starting iLQR solve: N=${N} steps (${DURATION}s @ ${DT * 1000}ms dt)`);
const t0 = performance.now();
const result = ilqr(STATE_HANGING, usInit, DT, cost, {
  maxIter: 300,
  tol: 1e-6,
  verbose: true,
});
const solveTimeMs = performance.now() - t0;

console.log(
  `\nSolve completed in ${solveTimeMs.toFixed(1)} ms (${result.iters} iterations, converged: ${result.converged})`
);
console.log(`Final cost: ${result.cost.toFixed(4)}`);

const initialE = totalEnergy(result.xs[0]);
const finalE = totalEnergy(result.xs[N]);
const peakForce = Math.max(...result.us.map(Math.abs));
const maxCartX = Math.max(...result.xs.map((x) => Math.abs(x[0])));

console.log(
  `Energy transition: ${initialE.toFixed(2)} J -> ${finalE.toFixed(2)} J (target: +19.62 J)`
);
console.log(`Peak cart force: ${peakForce.toFixed(2)} N (limit: ${uLim} N)`);
console.log(`Peak cart excursion: ${maxCartX.toFixed(3)} m (limit: ${xLim} m)`);

// Compute TVLQR tracking gains along nominal trajectory
console.log('\nComputing Time-Varying LQR gain schedule K(t)...');
const balance = computeBalanceLQR();
const Ks = computeTVLQR(result.xs, result.us, DT, [10, 1, 150, 10, 150, 10], 0.1, balance.P);

console.log(`Successfully generated ${Ks.length} TVLQR gain vectors.`);
console.log(`Final TVLQR gain at t=4.0s: [${Ks[N - 1].map((v) => v.toFixed(2)).join(', ')}]`);
console.log(`Steady-state balance LQR:   [${balance.K.map((v) => v.toFixed(2)).join(', ')}]`);
