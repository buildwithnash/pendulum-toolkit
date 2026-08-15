# Double Inverted Pendulum Control Toolkit

A zero-dependency, high-performance TypeScript control, trajectory optimization, and simulation library for underactuated double inverted pendulums on a cart.

[![CI](https://github.com/buildwithnash/pendulum-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/buildwithnash/pendulum-toolkit/actions)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-brightgreen.svg)](https://buildwithnash.github.io/pendulum-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/Language-TypeScript%205.5-3178C6.svg)](https://www.typescriptlang.org/)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-0%20runtime-brightgreen.svg)](<>)

> 🎮 **[Launch Live Interactive Canvas Demo →](https://buildwithnash.github.io/pendulum-toolkit/)**

---

## Overview

The double inverted pendulum on a cart is a classic benchmark in underactuated nonlinear robotics: **one actuator** (horizontal force on the cart $u$) controls **three degrees of freedom** ($x$, $\theta_1$, $\theta_2$).

This toolkit provides modern control pipelines in pure TypeScript, running with zero runtime dependencies in **Node.js, Bun, and modern browsers**:

- **Continuous Algebraic Riccati Equation (CARE) Solver**: Steady-state LQR for upright balance and hanging brake.
- **Iterative Linear Quadratic Regulator (iLQR / DDP)**: Gauss-Newton differential dynamic programming for non-linear swing-up trajectory optimization with soft state/control barriers.
- **Time-Varying LQR (TVLQR)**: Discrete Riccati backward-pass gain scheduling along swing-up trajectories with continuous terminal handover.
- **Real-Time Model Predictive Control (NMPC)**: Real-Time Iteration (RTI) 1-step tracking MPC and online reference-free Energy NMPC.
- **Hardware Bench Simulation**: Simulates physical reality including Coulomb dry friction (stiction), loop transport latency (40ms+ delay), sensor quantization, and forward state prediction.
- **Interactive Browser Canvas Demo**: Standalone zero-build HTML5 Canvas visualizer.

Read the complete technical deep dive and interactive essays at [davidnash.dev](https://davidnash.dev).

---

## Why Zero-Dependency TypeScript?

1. **Frictionless Portability**: No C++ compilation toolchains (CMake, Eigen, BLAS), no Python virtualenv/wheel management, and no native binaries.
2. **Universal Runtime**: The exact same mathematical models and controllers run in high-throughput Node.js microservices, edge workers, and client-side browser simulations at 60+ fps.
3. **High Numerical Performance**: Uses preallocated flat typed buffers (`Float64Array`) and scalar reduction for single-actuator systems, achieving **< 0.5 ms solve times** for real-time MPC loops.

---

## Performance Benchmarks

Run on an Apple M-series processor (single-threaded JavaScript / V8):

| Routine / Operation                         | Iterations | Mean Execution Time  | Max Throughput |
| :------------------------------------------ | :--------- | :------------------- | :------------- |
| **RK4 Physics Step (6D non-linear)**        | 100,000    | **~1.8 µs / step**   | > 500 kHz      |
| **Continuous Riccati (CARE) Solve**         | 100        | **~4.5 ms / solve**  | ~220 Hz        |
| **Tracking MPC (1-Step RTI Gauss-Newton)**  | 1,000      | **~0.35 ms / solve** | > 2,800 Hz     |
| **From-Scratch Energy NMPC (5 iterations)** | 200        | **~1.6 ms / solve**  | ~600 Hz        |

Run the benchmark suite locally:

```bash
pnpm run bench
```

---

## Quickstart

### 1. Installation

```bash
pnpm add @buildwithnash/pendulum-toolkit
# or npm install @buildwithnash/pendulum-toolkit
```

---

### 2. Upright Balancing & Hanging Brake with LQR

```typescript
import {
  computeBalanceLQR,
  computeBrakeLQR,
  evaluateLQR,
  rk4,
  STATE_UPRIGHT,
} from '@buildwithnash/pendulum-toolkit';

// Solve Continuous Algebraic Riccati Equation (CARE) about upright equilibrium
const balance = computeBalanceLQR();
console.log('Balance Gains K:', balance.K); // [K_x, K_v, K_th1, K_w1, K_th2, K_w2]

// Closed-loop simulation step
let state = [0, 0, 0.08, 0, -0.05, 0]; // state: [x, v, th1, w1, th2, w2]
const dt = 0.002; // 500 Hz

for (let step = 0; step < 1000; step++) {
  const u = evaluateLQR(state, balance.K, STATE_UPRIGHT);
  state = rk4(state, u, dt);
}
```

---

### 3. Non-Linear Swing-Up via iLQR & TVLQR

```typescript
import {
  ilqr,
  computeTVLQR,
  computeBalanceLQR,
  STATE_HANGING,
  type CostFunction,
} from '@buildwithnash/pendulum-toolkit';

const dt = 0.02; // 20 ms knot spacing
const N = 200; // 4.0 second horizon

// Define stage and terminal costs with track and actuator limits
const cost: CostFunction = {
  run(s, u) {
    return 0.5 * 0.05 * u * u + 0.5 * (s[0] ** 2 + s[2] ** 2 + s[4] ** 2);
  },
  term(s) {
    return 500 * (s[0] ** 2 + 10 * s[2] ** 2 + 10 * s[4] ** 2);
  },
  runDeriv(s, u) {
    /* gradients & hessians */
  },
  termDeriv(s) {
    /* terminal gradients & hessians */
  },
};

const uGuess = new Array(N).fill(0);
const trajectory = ilqr(STATE_HANGING, uGuess, dt, cost, { maxIter: 300 });

// Compute TVLQR tracking gains K(t) initialized with steady-state balance P
const balance = computeBalanceLQR();
const trackingGains = computeTVLQR(
  trajectory.xs,
  trajectory.us,
  dt,
  [10, 1, 150, 10, 150, 10],
  0.1,
  balance.P
);
```

---

### 4. Real-Time Nonlinear Model Predictive Control (NMPC)

```typescript
import { EnergyNMPC, rk4 } from '@buildwithnash/pendulum-toolkit';

// Reference-free swing-up NMPC with periodic catch bowl and gated LQR cost-to-go blending
const nmpc = new EnergyNMPC(1.0, 0.02); // 1.0s horizon @ 20ms steps
let state = [0, 0, Math.PI, 0, Math.PI, 0]; // hanging start

// 50 Hz control loop
setInterval(() => {
  const u = nmpc.computeControl(state, 15, 1e-3); // ~1-3 ms solve time
  state = rk4(state, u, 0.02);
}, 20);
```

---

### 5. Hardware Realism: Transport Delay & Stiction Compensation

```typescript
import {
  HardwareBench,
  predictForward,
  computeBalanceLQR,
  evaluateLQR,
  STATE_UPRIGHT,
} from '@buildwithnash/pendulum-toolkit';

const bench = new HardwareBench([0, 0, 0.08, 0, -0.05, 0], {
  loopDelay: 0.04, // 40 ms transport / loop latency
  coulombFriction: 0.8, // 0.8 N dry Coulomb friction
  encoderBits: 14, // 14-bit angular encoder quantization
});

const balance = computeBalanceLQR();
const dt = 0.002;

for (let i = 0; i < 2000; i++) {
  const measured = bench.getSensorMeasurement();

  // Model-based state prediction across loop delay with friction feedforward
  const predicted = predictForward(measured, bench.getControlHistory(), dt, 0.04, {
    coulombEstimate: 0.8 * 0.85,
  });

  const u = evaluateLQR(predicted, balance.K, STATE_UPRIGHT);
  bench.step(u, dt);
}
```

---

## Notation & Variable Cheat Sheet

When studying control theory papers (Todorov, Tassa, Tedrake) and working with this codebase, here is a quick mapping of key terms and variable names:

| Mathematical Symbol                                             | Code Variable       | Concept & Intuition                                                                                                                                                                        |
| :-------------------------------------------------------------- | :------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathbf{s} = [x, v, \theta_1, \omega_1, \theta_2, \omega_2]^T$ | `s`, `xs`           | **State Vector**: Cart position ($x$) & velocity ($v$), Link 1 angle ($\theta_1$) & rate ($\omega_1$), Link 2 angle ($\theta_2$) & rate ($\omega_2$). Upright is $0$, hanging is $\pm\pi$. |
| $u$                                                             | `u`, `us`           | **Control Input**: Horizontal force in Newtons applied to the cart.                                                                                                                        |
| $\mathbf{Q}, R, \mathbf{Q}_f$                                   | `Q`, `R`, `Qf`      | **Cost Matrices**: State error penalty ($Q$), control effort penalty ($R$), and terminal goal penalty ($Q_f$).                                                                             |
| $\mathbf{P}$                                                    | `P`, `P_BALANCE`    | **Cost-to-Go Matrix**: Steady-state Riccati solution ($V(x) = \frac{1}{2} x^T P x$) from CARE.                                                                                             |
| $\mathbf{V}_x, \mathbf{V}_{xx}$                                 | `Vx`, `Vxx`         | **Value Function Gradient & Hessian**: Slope and curvature of total remaining cost from current state forward.                                                                             |
| $\mathbf{Q}_x, Q_u$                                             | `Qx`, `Qu`          | **Action-Value Gradient**: First derivatives of total cost w.r.t. state and control.                                                                                                       |
| $\mathbf{Q}_{xx}, \mathbf{Q}_{ux}, Q_{uu}$                      | `Qxx`, `Qux`, `Quu` | **Action-Value Curvature**: Second derivatives of cost. Single-actuator control makes $Q_{uu}$ a scalar!                                                                                   |
| $\mathbf{k}_t$                                                  | `ks`, `k_t`         | **Feedforward Adjustment**: $-Q_{uu,\text{reg}}^{-1} Q_u$ (nominal force adjustment step).                                                                                                 |
| $\mathbf{K}_t$                                                  | `Ks`, `K_t`         | **Feedback Gain Matrix**: $-Q_{uu,\text{reg}}^{-1} Q_{ux}$ (real-time corrective feedback rule).                                                                                           |
| $\mu$                                                           | `mu`                | **Levenberg-Marquardt Damping**: Regularization added to $Q_{uu}$ ($Q_{uu} + \mu$) to ensure positive curvature and convex steps.                                                          |
| $\alpha$                                                        | `alphas`, `a`       | **Line Search Factor**: Step size scalar along the search direction ($\alpha \in [1.0, 0.8, \dots, 0.005]$).                                                                               |

_(See [`docs/cheatsheet.md`](docs/cheatsheet.md) for the complete comprehensive reference table)._

---

## Interactive Browser Canvas Demo

The repository includes a self-contained, zero-build HTML5 Canvas visualizer:

🎮 **[Click here to open the Live GitHub Pages Demo](https://buildwithnash.github.io/pendulum-toolkit/)**

Or open [`examples/browser-demo/index.html`](examples/browser-demo/index.html) locally in any browser to:

- **Switch Modes**:
  - **Auto (iLQR + TVLQR &rarr; LQR)**: Smooth precomputed 4.0s swing-up with linear feedback tracking and automatic steady-state balance handover.
  - **Real-Time Energy NMPC**: From-scratch online trajectory discovery and catch without precomputed references.
  - **Hold Balance LQR & Hanging Brake**: Steady-state CARE Riccati equilibrium regulators.
- **Real-Time Horizon Prediction Ghosts**: Visualizes the solver's rolling planned trajectory fanning out in translucent preview links ahead of the cart.
- **Live Compute Speed & Budget Telemetry**: Color-coded monitor verifying real-time loop feasibility at 100 Hz / 200 Hz.
- **Interactive Tuning**: Adjust control loop frequency (20–200 Hz) and horizon length (0.5–1.5s) on the fly.
- **Direct Disturbance Testing**: Drag the cart with mouse/touch or inject velocity perturbations to stress test controller stability.

---

## Mathematical Formulation

### 1. Plant Dynamics

The Lagrangian equations of motion for a cart of mass $M$ and two links of masses $m_1, m_2$ and lengths $L_1, L_2$ are:

$$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) + D\dot{q} = \begin{bmatrix} u \\ 0 \\ 0 \end{bmatrix}$$

where $q = [x, \theta_1, \theta_2]^T$ and $u$ is the horizontal force on the cart.

### 2. Continuous Algebraic Riccati Equation (CARE)

The linearised continuous-time system $\dot{x} = Ax + Bu$ minimises:

$$J = \int_0^\infty \left( x^T Q x + u^T R u \right) dt$$

The steady-state solution $P$ satisfies:

$$A^T P + P A - P B R^{-1} B^T P + Q = 0$$

yielding the optimal feedback law $u = -K x = -R^{-1} B^T P x$.

### 3. Iterative LQR (iLQR / DDP)

For non-linear discrete dynamics $x_{t+1} = f(x_t, u_t)$, the value function $V(x)$ is approximated quadratically:

$$Q_x = l_x + f_x^T V_x', \quad Q_u = l_u + f_u^T V_x'$$
$$Q_{xx} = l_{xx} + f_x^T V_{xx}' f_x, \quad Q_{uu} = l_{uu} + f_u^T V_{xx}' f_u, \quad Q_{ux} = f_u^T V_{xx}' f_x$$

For single-actuator systems ($\dim(u) = 1$), $Q_{uu}$ is a scalar, bypassing all matrix inverses:

$$k_t = -\frac{Q_u}{Q_{uu} + \mu}, \quad K_t = -\frac{Q_{ux}}{Q_{uu} + \mu}$$

---

## Runnable Examples

Run any example directly using `tsx`:

```bash
# Offline iLQR swing-up and TVLQR gain scheduling
pnpm run example:swingup

# Balance and hanging brake LQR with basin of attraction sweep
pnpm run example:lqr

# Real-time 50 Hz / 100 Hz NMPC simulation
pnpm run example:nmpc

# Hardware bench with 40ms transport delay and Coulomb friction
pnpm run example:bench

# Run full performance benchmark suite
pnpm run bench
```

Run test suite:

```bash
pnpm test
```

---

## License

MIT License © 2026 David Nash. See [LICENSE](LICENSE) for details.
