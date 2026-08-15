/**
 * State vector representing the double inverted pendulum on a cart.
 * [x, v, theta1, omega1, theta2, omega2]
 *
 * Coordinates:
 * - x: Cart horizontal position (m)
 * - v: Cart horizontal velocity (m/s)
 * - theta1: Angle of first link from upright (rad). Hanging = ±π.
 * - omega1: Angular velocity of first link (rad/s)
 * - theta2: Angle of second link from upright (rad). Hanging = ±π.
 * - omega2: Angular velocity of second link (rad/s)
 */
export type State = [number, number, number, number, number, number];

/** State dimension: 6 */
export const STATE_DIM = 6;

/** Control input dimension: 1 (horizontal force on cart in Newtons) */
export const CONTROL_DIM = 1;

/** Upright equilibrium state: [0, 0, 0, 0, 0, 0] */
export const STATE_UPRIGHT: State = [0, 0, 0, 0, 0, 0];

/** Hanging equilibrium state: [0, 0, π, 0, π, 0] */
export const STATE_HANGING: State = [0, 0, Math.PI, 0, Math.PI, 0];

/**
 * Plant physical parameters.
 */
export interface PlantParams {
  /** Gravitational acceleration (m/s^2), default 9.81 */
  g: number;
  /** Cart mass (kg), default 1.0 */
  M: number;
  /** Link 1 mass (kg), default 1.0 */
  m1: number;
  /** Link 2 mass (kg), default 1.0 */
  m2: number;
  /** Link 1 length (m), default 1.0 */
  L1: number;
  /** Link 2 length (m), default 1.0 */
  L2: number;
  /** Hinge viscous damping coefficient (N·m·s/rad), default 0.02 */
  jointDamp: number;
  /** Cart rail viscous damping coefficient (N·s/m), default 0.1 */
  cartDamp: number;
}

/** Default physical parameters matching reference implementation */
export const DEFAULT_PLANT_PARAMS: PlantParams = {
  g: 9.81,
  M: 1.0,
  m1: 1.0,
  m2: 1.0,
  L1: 1.0,
  L2: 1.0,
  jointDamp: 0.02,
  cartDamp: 0.1,
};

/**
 * Stage and terminal cost interface for optimization and MPC.
 */
export interface CostFunction {
  /** Running stage cost l(x, u, t) */
  run(s: State, u: number, t: number): number;
  /** Terminal cost l_f(x, t) */
  term(s: State, t: number): number;
  /**
   * Derivatives of stage cost:
   * lx: gradient wrt state (dim 6)
   * lxxDiag: diagonal elements of state Hessian (dim 6)
   * lu: derivative wrt control input (scalar)
   * luu: second derivative wrt control input (scalar)
   */
  runDeriv(
    s: State,
    u: number,
    t: number
  ): { lx: Float64Array | number[]; lxxDiag: Float64Array | number[]; lu: number; luu: number };
  /**
   * Derivatives of terminal cost:
   * lx: gradient wrt state (dim 6)
   * lxxDiag: diagonal elements of state Hessian (dim 6)
   */
  termDeriv(s: State, t: number): { lx: Float64Array | number[]; lxxDiag: Float64Array | number[] };
}

/**
 * Trajectory optimization result from iLQR.
 */
export interface TrajectoryResult {
  /** States along the trajectory, length N + 1 */
  xs: State[];
  /** Control inputs, length N */
  us: number[];
  /** Feedback gain matrices K_t (1x6 vectors), length N */
  Ks: number[][];
  /** Feedforward adjustments k_t, length N */
  ks: number[];
  /** Total trajectory cost */
  cost: number;
  /** Number of iterations performed */
  iters: number;
  /** Whether the solver met convergence tolerance */
  converged: boolean;
}

/**
 * LQR gain result.
 */
export interface LQRResult {
  /** 1x6 Feedback gain vector u = -K * x */
  K: number[];
  /** 6x6 Riccati solution matrix P */
  P: number[][];
  /** 6x6 State transition matrix A */
  A: number[][];
  /** 6x1 Control input matrix B */
  B: number[][];
}
