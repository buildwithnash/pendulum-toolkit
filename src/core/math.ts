/**
 * Wraps an angle to [-PI, PI].
 */
export function wrapPi(theta: number): number {
  let a = theta % (2 * Math.PI);
  if (a > Math.PI) a -= 2 * Math.PI;
  if (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

/**
 * Creates an n-element array initialized with zeros.
 */
export function zeros(n: number): number[] {
  return new Array(n).fill(0);
}

/**
 * Creates an n x m 2D array initialized with zeros.
 */
export function zeros2(n: number, m: number): number[][] {
  return Array.from({ length: n }, () => new Array(m).fill(0));
}

/**
 * Clones a 2D matrix.
 */
export function cloneMatrix(M: number[][]): number[][] {
  return M.map((row) => row.slice());
}

/**
 * Matrix transpose: M^T
 */
export function transpose(M: number[][]): number[][] {
  const rows = M.length;
  const cols = M[0].length;
  const res = zeros2(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      res[j][i] = M[i][j];
    }
  }
  return res;
}

/**
 * Matrix multiplication: A * B
 */
export function matMul(A: number[][], B: number[][]): number[][] {
  const rA = A.length;
  const cA = A[0].length;
  const cB = B[0].length;
  const res = zeros2(rA, cB);
  for (let i = 0; i < rA; i++) {
    for (let k = 0; k < cA; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < cB; j++) {
        res[i][j] += aik * B[k][j];
      }
    }
  }
  return res;
}

/**
 * Matrix addition: A + B
 */
export function matAdd(A: number[][], B: number[][]): number[][] {
  const rows = A.length;
  const cols = A[0].length;
  const res = zeros2(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      res[i][j] = A[i][j] + B[i][j];
    }
  }
  return res;
}

/**
 * Matrix scalar scaling: s * A
 */
export function matScale(A: number[][], s: number): number[][] {
  const rows = A.length;
  const cols = A[0].length;
  const res = zeros2(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      res[i][j] = A[i][j] * s;
    }
  }
  return res;
}

/**
 * Analytical inverse of a 3x3 matrix.
 */
export function inv3(m: number[][]): number[][] {
  const det =
    m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

  if (Math.abs(det) < 1e-15) {
    throw new Error('Matrix is singular or near-singular');
  }

  const id = 1 / det;
  return [
    [
      (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * id,
      (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id,
      (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id,
    ],
    [
      (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id,
      (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id,
      (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * id,
    ],
    [
      (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * id,
      (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * id,
      (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * id,
    ],
  ];
}
