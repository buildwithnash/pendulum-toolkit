/** Small dense linear algebra on flat row-major Float64Arrays, sized for the Riccati solver. */

/** Frobenius norm of a flat matrix. */
export function frobenius(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

/**
 * Inverse of an n x n row-major matrix by Gauss-Jordan elimination with partial pivoting.
 * Throws if a pivot is exactly zero or non-finite.
 */
export function invert(a: ArrayLike<number>, n: number): Float64Array {
  const w = 2 * n;
  const m = new Float64Array(n * w);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) m[i * w + j] = a[i * n + j];
    m[i * w + n + i] = 1;
  }

  for (let col = 0; col < n; col++) {
    let pivot = col;
    let best = Math.abs(m[col * w + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(m[r * w + col]);
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (best === 0 || !Number.isFinite(best)) throw new Error('Matrix is singular');

    if (pivot !== col) {
      for (let j = 0; j < w; j++) {
        const tmp = m[col * w + j];
        m[col * w + j] = m[pivot * w + j];
        m[pivot * w + j] = tmp;
      }
    }

    const inv = 1 / m[col * w + col];
    for (let j = 0; j < w; j++) m[col * w + j] *= inv;

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r * w + col];
      if (f === 0) continue;
      for (let j = 0; j < w; j++) m[r * w + j] -= f * m[col * w + j];
    }
  }

  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) out[i * n + j] = m[i * w + n + j];
  }
  return out;
}

/**
 * Least-squares solution of A X = B for an m x n matrix A (m >= n) and m x k matrix B,
 * using Householder QR. Returns the n x k solution X. Throws if A is rank deficient.
 */
export function solveLeastSquares(
  A: ArrayLike<number>,
  m: number,
  n: number,
  B: ArrayLike<number>,
  k: number
): Float64Array {
  const a = Float64Array.from(A);
  const b = Float64Array.from(B);
  const v = new Float64Array(m);

  for (let j = 0; j < n; j++) {
    let norm = 0;
    for (let i = j; i < m; i++) norm += a[i * n + j] * a[i * n + j];
    norm = Math.sqrt(norm);
    if (norm === 0) throw new Error('Matrix is rank deficient');

    const alpha = a[j * n + j] > 0 ? -norm : norm;
    let vnorm2 = 0;
    for (let i = j; i < m; i++) {
      v[i] = a[i * n + j] - (i === j ? alpha : 0);
      vnorm2 += v[i] * v[i];
    }
    if (vnorm2 === 0) continue;

    // Apply the reflection H = I - 2 v v^T / (v^T v) to the remaining columns of A and to B.
    for (let c = j; c < n; c++) {
      let dot = 0;
      for (let i = j; i < m; i++) dot += v[i] * a[i * n + c];
      const f = (2 * dot) / vnorm2;
      for (let i = j; i < m; i++) a[i * n + c] -= f * v[i];
    }
    for (let c = 0; c < k; c++) {
      let dot = 0;
      for (let i = j; i < m; i++) dot += v[i] * b[i * k + c];
      const f = (2 * dot) / vnorm2;
      for (let i = j; i < m; i++) b[i * k + c] -= f * v[i];
    }
  }

  // A tiny diagonal entry of R, relative to the largest, means dependent columns.
  let rMax = 0;
  for (let i = 0; i < n; i++) rMax = Math.max(rMax, Math.abs(a[i * n + i]));
  for (let i = 0; i < n; i++) {
    if (Math.abs(a[i * n + i]) <= 1e-13 * rMax) throw new Error('Matrix is rank deficient');
  }

  // Back-substitute R X = (Q^T B)[0..n).
  const x = new Float64Array(n * k);
  for (let c = 0; c < k; c++) {
    for (let i = n - 1; i >= 0; i--) {
      let s = b[i * k + c];
      for (let j = i + 1; j < n; j++) s -= a[i * n + j] * x[j * k + c];
      x[i * k + c] = s / a[i * n + i];
    }
  }
  return x;
}
