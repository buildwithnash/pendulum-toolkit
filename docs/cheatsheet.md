# Control Theory & iLQR Notation Cheat Sheet

A reference guide mapping mathematical notation from the control literature (Todorov, Tassa, Tedrake) to variable names used in this toolkit.

---

### 1. Physical State & Actuation

| Mathematical Symbol            | Code Variable    | Type / Dimension        | Definition & Units                                 | Physical Meaning                                                           |
| :----------------------------- | :--------------- | :---------------------- | :------------------------------------------------- | :------------------------------------------------------------------------- |
| $\mathbf{s}$ (or $\mathbf{x}$) | `s`, `xs`        | `State` / $6 \times 1$  | $[x, v, \theta_1, \omega_1, \theta_2, \omega_2]^T$ | **Full state vector** of cart and both links.                              |
| $x$                            | `s[0]`           | `number`                | Position ($\text{m}$)                              | Cart horizontal displacement on rail ($0 = \text{center}$).                |
| $v = \dot{x}$                  | `s[1]`           | `number`                | Velocity ($\text{m/s}$)                            | Cart horizontal velocity.                                                  |
| $\theta_1$                     | `s[2]`           | `number`                | Angle ($\text{rad}$)                               | Link 1 angle from upright ($0 = \text{upright}, \pm\pi = \text{hanging}$). |
| $\omega_1 = \dot{\theta}_1$    | `s[3]`           | `number`                | Angular rate ($\text{rad/s}$)                      | Link 1 angular velocity.                                                   |
| $\theta_2$                     | `s[4]`           | `number`                | Angle ($\text{rad}$)                               | Link 2 angle from upright ($0 = \text{upright}, \pm\pi = \text{hanging}$). |
| $\omega_2 = \dot{\theta}_2$    | `s[5]`           | `number`                | Angular rate ($\text{rad/s}$)                      | Link 2 angular velocity.                                                   |
| $u$                            | `u`, `us`        | `number` / $1 \times 1$ | Force ($\text{N}$)                                 | **Control input**: horizontal motor force applied to cart.                 |
| $\Delta t$                     | `dt`, `DT`       | `number`                | Seconds ($\text{s}$)                               | Discrete timestep interval (e.g. $0.02\text{s} = 50\text{ Hz}$).           |
| $E$                            | `totalEnergy(s)` | `number`                | Joules ($\text{J}$)                                | Total mechanical energy ($T_{\text{kinetic}} + V_{\text{potential}}$).     |

---

### 2. Cost Function Weights & Bounds

| Mathematical Symbol                         | Code Variable    | Type                       | Definition & Intuition                                                                              |
| :------------------------------------------ | :--------------- | :------------------------- | :-------------------------------------------------------------------------------------------------- |
| $J$                                         | `J`, `cost`      | `number`                   | **Total cost** of the trajectory ($J = \sum_{t=0}^{N-1} l(x_t, u_t) + l_f(x_N)$).                   |
| $\mathbf{Q}$ (or $\mathbf{Q}_{\text{run}}$) | `Q`, `Qrun`      | $6 \times 6$ or $6$-vector | **Running state penalty**: Penalizes tracking error along knot points $0 \dots N-1$.                |
| $R$                                         | `R`              | `number` (scalar)          | **Control penalty**: Penalizes control effort ($0.5 R u^2$) to avoid excessive motor torque.        |
| $\mathbf{Q}_f$                              | `Qf`             | $6 \times 6$ or $6$-vector | **Terminal state penalty**: Heavy penalty on final knot $N$ ensuring it lands at the goal.          |
| $\mathbf{P}$                                | `P`, `P_BALANCE` | $6 \times 6$ matrix        | **Riccati Cost-to-Go matrix**: Exact steady-state curvature $V(x) = \frac{1}{2} x^T P x$ from CARE. |
| $x_{\text{lim}}, u_{\text{lim}}$            | `xLim`, `uLim`   | `number`                   | Physical rail travel limit ($\text{m}$) and motor maximum push limit ($\text{N}$).                  |

---

### 3. Dynamics & Linearization Jacobians

| Mathematical Symbol                             | Code Variable | Dimension    | Definition & Purpose                                                                                      |
| :---------------------------------------------- | :------------ | :----------- | :-------------------------------------------------------------------------------------------------------- |
| $\mathbf{M}(q)$                                 | `Mm`          | $3 \times 3$ | **Generalized Mass Matrix**: Non-linear inertial coupling between cart and joints.                        |
| $\mathbf{C}(q, \dot{q})\dot{q} + \mathbf{G}(q)$ | `h1, h2, h3`  | $3 \times 1$ | **Non-inertial Forces**: Combined Coriolis, centrifugal, and gravity forces.                              |
| $\mathbf{f}_x$ (or $\mathbf{A}$)                | `fx`, `A`     | $6 \times 6$ | **State Jacobian**: $\partial f / \partial x$ — sensitivity of next state to current state perturbations. |
| $\mathbf{f}_u$ (or $\mathbf{B}$)                | `fu`, `B`     | $6 \times 1$ | **Control Jacobian**: $\partial f / \partial u$ — sensitivity of next state to control input changes.     |
| $\epsilon$                                      | `eps`         | `number`     | Step size for finite-difference numerical gradients ($10^{-6}$).                                          |

---

### 4. iLQR / DDP Backward Pass ($Q$-Function & Value Function)

In iterative LQR, we approximate the **Value function** $V(x)$ and **Action-Value function** $Q(x, u)$ quadratically:

$$Q(x, u) \approx \frac{1}{2} \begin{bmatrix} 1 \\ \delta x \\ \delta u \end{bmatrix}^T \begin{bmatrix} 0 & Q_x^T & Q_u^T \\ Q_x & Q_{xx} & Q_{ux}^T \\ Q_u & Q_{ux} & Q_{uu} \end{bmatrix} \begin{bmatrix} 1 \\ \delta x \\ \delta u \end{bmatrix}$$

| Variable          | Code Name        | Dimension                | Intuition & Meaning                                                                                                 |
| :---------------- | :--------------- | :----------------------- | :------------------------------------------------------------------------------------------------------------------ |
| $\mathbf{V}_x$    | `Vx`             | $6 \times 1$             | **Value Gradient**: Direction of greatest cost increase from current state.                                         |
| $\mathbf{V}_{xx}$ | `Vxx`            | $6 \times 6$             | **Value Hessian**: Local curvature of the cost bowl around the trajectory.                                          |
| $l_x, l_u$        | `lx`, `lu`       | $6 \times 1, 1 \times 1$ | Gradients of immediate stage cost $l(x,u)$ w.r.t. state and control.                                                |
| $l_{xx}, l_{uu}$  | `lxxDiag`, `luu` | $6 \times 6, 1 \times 1$ | Hessians (second derivatives) of immediate stage cost.                                                              |
| $\mathbf{Q}_x$    | `Qx`             | $6 \times 1$             | $\mathbf{l}_x + \mathbf{f}_x^T \mathbf{V}_x'$ — Total sensitivity of future cost to state.                          |
| $Q_u$             | `Qu`             | `number` (scalar)        | $l_u + \mathbf{f}_u^T \mathbf{V}_x'$ — Total sensitivity of future cost to control force.                           |
| $\mathbf{Q}_{xx}$ | `Qxx`            | $6 \times 6$             | $\mathbf{l}_{xx} + \mathbf{f}_x^T \mathbf{V}_{xx}' \mathbf{f}_x$ — State-state curvature of action-value.           |
| $\mathbf{Q}_{ux}$ | `Qux`            | $1 \times 6$             | $\mathbf{f}_u^T \mathbf{V}_{xx}' \mathbf{f}_x$ — Cross-term coupling between state and control.                     |
| $Q_{uu}$          | `Quu`            | `number` (scalar)        | $l_{uu} + \mathbf{f}_u^T \mathbf{V}_{xx}' \mathbf{f}_u$ — Control curvature (strictly positive for convex descent). |

---

### 5. Policy Extraction & Regularization

| Variable       | Code Name     | Dimension           | Meaning & Role                                                                                                                                                       |
| :------------- | :------------ | :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathbf{k}_t$ | `ks`, `k_t`   | `number` (scalar)   | **Feedforward adjustment step**: $k_t = -Q_{uu,\text{reg}}^{-1} Q_u$.                                                                                                |
| $\mathbf{K}_t$ | `Ks`, `K_t`   | $1 \times 6$ vector | **Feedback gain matrix**: $K_t = -Q_{uu,\text{reg}}^{-1} Q_{ux}$ (real-time error correction rule).                                                                  |
| $\mu$          | `mu`          | `number`            | **Levenberg-Marquardt regularization parameter**: Added to $Q_{uu}$ ($Q_{uu,\text{reg}} = Q_{uu} + \mu$) to guarantee positive-definiteness when curvature is small. |
| $\alpha$       | `alphas`, `a` | `number`            | **Backtracking line search factor**: Step length multiplier ($\alpha \in [1.0, 0.8, \dots, 0.005]$).                                                                 |
| $u_t(x)$       | `u`           | `number`            | **Closed-loop control law**: $u_t = u_{\text{nominal},t} + \alpha k_t + K_t (x - x_{\text{nominal},t})$.                                                             |
