# Reservoir Modeling

> This is a quick summary from a larger draft written by Abhiroop Chattopadhyay (not shown) and thus may be out-of-date. [Email him](ac33@illinois.edu) for more details.

We use a multi-period model where the set $\mathcal K = \{0, 1, 2, \dots, N - 1\}$ denotes a set of $N$ periods.

For a period $k \in \mathcal K$:

- Let $s(k)$ and $a(k)$ denote the storage/volume and area of the reservoir **at the start of the period**.
- Let $r(k)$, $e(k)$, $q(k)$ denote the preciptation inflow, loss by evaporation, and pumping outflow **over the period**.

1. We can model the reservoir storage dynamics as a linear volumetric balance formulation.

$$v(k + 1) = s(k) + r(k) - e(k) - q(k)$$

2. Reservoir overflow occurs when the inflow $r(k)$ causes the reservoir to reach its maximum volume $S$, i.e. when $v(k + 1) > S$. Let the following denote the non-negative overflow voume **over the period**:

$$w(k) = \text{max}\{v(k + 1) - S, 0\}$$

3. We derive the true storage state of the reservoir $s(k + 1)$ as the difference between the "theoretical" volume and overflow: $s(k + 1) = v(k + 1) - w(k)$.

4. We establish a power-law relationship between area $a(k)$ and volume $s(k)$, which enables us to estimate one off the other. We derive the coefficients $\gamma_1$ and $\gamma_2$ from a regression analysis on similar reservoirs (not shown).

$$s(k) = \gamma_1 a(k)^{\gamma_2}$$

> For a fully-linear model, this can alternatively be $s(k) = \frac SA \cdot A(k)$.

5. We upper-bound the surface area $a(k)$ to a maximum $A$: $a(k) \leq A$.

6. We estimate rainfall inflow $r(k)$ **over the period** as the product of the percepitation rate $p(k)$ and the catchment area $A_c$ with an infilitration coefficient $C_i$: $r(k) = C_i A_c p(k)$.

7. Similarly, we estimate the evaporation $e(k)$ **over a period** as the product of the evaporation depth $e_p(k)$ and average surface area with the evaporation pan coefficient $C_p$:

$$e(k) = \left(\frac{a(k + 1) + a(k)}2\right) \cdot C_p e_p(k)$$

8. Finally, we upper-bound the pumping rate $q(k)$ to a maximum $Q$: $q(k) \leq Q$.

9. We define the objective minimization function as the total overflow across all periods:

$$\text{min} \sum_k w(k)$$

## Bounding Overflow

Pyomo does not support binary decision variables/computations, so we cannot directly encode our original $\text{max}$ formulation for $w(k)$. Instead, we simulate this by introducing an unbounded nonphysical overflow $w_u$ that can be negative, i.e. when $v(t + 1) < S$.

$$
\begin{align*}
    w_u(t)
    &= v(t + 1) - S \\
    w(t)
    &= \text{max}(0, w_u(t)) \\
    &= \frac{w_u + |w_u|}2
\end{align*}
$$
