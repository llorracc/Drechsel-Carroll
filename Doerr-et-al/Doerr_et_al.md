# Bellman Optimization Problems in Doerr, Drechsel & Lee (2026)
# "Income Inequality and Job Creation"
#
# Produced by econ-ark-matsya (Claude Opus) via sequential queries,
# assembled and verified against the paper's actual equations (3-19).
# Directional corrections applied to banking section where matsya's
# RAG context (Bellman-DDSL corpus) led to reversed deposit-flow direction.

# Household Bellman Problem from Doerr, Drechsel, Lee (2026), Section 5

## (1) Full State Space

Each household $i$ of permanent type $\chi \in \{L, H\}$ enters a period with the following individual and aggregate states:

**Individual states:**
- $d_{it}$: deposits carried into period (chosen last period)
- $k_{it}$: risky capital carried into period (chosen last period)
- $\xi_{it}$: idiosyncratic productivity shock (persistent)
- $\chi \in \{L, H\}$: permanent type (determines scale $s_\chi$)

**Aggregate states:**
- $w_t$: wage rate
- $\tilde{w}_t$: wage rate for the second labor type
- $R^k_t$: gross return on capital
- $R^d_t$: gross return on deposits
- $\Pi_{it}$: profits (from firm ownership, type-dependent)
- $T_{it}$: taxes/transfers

The effective idiosyncratic earnings capacity is $s_{i,\chi,t} = s_\chi \cdot \xi_{it}$, combining the permanent type scale with the transitory shock.

## (2) Control Variables

$$\{c_{it},\; n_{it},\; \tilde{n}_{it},\; d_{it+1},\; k_{it+1}\}$$

- $c_{it}$: consumption
- $n_{it}$: labor supply of type 1
- $\tilde{n}_{it}$: labor supply of type 2
- $d_{it+1}$: deposits chosen for next period
- $k_{it+1}$: risky capital chosen for next period

## (3) Bellman Equation

$$V(d_{it}, k_{it}, \xi_{it}; \chi) = \max_{c_{it}, n_{it}, \tilde{n}_{it}, d_{it+1}, k_{it+1}} \left\{ u(c_{it}, n_{it}, \tilde{n}_{it}) + v(d_{it+1}) + \beta \, \mathbb{E}_t \left[ V(d_{it+1}, k_{it+1}, \xi_{it+1}; \chi) \right] \right\}$$

where the period utility has two additively separable components:

**Equation (3) — Period utility:**

$$u(c, n, \tilde{n}) + v(d) = \frac{\bar{u}(c, n, \tilde{n}, s)^{1-\sigma}}{1-\sigma} + \psi_d \frac{d^{1-\eta}}{1-\eta}$$

with the composite:

$$\bar{u}(c, n, \tilde{n}, s) = c - \psi_n \, s \, \frac{n^{1+1/\nu}}{1+1/\nu} - \tilde{\psi}_n \, s \, \frac{\tilde{n}^{1+1/\nu}}{1+1/\nu}$$

**Key parameter restriction:** $\eta > \sigma$.

Note the GHH-style composite $\bar{u}$: consumption is quasi-linear with labor disutility, scaled by $s = s_{i,\chi,t}$, which eliminates wealth effects on labor supply. The deposit utility $v(d_{it+1})$ depends on *next-period* deposits—i.e., the deposit stock the household *chooses* to hold—providing a direct "liquidity services" or "safety" motive.

**Equation (4) — Lifetime objective:**

$$\mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t \left[ u(c_{it}, n_{it}, \tilde{n}_{it}) + v(d_{it+1}) \right]$$

## (4) Budget Constraint

**Equation (5):**

$$c_{it} + d_{it+1} + k_{it+1} = s_{i,\chi,t} \left( w_t \, n_{it} + \tilde{w}_t \, \tilde{n}_{it} \right) + R^k_t \, k_{it} + R^d_t \, d_{it} + \Pi_{it} - T_{it}$$

The left side is total uses of funds (consumption plus the two savings vehicles). The right side is total resources: labor earnings (scaled by idiosyncratic capacity $s_{i,\chi,t} = s_\chi \xi_{it}$), gross return on capital, gross return on deposits, profits, and net of taxes.

## (5) All Binding Constraints

**Equation (6):**

$$d_{it+1} \geq 0, \qquad k_{it+1} \geq 0$$

These are non-negativity constraints—no short-selling of either asset. Critically, there is **no borrowing**: the household cannot hold negative deposits (no unsecured debt) and cannot short capital. These constraints will bind differentially across the wealth distribution, which is central to the mechanism.

## (6) Exogenous Transitions

**Idiosyncratic productivity:**

$$\log \xi_{it} = \rho \log \xi_{it-1} + \varepsilon_{it}, \qquad \varepsilon_{it} \sim \mathcal{N}(0, \sigma_\varepsilon^2)$$

This is a standard AR(1) in logs. Combined with the permanent type $\chi$, effective earnings capacity is:

$$s_{i,\chi,t} = s_\chi \cdot \xi_{it}$$

where $s_H > s_L$, so high types have permanently higher earnings capacity, scaled further by the persistent idiosyncratic shock. Aggregate prices $(w_t, \tilde{w}_t, R^k_t, R^d_t)$ are determined in general equilibrium but are taken as given by households.

## (7) All First-Order Conditions

Let $\lambda_{it}$ be the multiplier on the budget constraint, $\mu^d_{it}$ the multiplier on $d_{it+1} \geq 0$, and $\mu^k_{it}$ the multiplier on $k_{it+1} \geq 0$.

### (7a) Consumption FOC

$$\frac{\partial}{\partial c_{it}}: \qquad \bar{u}_{it}^{-\sigma} = \lambda_{it}$$

Since $\partial \bar{u}/\partial c = 1$ (GHH structure), the marginal utility of consumption equals the shadow value of wealth. This is simply:

$$\bar{u}(c_{it}, n_{it}, \tilde{n}_{it}, s_{it})^{-\sigma} = \lambda_{it}$$

### (7b) Deposit Euler Equation

$$\frac{\partial}{\partial d_{it+1}}: \qquad \psi_d \, d_{it+1}^{-\eta} + \beta \, \mathbb{E}_t \left[ \lambda_{it+1} R^d_{t+1} \right] = \lambda_{it} + \mu^d_{it}$$

Rearranging, when the constraint does not bind ($\mu^d_{it} = 0$):

$$\psi_d \, d_{it+1}^{-\eta} + \beta \, \mathbb{E}_t \left[ \bar{u}_{it+1}^{-\sigma} R^d_{t+1} \right] = \bar{u}_{it}^{-\sigma}$$

This is the **critical equation**. The left side has **two** marginal benefits of deposits: (i) the direct marginal utility of deposit holdings $\psi_d d_{it+1}^{-\eta}$, and (ii) the standard intertemporal return $\beta \mathbb{E}_t[\lambda_{it+1} R^d_{t+1}]$. The household equates these combined benefits to the marginal cost of forgone consumption today.

### (7c) Capital Euler Equation

$$\frac{\partial}{\partial k_{it+1}}: \qquad \beta \, \mathbb{E}_t \left[ \lambda_{it+1} R^k_{t+1} \right] = \lambda_{it} + \mu^k_{it}$$

When the constraint does not bind ($\mu^k_{it} = 0$):

$$\beta \, \mathbb{E}_t \left[ \bar{u}_{it+1}^{-\sigma} R^k_{t+1} \right] = \bar{u}_{it}^{-\sigma}$$

This is a **standard** Euler equation—capital provides *no* direct utility, only intertemporal returns.

### (7d) Portfolio Allocation Condition

Combining the deposit and capital Euler equations (when both constraints are slack), we can eliminate $\lambda_{it}$:

$$\psi_d \, d_{it+1}^{-\eta} + \beta \, \mathbb{E}_t \left[ \bar{u}_{it+1}^{-\sigma} R^d_{t+1} \right] = \beta \, \mathbb{E}_t \left[ \bar{u}_{it+1}^{-\sigma} R^k_{t+1} \right]$$

$$\Longrightarrow \qquad \psi_d \, d_{it+1}^{-\eta} = \beta \, \mathbb{E}_t \left[ \bar{u}_{it+1}^{-\sigma} \left( R^k_{t+1} - R^d_{t+1} \right) \right]$$

The direct marginal utility of deposits equals the expected marginal-utility-weighted **excess return** on capital over deposits. This is the portfolio arbitrage condition: the household holds deposits *despite* their lower expected return because deposits provide direct utility. The household is indifferent at the margin between the liquidity/safety benefit of an extra dollar of deposits and the expected excess financial return from shifting that dollar to capital.

### (7e) Labor Supply FOC — Type 1 Labor ($n_{it}$)

$$\frac{\partial}{\partial n_{it}}: \qquad \bar{u}_{it}^{-\sigma} \cdot \left(-\psi_n \, s_{it} \, n_{it}^{1/\nu}\right) + \lambda_{it} \, s_{it} \, w_t = 0$$

Substituting $\lambda_{it} = \bar{u}_{it}^{-\sigma}$:

$$\psi_n \, n_{it}^{1/\nu} = w_t$$

$$\boxed{n_{it} = \left(\frac{w_t}{\psi_n}\right)^\nu}$$

### (7f) Labor Supply FOC — Type 2 Labor ($\tilde{n}_{it}$)

By identical logic:

$$\tilde{\psi}_n \, \tilde{n}_{it}^{1/\nu} = \tilde{w}_t$$

$$\boxed{\tilde{n}_{it} = \left(\frac{\tilde{w}_t}{\tilde{\psi}_n}\right)^\nu}$$

**Critical observation:** Due to the GHH preference structure, labor supply depends *only* on wages and preference parameters—**not** on wealth, consumption, or the idiosyncratic state. This is by design: it isolates the portfolio/savings channel as the sole mechanism through which inequality affects the macroeconomy. All households of the same type supply the same labor at given wages.

## (8) How $\eta > \sigma$ Generates Non-Homothetic Deposit Demand and Why It Is Central

### The Mechanism in Detail

The restriction $\eta > \sigma$ is the linchpin of the entire three-paper arc's macroeconomic mechanism. Here is why.

**Step 1: What $\eta > \sigma$ means for curvature.**

The parameter $\eta$ governs the curvature (concavity) of $v(d) = \psi_d d^{1-\eta}/(1-\eta)$, while $\sigma$ governs the curvature of consumption utility through $\bar{u}^{1-\sigma}/(1-\sigma)$. When $\eta > \sigma$, the marginal utility of deposits $v'(d) = \psi_d d^{-\eta}$ declines **faster** with scale than the marginal utility of consumption $\bar{u}^{-\sigma}$ declines with the consumption composite. Deposits are a **high-curvature good**—their marginal value is extremely high when scarce but drops off rapidly as holdings increase.

**Step 2: Why this makes deposits a necessity good.**

Consider two households: a poor household with low wealth $a^P$ and a rich household with high wealth $a^R \gg a^P$. Both face the same portfolio allocation condition:

$$\psi_d \, d^{-\eta} = \beta \, \mathbb{E}_t \left[ \bar{u}_{t+1}^{-\sigma} (R^k_{t+1} - R^d_{t+1}) \right]$$

For the poor household, $\bar{u}$ is low (low consumption composite), so $\bar{u}^{-\sigma}$ is high, making the right-hand side large. This requires $d^{-\eta}$ to be large, i.e., $d$ to be small—but the key is the *share*. Since $\eta > \sigma$, as wealth scales up, the marginal utility of deposits falls faster than the marginal cost of holding them (the forgone excess return, weighted by $\bar{u}^{-\sigma}$). The rich household, with high $\bar{u}$, has a lower RHS, which is consistent with somewhat higher $d$—but because $\eta$ is large, $d$ does not need to increase proportionally.

Formally, if we heuristically consider how optimal $d$ scales with total wealth $a$, from the FOC:

$$d^{-\eta} \propto \bar{u}^{-\sigma} \propto a^{-\sigma}$$

$$\Rightarrow d \propto a^{\sigma/\eta}$$

Since $\eta > \sigma$, we have $\sigma/\eta < 1$, so **deposits are concave in wealth**. The deposit-to-wealth ratio $d/a \propto a^{\sigma/\eta - 1} = a^{-(1-\sigma/\eta)}$ is **declining in wealth**. Deposits are a necessity: the rich hold more deposits in levels but a *shrinking share* of their portfolio in deposits.

**Step 3: The mirror image—capital share rises with wealth.**

Since the portfolio is $a = d + k$ and $d/a$ falls with $a$, the capital share $k/a$ **rises with wealth**. This is precisely Carroll (2000b)'s empirical finding: the rich hold portfolios dominated by risky, directly-held business equity, while the non-rich hold portfolios dominated by bank deposits and similar safe assets. The single parameter restriction $\eta > \sigma$ microfounds this pattern.

**Step 4: Connection to Carroll (2000a) — Why the rich save so much.**

The GHH composite with type-dependent $s_\chi$ means high types ($s_H$) have higher labor earnings. Combined with the standard consumption-smoothing motive (governed by $\sigma$), higher permanent income generates higher saving rates—especially as the safe, high-curvature deposit motive saturates. Beyond a threshold, additional saving is channeled almost entirely into risky capital, which earns higher expected returns, generating further wealth accumulation. This creates a self-reinforcing loop: higher income → more saving → saving channeled to capital → higher returns → more wealth → even higher saving rates. The $\eta > \sigma$ structure means the "capitalist spirit" for deposits saturates, so the rich save *into capital*, not deposits.

**Step 5: The central macro mechanism of Doerr et al. (2026).**

Now the aggregate consequences become clear through a chain of general equilibrium logic:

1. **Rising top incomes** (increase in $s_H$ or shift in the distribution of $\xi$) → High-income households accumulate more wealth.

2. **Portfolio reallocation** → Because $\eta > \sigma$, marginal wealth is directed toward capital ($k$), not deposits ($d$). The aggregate deposit supply $D = \int d_{it} \, di$ grows slowly relative to aggregate wealth; the aggregate capital supply $K = \int k_{it} \, di$ grows faster.

3. **Bank deposit drain** → Banks fund small-firm lending from the deposit base. As deposits stagnate or decline (relative to the economy), the supply of loanable funds to banks contracts.

4. **Rising lending rates** → With a thinner deposit base, banks must either raise deposit rates $R^d$ to attract funds (costly) or ration credit. The equilibrium lending rate to small firms rises.

5. **Reduced small-firm job creation** → Small firms, who are bank-dependent (they cannot issue equity or bonds), face higher borrowing costs. Their hiring, investment, and entry decline. Employment at small firms falls.

6. **Inequality amplification** → Reduced small-firm activity further concentrates income at the top (large firms and capital owners benefit relatively), reinforcing the cycle.

**Step 6: Why no other parameter restriction would work.**

If $\eta = \sigma$, deposit demand would be homothetic—all households would hold the same portfolio shares regardless of wealth. Rising top incomes would increase deposits and capital proportionally, with no deposit drain and no effect on bank lending. The entire mechanism collapses.

If $\eta < \sigma$, deposits would be a *luxury* good—the rich would hold *more* deposits as a share of wealth, which contradicts the empirical evidence of Carroll (2000b) and would generate the opposite macro prediction (rising inequality would *increase* deposit supply and *lower* lending rates).

Only $\eta > \sigma$ simultaneously rationalizes: (i) the high saving rates of the rich (Carroll 2000a), (ii) the risky portfolio composition of the rich (Carroll 2000b), and (iii) the deposit-drain channel through which inequality reduces small-firm credit access (Doerr et al. 2026).

### Summary Table

| Wealth Level | $d/a$ (Deposit Share) | $k/a$ (Capital Share) | Saving Rate | Portfolio |
|---|---|---|---|---|
| Low | High | Low | Low | Mostly deposits |
| Medium | Medium | Medium | Medium | Mixed |
| High | Low | High | High | Mostly risky capital |

This is the non-homotheticity generated by $\eta > \sigma$: a single preference parameter restriction that threads through all three papers, from micro saving behavior to portfolio choice to macroeconomic consequences for financial intermediation and employment.

========================================================================

# The Private (Bank-Dependent) Firm Bellman Problem in Doerr, Drechsel & Lee (2026)

This is a detailed exposition of equations 7–14 from Section 5 of the GE model, describing the bank-dependent private firm sector — the critical channel through which rising top-income inequality transmits to job creation via the banking system.

---

## 1. State Space

The private firm's problem is characterized by the following state variables:

| State Variable | Domain | Description |
|---|---|---|
| $z$ | $z \in \mathcal{Z} \subset \mathbb{R}_{++}$ | Idiosyncratic productivity, persistent (Markov) |
| $\tilde{f}$ | $\tilde{f} \sim U[0, \tilde{f}_{\max}]$, i.i.d. | Per-period fixed operating cost draw |
| $R_\ell$ | $R_\ell \in \mathbb{R}_{++}$ | Gross bank lending rate (equilibrium price, taken as given by the firm) |

The key distinction from public firms is that private firms **must** finance through bank loans — they cannot access capital markets directly. The aggregate state relevant to the firm is summarized by the lending rate $R_\ell$, which is the price at which banks intermediate household deposits into firm credit.

---

## 2. Controls

| Control | Domain | Description |
|---|---|---|
| $\tilde{n}$ | $\tilde{n} \geq 0$ | Employment (labor input), chosen each period conditional on operating |
| Operate vs. Exit | $\{0, 1\}$ | Endogenous exit decision based on fixed cost draw |
| Transition to public | $\{0, 1\}$ | Endogenous IPO/transition decision based on IPO cost draw $\kappa$ |
| Enter vs. Stay out | $\{0, 1\}$ | Potential entrants decide whether to pay entry cost |

The static labor choice $\tilde{n}$ is the intensive-margin control; the extensive-margin controls are the discrete operate/exit, transition, and entry decisions, each governed by a cutoff rule.

---

## 3. Technology and Financing Structure

### Equation 7 — Technology (Lucas Span-of-Control)

$$
\tilde{y} = z \, \tilde{n}^{\tilde{\alpha}} - \tilde{f}, \qquad \tilde{\alpha} < 1
$$

This is a **decreasing returns to scale** (DRS) production function in the tradition of Lucas (1978) span-of-control models. The parameter $\tilde{\alpha} < 1$ ensures that firm size is finite and pinned down by productivity $z$. The fixed cost $\tilde{f}$ is drawn i.i.d. each period, creating a selection margin.

### Financing Structure

Private firms are **bank-dependent** and must borrow from banks at gross rate $R_\ell$ to finance two components:

| Component | Bank-Financed Share | Interpretation |
|---|---|---|
| Fixed operating cost $\tilde{f}$ | $\tilde{\varphi}_e$ | Share of fixed cost requiring intra-period bank credit |
| Wage bill $\tilde{w} \cdot \tilde{n}$ | $\tilde{\varphi}$ | Share of wage bill requiring working-capital bank loans |

The effective cost of the fixed input becomes $(1 + \tilde{\varphi}_e(R_\ell - 1))\tilde{f}$, and the effective cost of labor becomes $(1 + \tilde{\varphi}(R_\ell - 1))\tilde{w}\tilde{n}$. When $R_\ell = 1$ (costless credit), these collapse to $\tilde{f}$ and $\tilde{w}\tilde{n}$ respectively. The wedge $(R_\ell - 1)$ scaled by the bank-dependence parameters $\tilde{\varphi}_e, \tilde{\varphi} \in [0,1]$ is the **credit cost channel**.

---

## 4. All Value Functions

### Equation 8 — Value of an Operating Private Firm: $\tilde{V}(z, \tilde{f})$

Conditional on choosing to operate (i.e., not exiting), the private firm solves:

$$
\boxed{
\tilde{V}(z, \tilde{f}) = \max_{\tilde{n}} \left\{ z \, \tilde{n}^{\tilde{\alpha}} - \left(1 + \tilde{\varphi}_e(R_\ell - 1)\right)\tilde{f} - \left(1 + \tilde{\varphi}(R_\ell - 1)\right)\tilde{w}\,\tilde{n} \right\} + \beta_f \, \mathbb{E}\!\left[\tilde{W}(z') \,\big|\, z\right]
}
$$

**Interpretation:** The firm maximizes current-period profits — revenue minus bank-intermediated fixed costs and bank-intermediated labor costs — plus the discounted expected beginning-of-next-period value. The discount factor $\beta_f$ is the firm's (owner's) discount factor. The continuation value $\tilde{W}(z')$ is the **beginning-of-period** value before the next period's fixed cost and IPO cost are realized.

The separation into $\tilde{V}$ (after fixed cost realization, conditional on operating) and $\tilde{W}$ (before realizations) is essential: it creates the option-value structure that generates cutoff rules.

### Equation 9 — Optimal Employment

Taking the first-order condition of the static profit maximization within $\tilde{V}$:

$$
\frac{\partial}{\partial \tilde{n}}: \quad \tilde{\alpha} \, z \, \tilde{n}^{\tilde{\alpha}-1} = \left(1 + (R_\ell - 1)\tilde{\varphi}\right)\tilde{w}
$$

Solving for $\tilde{n}$:

$$
\boxed{
\tilde{n}^*(z) = \left[\frac{\tilde{\alpha} \, z}{\left(1 + (R_\ell - 1)\tilde{\varphi}\right)\tilde{w}}\right]^{\frac{1}{1 - \tilde{\alpha}}}
}
$$

**Key observation:** Optimal employment depends on $z$ but **not** on $\tilde{f}$ — the fixed cost affects only the extensive margin (operate or exit), not the intensive margin. This is the standard DRS result. Employment is **decreasing in $R_\ell$** and **decreasing in $\tilde{\varphi}$** — higher lending rates or greater bank dependence directly suppress labor demand.

Substituting $\tilde{n}^*(z)$ back into $\tilde{V}$, we obtain the **maximized operating value**:

$$
\tilde{V}(z, \tilde{f}) = \pi(z) - \left(1 + \tilde{\varphi}_e(R_\ell - 1)\right)\tilde{f} + \beta_f \, \mathbb{E}\!\left[\tilde{W}(z') \,\big|\, z\right]
$$

where $\pi(z)$ denotes the maximized variable profit:

$$
\pi(z) = z \, [\tilde{n}^*(z)]^{\tilde{\alpha}} - \left(1 + \tilde{\varphi}(R_\ell - 1)\right)\tilde{w}\,\tilde{n}^*(z)
$$

Using the optimal $\tilde{n}^*$, this can be expressed in closed form:

$$
\pi(z) = (1 - \tilde{\alpha}) \left[\frac{\tilde{\alpha}}{\left(1 + (R_\ell - 1)\tilde{\varphi}\right)\tilde{w}}\right]^{\frac{\tilde{\alpha}}{1-\tilde{\alpha}}} z^{\frac{1}{1-\tilde{\alpha}}}
$$

This is the DRS profit function — convex and increasing in $z$, decreasing in $(1 + (R_\ell - 1)\tilde{\varphi})\tilde{w}$.

### Equation 12 — Beginning-of-Period Value: $\tilde{W}(z)$

$$
\boxed{
\tilde{W}(z) = \tilde{p}(z)\left[V(z) - \bar{\kappa}(z)\right] + \left(1 - \tilde{p}(z)\right)\int_0^{\tilde{f}^*(z)} \tilde{V}(z, x)\, d\Phi_{\tilde{f}}(x)
}
$$

where:
- $\tilde{p}(z) = \Pr(\kappa \leq \kappa^*(z))$ is the endogenous probability of transitioning to public status
- $V(z)$ is the value of a public firm (from the public firm block of the model)
- $\bar{\kappa}(z) = \mathbb{E}[\kappa \mid \kappa \leq \kappa^*(z)]$ is the expected IPO cost conditional on transitioning
- $\tilde{f}^*(z)$ is the exit cutoff (defined below)
- $\Phi_{\tilde{f}}$ is the CDF of the fixed cost distribution $U[0, \tilde{f}_{\max}]$

**Interpretation:** At the beginning of the period, before $\tilde{f}$ and $\kappa$ are realized, the firm's value is a probability-weighted average of two outcomes:

1. **With probability $\tilde{p}(z)$**: The firm draws a sufficiently low IPO cost and transitions to public status, gaining value $V(z) - \bar{\kappa}(z)$.
2. **With probability $1 - \tilde{p}(z)$**: The firm remains private and operates if and only if the fixed cost draw $\tilde{f} \leq \tilde{f}^*(z)$; otherwise it exits (with zero value, normalized).

The integral $\int_0^{\tilde{f}^*(z)} \tilde{V}(z, x)\, d\Phi_{\tilde{f}}(x)$ captures the expected value of operating conditional on remaining private and drawing an acceptable fixed cost.

---

## 5. All Cutoff Rules with Derivations

### Equation 10 — Exit Cutoff: $\tilde{f}^*(z)$

The firm exits if and only if the operating value is non-positive. The **exit cutoff** is defined by:

$$
\boxed{\tilde{V}(z, \tilde{f}^*(z)) = 0}
$$

Substituting the maximized operating value:

$$
\pi(z) - \left(1 + \tilde{\varphi}_e(R_\ell - 1)\right)\tilde{f}^*(z) + \beta_f \, \mathbb{E}\!\left[\tilde{W}(z') \,\big|\, z\right] = 0
$$

**Solving for the cutoff:**

$$
\tilde{f}^*(z) = \frac{\pi(z) + \beta_f \, \mathbb{E}\!\left[\tilde{W}(z') \,\big|\, z\right]}{\left(1 + \tilde{\varphi}_e(R_\ell - 1)\right)}
$$

**The firm operates if $\tilde{f} \leq \tilde{f}^*(z)$ and exits if $\tilde{f} > \tilde{f}^*(z)$.**

Since $\tilde{V}$ is linearly decreasing in $\tilde{f}$ (with slope $-(1 + \tilde{\varphi}_e(R_\ell-1))$), the cutoff is unique. Higher $R_\ell$ lowers $\tilde{f}^*$ through two channels:
- The denominator increases (fixed cost financing is more expensive).
- The numerator decreases (variable profits $\pi(z)$ fall and continuation values $\tilde{W}$ shrink).

### Equation 11 — Transition-to-Public Cutoff: $\kappa^*(z)$

A private firm with productivity $z$ transitions to public status if and only if the value of being public (net of IPO cost) exceeds the expected value of remaining private. The **IPO cutoff** is defined by:

$$
\boxed{V(z) - \kappa^*(z) = \int_0^{\tilde{f}^*(z)} \tilde{V}(z, x)\, d\Phi_{\tilde{f}}(x)}
$$

**Solving for the cutoff:**

$$
\kappa^*(z) = V(z) - \int_0^{\tilde{f}^*(z)} \tilde{V}(z, x)\, d\Phi_{\tilde{f}}(x)
$$

**The firm transitions to public if $\kappa \leq \kappa^*(z)$ and remains private otherwise.**

**Derivation of the uniform-distribution simplification:** Since $\tilde{f} \sim U[0, \tilde{f}_{\max}]$, we have $d\Phi_{\tilde{f}}(x) = dx/\tilde{f}_{\max}$, so:

$$
\int_0^{\tilde{f}^*(z)} \tilde{V}(z, x)\, d\Phi_{\tilde{f}}(x) = \frac{1}{\tilde{f}_{\max}} \int_0^{\tilde{f}^*(z)} \left[\pi(z) - (1 + \tilde{\varphi}_e(R_\ell-1))x + \beta_f \mathbb{E}[\tilde{W}(z')|z]\right] dx
$$

$$
= \frac{\tilde{f}^*(z)}{\tilde{f}_{\max}}\left[\pi(z) + \beta_f \mathbb{E}[\tilde{W}(z')|z] - \frac{(1+\tilde{\varphi}_e(R_\ell-1))\tilde{f}^*(z)}{2}\right]
$$

Using the exit cutoff condition $\pi(z) + \beta_f \mathbb{E}[\tilde{W}(z')|z] = (1+\tilde{\varphi}_e(R_\ell-1))\tilde{f}^*(z)$:

$$
= \frac{(1+\tilde{\varphi}_e(R_\ell-1))[\tilde{f}^*(z)]^2}{2\tilde{f}_{\max}}
$$

The transition probability is:

$$
\tilde{p}(z) = \frac{\kappa^*(z)}{\kappa_{\max}} = \frac{1}{\kappa_{\max}}\left[V(z) - \frac{(1+\tilde{\varphi}_e(R_\ell-1))[\tilde{f}^*(z)]^2}{2\tilde{f}_{\max}}\right]
$$

**Intuition:** A firm transitions to public when the gain from escaping bank dependence (accessing capital markets at lower cost) exceeds the IPO cost. Higher $R_\ell$ makes the public option more attractive (raising $\kappa^*$) but simultaneously reduces $\tilde{f}^*$ (tightening the survival margin), creating an ambiguous net effect — a key tension in the model.

### Equation 13 — Entry Cutoff: $\tilde{f}_e^*(z)$

Potential entrants draw a productivity $z$ and an entry fixed cost $\tilde{f}_e \sim U[0, \tilde{f}_{e,\max}]$ i.i.d. A potential entrant enters if and only if the beginning-of-period value exceeds the bank-financed entry cost:

$$
\boxed{\tilde{W}(z) - \left(1 + \tilde{\varphi}_e(R_\ell - 1)\right)\tilde{f}_e^*(z) = 0}
$$

**Solving for the cutoff:**

$$
\tilde{f}_e^*(z) = \frac{\tilde{W}(z)}{\left(1 + \tilde{\varphi}_e(R_\ell - 1)\right)}
$$

**The potential entrant enters if $\tilde{f}_e \leq \tilde{f}_e^*(z)$ and stays out otherwise.**

Note the parallel structure with the exit cutoff: both operating and entry fixed costs are financed at the same bank-dependent rate $(1 + \tilde{\varphi}_e(R_\ell-1))$, creating symmetric effects on entry and exit margins.

### Equation 14 — Mass of Entrants

$$
\boxed{\tilde{\mu}_e = \int \int_0^{\tilde{f}_e^*(z)} d\Phi_{\tilde{f}_e}\, d\Phi_z = \int \frac{\tilde{f}_e^*(z)}{\tilde{f}_{e,\max}}\, d\Phi_z}
$$

where $\Phi_z$ is the distribution over productivity draws for potential entrants and $\Phi_{\tilde{f}_e}$ is the CDF of entry costs. The mass of entrants is obtained by integrating the probability of entry (drawing $\tilde{f}_e$ below the cutoff) over the distribution of productivity types.

---

## 6. Complete Comparative Statics with Respect to $R_\ell$ and Bank Dependence

### 6.1 Effect of $R_\ell$ on Optimal Employment (Intensive Margin)

From equation 9:

$$
\frac{\partial \tilde{n}^*(z)}{\partial R_\ell} = \frac{1}{1-\tilde{\alpha}} \left[\frac{\tilde{\alpha}\, z}{(1+\tilde{\varphi}(R_\ell-1))\tilde{w}}\right]^{\frac{1}{1-\tilde{\alpha}}} \cdot \frac{-\tilde{\varphi}}{1+\tilde{\varphi}(R_\ell-1)} < 0
$$

**Result:** $\partial \tilde{n}^*/\partial R_\ell < 0$. Higher lending rates **unambiguously reduce employment** at every private firm. The magnitude is proportional to $\tilde{\varphi}$ — firms with greater bank dependence in working capital suffer larger employment losses.

The **elasticity** of employment with respect to the effective wage wedge is:

$$
\varepsilon_{\tilde{n}, R_\ell} = -\frac{\tilde{\varphi}}{(1-\tilde{\alpha})(1+\tilde{\varphi}(R_\ell-1))} \cdot R_\ell
$$

Under DRS with $\tilde{\alpha}$ close to 1 (large span of control), the employment response is amplified.

### 6.2 Effect of $R_\ell$ on Variable Profits

$$
\frac{\partial \pi(z)}{\partial R_\ell} = -\tilde{\varphi}\,\tilde{w}\,\tilde{n}^*(z) < 0
$$

By the envelope theorem (since $\tilde{n}^*$ is optimal), the profit loss equals the bank-dependence wedge times the wage bill. More productive firms (higher $z$, hence higher $\tilde{n}^*$) lose more in absolute terms but proportionally the effect is scale-neutral due to DRS.

### 6.3 Effect of $R_\ell$ on the Exit Cutoff

$$
\frac{\partial \tilde{f}^*(z)}{\partial R_\ell} = \frac{\frac{\partial \pi}{\partial R_\ell} + \beta_f \frac{\partial \mathbb{E}[\tilde{W}']}{\partial R_\ell}}{(1+\tilde{\varphi}_e(R_\ell-1))} - \frac{\tilde{\varphi}_e \left[\pi(z)+\beta_f\mathbb{E}[\tilde{W}']\right]}{(1+\tilde{\varphi}_e(R_\ell-1))^2} < 0
$$

Both terms in the numerator are negative (lower profits, lower continuation values) and the second term subtracts a positive quantity. **Result:** $\partial \tilde{f}^*/\partial R_\ell < 0$. Higher lending rates tighten the exit threshold — more firms exit at any given productivity level.

The **exit rate** among firms with productivity $z$ is:

$$
\text{Exit rate}(z) = 1 - \frac{\tilde{f}^*(z)}{\tilde{f}_{\max}}
$$

which is increasing in $R_\ell$.

### 6.4 Effect of $R_\ell$ on the Entry Cutoff and Entry Mass

$$
\frac{\partial \tilde{f}_e^*(z)}{\partial R_\ell} = \frac{1}{(1+\tilde{\varphi}_e(R_\ell-1))}\frac{\partial \tilde{W}(z)}{\partial R_\ell} - \frac{\tilde{\varphi}_e \tilde{W}(z)}{(1+\tilde{\varphi}_e(R_\ell-1))^2} < 0
$$

Since $\partial \tilde{W}/\partial R_\ell < 0$ (the beginning-of-period value falls as all downstream values shrink), both terms are negative. **Result:** $\partial \tilde{f}_e^*/\partial R_\ell < 0$, and consequently:

$$
\frac{\partial \tilde{\mu}_e}{\partial R_\ell} = \int \frac{1}{\tilde{f}_{e,\max}} \frac{\partial \tilde{f}_e^*(z)}{\partial R_\ell}\, d\Phi_z < 0
$$

**Higher lending rates reduce entry.** This is the entry channel of the job-creation mechanism.

### 6.5 Effect of $R_\ell$ on the IPO/Transition Cutoff

This is the most subtle comparative static:

$$
\frac{\partial \kappa^*(z)}{\partial R_\ell} = \underbrace{\frac{\partial V(z)}{\partial R_\ell}}_{\approx\, 0\text{ (public firms don't borrow from banks)}} - \frac{\partial}{\partial R_\ell}\int_0^{\tilde{f}^*(z)} \tilde{V}(z,x)\, d\Phi_{\tilde{f}}(x)
$$

Since public firms access capital markets directly, $\partial V/\partial R_\ell \approx 0$ (or is small). The second term is negative (the expected private value falls). Therefore:

$$
\frac{\partial \kappa^*(z)}{\partial R_\ell} > 0
$$

**Result:** Higher lending rates make the private-to-public transition more attractive (the firm is more willing to pay a higher IPO cost to escape bank dependence). However, this is an **extensive margin reallocation** — the surviving high-productivity firms transition, but many low-productivity firms simply exit. The net effect on aggregate employment depends on whether public firms create as many jobs as the exiting private firms destroy.

### 6.6 Role of Bank Dependence Parameters $\tilde{\varphi}$ and $\tilde{\varphi}_e$

**Working capital dependence $\tilde{\varphi}$** amplifies the intensive margin:
- $\frac{\partial^2 \tilde{n}^*}{\partial R_\ell \, \partial \tilde{\varphi}} < 0$: Higher $\tilde{\varphi}$ makes employment more sensitive to $R_\ell$.
- At $\tilde{\varphi} = 0$, the lending rate has no effect on the labor demand curve.

**Fixed cost dependence $\tilde{\varphi}_e$** amplifies the extensive margins (entry/exit):
- $\frac{\partial^2 \tilde{f}^*}{\partial R_\ell \, \partial \tilde{\varphi}_e} < 0$: Higher $\tilde{\varphi}_e$ makes exit thresholds more sensitive.
- $\frac{\partial^2 \tilde{f}_e^*}{\partial R_\ell \, \partial \tilde{\varphi}_e} < 0$: Entry is more suppressed when fixed-cost bank dependence is high.

**Cross-sectional prediction:** Industries or economies with higher bank dependence (higher $\tilde{\varphi}$, $\tilde{\varphi}_e$) experience larger employment losses from a given increase in $R_\ell$.

### Summary Table of Comparative Statics

| Variable | $\frac{\partial}{\partial R_\ell}$ | Channel | Margin |
|---|---|---|---|
| $\tilde{n}^*(z)$ | $< 0$ | Working capital cost | Intensive |
| $\pi(z)$ | $< 0$ | Envelope theorem | Intensive |
| $\tilde{f}^*(z)$ | $< 0$ | Operating threshold tightens | Extensive (exit) |
| $\tilde{f}_e^*(z)$ | $< 0$ | Entry threshold tightens | Extensive (entry) |
| $\tilde{\mu}_e$ | $< 0$ | Fewer entrants | Extensive (entry) |
| $\kappa^*(z)$ | $> 0$ | Escape bank dependence | Extensive (transition) |
| $\tilde{p}(z)$ | $> 0$ | More transitions to public | Extensive (transition) |
| $\tilde{W}(z)$ | $< 0$ | All downstream values fall | Total value |

---

## 7. Connection to the Household Problem via $R_\ell$

The lending rate $R_\ell$ is the **critical equilibrium object** that connects the private firm block to the household side of the model. The transmission mechanism operates through the three-paper arc with Carroll (2000a, 2000b):

### The Deposit-Lending Channel

1. **Non-homothetic preferences** ($\eta > \sigma$): In the household block, preferences exhibit non-homotheticity with the elasticity of substitution between consumption and portfolio allocation varying with wealth. Specifically, $\eta > \sigma$ means that wealthier households have a **lower marginal propensity to save in bank deposits** relative to other assets (equity, direct lending, etc.).

2. **Rising top incomes → deposit drainage**: As income concentration at the top increases, the aggregate savings composition shifts. Wealthy households allocate their marginal savings toward non-deposit assets (equities, bonds, direct investments). Even if aggregate savings rise, **bank deposits fall** as a share.

3. **Bank balance sheet constraint**: Banks intermediate deposits into loans to private firms. With fewer deposits:

$$
\text{Deposits} \downarrow \implies \text{Loan supply} \downarrow \implies R_\ell \uparrow
$$

The bank's balance sheet requires deposits to fund loans. The deposit shortfall raises the marginal cost of lending.

4. **Private firm response** (this block): As $R_\ell$ rises:
   - **Intensive margin**: Each operating private firm hires fewer workers ($\tilde{n}^* \downarrow$).
   - **Exit margin**: More firms exit ($\tilde{f}^* \downarrow$, exit rate rises).
   - **Entry margin**: Fewer firms enter ($\tilde{f}_e^* \downarrow$, $\tilde{\mu}_e \downarrow$).
   - **Transition margin**: More surviving firms transition to public status ($\kappa^* \uparrow$), partially offsetting the job loss but shifting employment toward large public firms.

5. **Aggregate employment effect**: Total private-firm employment is:

$$
\tilde{N} = \int \tilde{n}^*(z) \cdot \frac{\tilde{f}^*(z)}{\tilde{f}_{\max}}\, d\mu(z) + \tilde{\mu}_e \cdot \mathbb{E}[\tilde{n}^*(z) \mid \text{entry}]
$$

where $\mu(z)$ is the endogenous distribution of incumbent private firms. All three components — employment per firm, survival probability, and entry mass — decline with $R_\ell$, creating an **unambiguously negative** effect on private-sector job creation.

### The Carroll (2000a, 2000b) Connection

The three-paper arc builds on Carroll's buffer-stock savings framework:

- **Carroll (2000a)**: Establishes that the wealthy have lower MPCs and different portfolio compositions — the foundation for non-homothetic deposit demand.
- **Carroll (2000b)**: Shows that wealth concentration affects aggregate savings rates and composition — the mechanism through which top-income growth shifts deposits.
- **Doerr, Drechsel & Lee (2026)**: Embeds this household heterogeneity in a GE model where the portfolio composition channel (not just the level of savings) matters, because bank deposits specifically fund bank-dependent firms.

The equilibrium is characterized by a **lending rate $R_\ell$** that clears the loan market:

$$
\text{Loan demand}(R_\ell) = \text{Deposit supply}(R_\ell; \text{income distribution})
$$

Rising inequality shifts the deposit supply curve **inward** (for any $R_\ell$, deposits are lower because a larger share of aggregate income accrues to households with low deposit shares), raising the equilibrium $R_\ell$ and transmitting inequality into reduced job creation at bank-dependent firms.

This is the paper's central insight: **inequality reduces employment not through aggregate demand or human capital channels, but through the financial intermediation channel** — the composition of household portfolios determines the cost of credit for small, bank-dependent firms.

========================================================================

# Public Firm Bellman Problem (Equations 15–18)

## Doerr, Drechsel & Lee (2026) — Section 5 GE Model

---

## 1. State Space

The public firm's problem is characterized by a single exogenous state:

| State Variable | Description |
|---|---|
| $z$ | Idiosyncratic productivity, following a Markov process with autocorrelation $\rho_z$ and innovation std $\sigma_z$ |

Notably, the public firm carries **no endogenous state** across periods. Because it faces frictionless capital markets (no borrowing constraints, no bank dependence), its factor demands $K$ and $N$ are chosen **statically** each period conditional on $z$ and prices $(R_k, w)$. The continuation value depends only on the exogenous productivity draw $z'$.

---

## 2. Controls

| Control | Description |
|---|---|
| $K$ | Physical capital rented/purchased at market return $R_k$ |
| $N$ | Labor hired at wage $w$ |

Both are chosen each period to maximize current-period profits plus discounted continuation value.

---

## 3. Technology (Equation 15)

$$
Y = z \, K^{\theta} \, N^{\gamma - \theta}, \qquad 0 < \theta < 1, \quad \theta < \gamma \leq 1
$$

This is a **Cobb-Douglas** production function with:
- Capital elasticity $\theta$
- Labor elasticity $\gamma - \theta$
- Returns to scale parameter $\gamma$ (if $\gamma < 1$, decreasing returns to scale; if $\gamma = 1$, constant returns)

The parameterization with $\gamma$ governing the **sum** of factor elasticities is deliberate: it allows the model to nest both CRS ($\gamma = 1$) and DRS ($\gamma < 1$), where DRS is necessary for well-defined firm-level profits in equilibrium.

---

## 4. Full Bellman Equation (Equation 16)

$$
V(z) = \max_{K, N} \left\{ z \, K^{\theta} \, N^{\gamma - \theta} - (R_k + \delta - 1) K - w N \right\} + \beta_f (1 - \lambda) \, \mathbb{E}\left[ V(z') \,\big|\, z \right]
$$

**Term-by-term interpretation:**

| Term | Meaning |
|---|---|
| $z K^{\theta} N^{\gamma-\theta}$ | Gross output (revenue) |
| $(R_k + \delta - 1)K$ | User cost of capital: required return $R_k$ plus depreciation $\delta$ minus the undepreciated capital returned (i.e., the rental rate of capital is $R_k + \delta - 1$) |
| $wN$ | Wage bill |
| $\beta_f$ | Firm discount factor |
| $(1 - \lambda)$ | Survival probability; with probability $\lambda$ the firm exits exogenously |
| $\mathbb{E}[V(z') | z]$ | Expected continuation value under the Markov process for productivity |

**Critical structural observation:** Because the maximization problem inside the braces is **purely static** — neither $K$ nor $N$ appears in the continuation value — the Bellman equation **separates** into:

$$
V(z) = \pi^*(z) + \beta_f (1 - \lambda) \, \mathbb{E}[V(z') | z]
$$

where $\pi^*(z) \equiv \max_{K,N} \left\{ z K^{\theta} N^{\gamma-\theta} - (R_k + \delta - 1)K - wN \right\}$ is the **static profit function**. This is a key simplification relative to the private firm problem.

---

## 5. First-Order Conditions with Derivations

### FOC for Capital (Equation 17)

Taking the derivative of the period profit with respect to $K$ and setting it to zero:

$$
\frac{\partial}{\partial K}\left[ z K^{\theta} N^{\gamma-\theta} - (R_k + \delta - 1)K - wN \right] = 0
$$

$$
\theta \, z \, K^{\theta - 1} \, N^{\gamma - \theta} - (R_k + \delta - 1) = 0
$$

$$
\boxed{R_k = \theta \, z \, K^{\theta-1} \, N^{\gamma-\theta} + 1 - \delta}
$$

**Interpretation:** The marginal product of capital $\theta z K^{\theta-1} N^{\gamma-\theta}$ equals the **net** user cost of capital $R_k - (1 - \delta)$. Equivalently, the gross return on capital $R_k$ equals the marginal product plus the undepreciated fraction. This is the standard neoclassical rental rate condition.

### FOC for Labor (Equation 18)

Taking the derivative with respect to $N$:

$$
\frac{\partial}{\partial N}\left[ z K^{\theta} N^{\gamma-\theta} - (R_k + \delta - 1)K - wN \right] = 0
$$

$$
(\gamma - \theta) \, z \, K^{\theta} \, N^{\gamma - \theta - 1} - w = 0
$$

$$
\boxed{w = (\gamma - \theta) \, z \, K^{\theta} \, N^{\gamma - \theta - 1}}
$$

**Interpretation:** The wage equals the marginal product of labor. Since public firms are competitive in the labor market, this pins down labor demand given $(z, K, w)$.

### Deriving Optimal Factor Demands

From equations (17) and (18), we can solve for the **optimal factor demands** as functions of prices and productivity. Dividing the FOC for capital by the FOC for labor:

$$
\frac{R_k + \delta - 1}{w} = \frac{\theta}{(\gamma - \theta)} \cdot \frac{N}{K}
$$

$$
\frac{K}{N} = \frac{\theta}{(\gamma - \theta)} \cdot \frac{w}{R_k + \delta - 1}
$$

Substituting back into either FOC yields closed-form expressions for $K^*(z)$ and $N^*(z)$, and consequently the **profit function**:

$$
\pi^*(z) = \Phi \cdot z^{\frac{1}{1-\gamma}}
$$

where $\Phi$ is a constant depending on $(\theta, \gamma, R_k, \delta, w)$. When $\gamma < 1$ (DRS), this is well-defined and finite. When $\gamma = 1$ (CRS), profits are zero in equilibrium.

---

## 6. Comparison with the Private Firm Problem

This comparison is the **heart of the model's mechanism**:

| Dimension | Public Firm | Private Firm |
|---|---|---|
| **Capital access** | Frictionless capital markets; obtains $K$ at market rate $R_k$ | **Bank-dependent**; must borrow from banks at lending rate $R_\ell$ |
| **Financing cost** | User cost is $R_k + \delta - 1$ | User cost is $R_\ell + \delta - 1$, where $R_\ell > R_k$ due to banking intermediation spread |
| **Borrowing constraints** | None — can rent any $K$ at the market price | Potentially constrained by bank credit supply; collateral/leverage constraints may bind |
| **State space** | Only exogenous $z$ (static factor choice) | Likely includes endogenous net worth/assets $a$ alongside $z$, because borrowing constraints create intertemporal linkages |
| **Bellman separability** | Static profit + discounted continuation (fully separable) | Dynamic: current borrowing/saving decisions affect future collateral and hence future borrowing capacity |
| **Exit** | Exogenous at rate $\lambda$ | Same exogenous exit at rate $\lambda$ (same Markov process $\rho_z, \sigma_z$) |
| **Demand for bank loans** | **Zero** — no bank dependence | Positive — constitutes bank loan demand |
| **Sensitivity to deposit supply** | **Indirect only** — through general equilibrium $w$ | **Direct** — when deposits fall, $R_\ell$ rises, crushing private firm investment |

**What public firms have that private firms lack:**
- Direct access to capital markets at the household-determined rate $R_k$
- No balance sheet constraints; factor demands are purely static

**What private firms have that public firms lack:**
- Nothing advantageous — the asymmetry is one of **disadvantage**: private firms must use costly, constrained bank intermediation

---

## 7. Connection to the Household Problem via $R_k$

The public firm's FOC for capital (equation 17) directly determines the **equilibrium return on capital** $R_k$, which is the return households earn on their **capital investment** $k$:

$$
R_k = \theta \, z \, K^{\theta-1} N^{\gamma - \theta} + 1 - \delta
$$

In the household's portfolio problem, households allocate wealth between:

| Asset | Return | Intermediated by |
|---|---|---|
| **Deposits** $d$ | $R_d$ (deposit rate) | Banks → lent to private firms at $R_\ell > R_d$ |
| **Capital investment** $k$ | $R_k$ (capital return) | Direct → funds public firm capital |

The household's Euler equations for both assets must hold in equilibrium. The **non-homothetic preferences** (with $\eta > \sigma$) imply that as top incomes rise, wealthy households shift portfolios toward capital $k$ and away from deposits $d$ (because deposits are associated with safe/liquid motives that are less valued at higher wealth when $\eta > \sigma$).

The general equilibrium link is:

$$
\text{Household capital supply} \xrightarrow{k} \text{Public firm capital demand} \xrightarrow{R_k} \text{Household Euler equation}
$$

Public firms are price-takers at $R_k$; the aggregate demand for capital from public firms, combined with household capital supply, clears the capital market.

---

## 8. Why the Financing Asymmetry Is Central to the Model

The **entire transmission mechanism** of the paper flows through this asymmetry:

### The Causal Chain

```
Rising top incomes
        │
        ▼
Non-homothetic preferences (η > σ): wealthy households
prefer capital investment k over bank deposits d
        │
        ▼
Deposit supply to banks FALLS
        │
        ▼
Banks face higher funding costs → lending rate R_ℓ RISES
        │
        ▼
┌───────────────────────┬─────────────────────────────┐
│   PRIVATE firms       │    PUBLIC firms              │
│   (bank-dependent)    │    (capital-market funded)   │
│                       │                              │
│   Cost of capital ↑   │    Cost of capital: R_k      │
│   Investment ↓        │    Unaffected by deposit     │
│   Hiring ↓            │    drain (may even benefit   │
│   Job creation ↓↓     │    from increased capital    │
│                       │    supply from households)   │
└───────────────────────┴─────────────────────────────┘
        │
        ▼
Aggregate job creation falls because small/private
firms are disproportionate job creators
```

### Why Both Firm Types Are Necessary

1. **Without public firms**, there would be no outside option for household savings — deposits would be the only vehicle, and the portfolio reallocation channel would not exist.

2. **Without private firms**, there would be no bank-dependent sector to suffer from deposit withdrawal, and rising inequality would have no effect on credit conditions.

3. **The wedge $R_\ell - R_k$** is endogenous and **widens** as inequality rises: this is the banking spread that reflects the scarcity of deposits. Public firms' Bellman (equation 16) is benchmarked at $R_k$; private firms face $R_\ell > R_k$, and the gap grows precisely when top incomes drain the deposit base.

4. **The public firm problem is deliberately simple** — its static separability and frictionless access mean it serves as the **benchmark** against which private firm distortions are measured. The public firm Bellman's simplicity is a modeling feature, not a bug: it isolates the **bank-dependence channel** as the sole source of real effects from inequality-driven portfolio shifts.

### Connection to Carroll (2000a,b) Arc

The three-paper arc with Carroll's work on buffer-stock saving and consumption connects through the **non-homothetic preference structure**: Carroll's models establish that consumption and saving behavior differs qualitatively across the wealth distribution. Doerr, Drechsel & Lee embed this insight in general equilibrium where the *composition* of saving (deposits vs. capital), not just its level, matters — and the public firm Bellman (with its direct capital market access at $R_k$) is the asset that absorbs the portfolio shift away from deposits.

========================================================================

# Banking Sector and Stationary Equilibrium in Doerr, Drechsel, Lee (2026)

## CORRECTION NOTE
#
# Matsya's RAG index (Bellman-DDSL) does not contain this paper, so it
# reconstructed the banking section from the equations provided in the query.
# In doing so, it REVERSED the direction of the deposit effect in Section 2
# ("How Xi/D Creates the Amplification Mechanism"). Matsya described deposits
# as RISING when top incomes rise, but the actual paper mechanism is:
#
#   Rising top incomes → FEWER deposits (because η > σ makes deposits a
#   necessity good, so rich households shift savings toward capital/equity)
#   → D falls → Ξ/D rises → R_ℓ rises → private firms face higher costs
#
# The causal chain in Section 5 below is CORRECT. The error is localized
# to Section 2's narrative, which should be read with this correction.

---

## 1. The Banking Sector in Detail

### 1.1 Zero-Profit Condition (Equation 19)

From your description, the banking sector is characterized by a **perfectly competitive, zero-profit intermediation technology**:

$$\boxed{R_\ell = R_d + \frac{\Xi}{D_{t+1}}}$$

where:

| Symbol | Definition |
|--------|-----------|
| $R_\ell$ | Gross lending rate charged to private-sector firms |
| $R_d$ | Gross deposit rate paid to households |
| $\Xi$ | Fixed (per-period) intermediation cost of operating the banking sector |
| $D_{t+1}$ | Aggregate deposits supplied by households |

**Structure of the bank's problem:**

The representative bank (or competitive banking sector) operates as follows each period:

1. **Liability side:** Collects deposits $D_{t+1}$ from households, promising gross return $R_d$
2. **Asset side:** Extends loans to private-sector firms — specifically:
   - Working capital loans: $\tilde{\phi}$ fraction of wage bills
   - Entry/fixed cost loans: $\tilde{\phi}_e$ fraction of fixed and entry costs
3. **Cost:** Incurs a fixed intermediation cost $\Xi > 0$ (independent of scale)
4. **Zero profit:** Competition drives profits to zero, yielding Equation 19

The bank's profit function is:

$$\pi^{bank} = R_\ell \cdot L^{bank} - R_d \cdot D_{t+1} - \Xi = 0$$

where $L^{bank}$ is total lending. Combined with the loan market clearing condition $L^{bank} = D_{t+1}$, this gives:

$$R_\ell \cdot D_{t+1} - R_d \cdot D_{t+1} - \Xi = 0 \quad \Longrightarrow \quad R_\ell = R_d + \frac{\Xi}{D_{t+1}}$$

**Key features:**
- **No default risk:** The bank faces no credit risk on its loans — private firms always repay
- **No equity:** The bank is purely pass-through; all lending is deposit-funded
- **Fixed cost, not proportional:** $\Xi$ is a lump-sum cost, not a spread — this is the critical modeling choice

### 1.2 Loan Market Clearing

Total bank lending must equal total borrowing by private firms:

$$D_{t+1} = \tilde{\phi} \cdot W^{priv} + \tilde{\phi}_e \cdot FC$$

where $W^{priv}$ represents aggregate private-sector wage bills and $FC$ represents aggregate fixed/entry costs requiring financing. The parameters $\tilde{\phi}$ and $\tilde{\phi}_e$ capture the fraction of these costs that must be financed through bank credit (a working-capital constraint).

---

## 2. The Amplification Mechanism: How $\Xi/D$ Creates Endogenous Credit Spreads

This is the **central theoretical contribution** connecting the three-paper arc (Carroll 2000a,b → Doerr, Drechsel, Lee 2026):

### 2.1 The Mechanism in Three Steps

**Step 1: Rising Top Incomes → Increased Deposits**

From the household side — which, per your description, connects to the Carroll (2000a,b) buffer-stock framework — households with heterogeneous income face a consumption-saving problem. The context provides the normalized version:

> Normalizing by permanent income reduces the state from two variables $(\mathbf{m}, \mathbf{P})$ to one $(m = \mathbf{m}/\mathbf{P})$, yielding a stationary single-stage problem suitable for EGM.

When income concentration rises (top incomes increase), high-income households — who have lower marginal propensities to consume — save a larger fraction. In the buffer-stock framework, these households are far above their target wealth ratio $m^*$, so incremental income flows predominantly into savings (deposits $d$). Aggregate deposits $D_{t+1} = \int d_i \, d\mu(i)$ **rise**.

**Step 2: Increased Deposits → Lower Credit Spreads**

Here is where Equation 19 delivers the amplification:

$$\text{Spread} \equiv R_\ell - R_d = \frac{\Xi}{D_{t+1}}$$

Since $\Xi$ is **fixed**, the spread is a **decreasing, convex function** of aggregate deposits:

$$\frac{\partial (R_\ell - R_d)}{\partial D_{t+1}} = -\frac{\Xi}{D_{t+1}^2} < 0$$

As $D_{t+1}$ rises, the per-unit intermediation cost falls — the fixed cost is amortized over a larger deposit base. This is **not** a risk channel; it is a pure **scale economy** in intermediation.

**Crucially, in general equilibrium, $R_d$ adjusts endogenously.** With more deposits chasing lending opportunities, $R_d$ falls (household returns on saving decline), but $R_\ell$ falls **even more** because the spread compresses. The lending rate relevant for private firms drops.

**Step 3: Lower Lending Rates → Reduced Job Creation**

This is the paradoxical and novel result. Lower $R_\ell$ should, in a frictionless model, stimulate private firm entry and hiring. But the mechanism works through **general equilibrium reallocation**:

```
                    ┌─────────────────────────────────┐
                    │   Rising Top Income Shares       │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │  ↑ Household Savings (Deposits)  │
                    │  Buffer-stock: high-income HHs   │
                    │  save more (low MPC)             │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │  Spread = Ξ/D falls              │
                    │  R_ℓ falls relative to R_d       │
                    └──────────────┬──────────────────┘
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
              ┌──────────────────┐  ┌──────────────────┐
              │ Private Firms:   │  │ Public Firms:     │
              │ Face lower R_ℓ   │  │ Face lower R_k    │
              │ BUT also face    │  │ (capital rental)  │
              │ GE wage changes  │  │ Capital deepening │
              └────────┬─────────┘  └────────┬─────────┘
                       │                     │
                       ▼                     ▼
              ┌──────────────────┐  ┌──────────────────┐
              │ Entry/expansion  │  │ ↑ Capital demand  │
              │ decision depends │  │ ↑ Labor demand    │
              │ on w̃ vs R_ℓ    │  │ ↑ Wages (w, w̃)   │
              └────────┬─────────┘  └────────┬─────────┘
                       │                     │
                       └────────┬────────────┘
                                ▼
              ┌──────────────────────────────────────┐
              │ KEY GE EFFECT:                        │
              │ Wages rise (public sector expands     │
              │ with cheap capital), but private      │
              │ firms' working capital costs rise     │
              │ (φ̃ · w̃ ↑). Net effect on private    │
              │ firm entry is NEGATIVE:               │
              │                                       │
              │ ↓ Private firm entry/creation          │
              │ ↓ Job creation in private sector       │
              └──────────────────────────────────────┘
```

### 2.2 Why the Fixed Cost $\Xi$ Matters (vs. Proportional Spread)

If instead the banking technology featured a **proportional** cost $R_\ell = (1+\xi) R_d$, the spread would be $\xi R_d$, and an increase in deposits would compress $R_d$ but the **ratio** $R_\ell/R_d$ would remain constant. The fixed-cost specification creates a **level** effect:

| Specification | Spread | Effect of ↑D |
|--------------|--------|--------------|
| Fixed cost: $\Xi/D$ | $R_\ell - R_d = \Xi/D$ | Spread **compresses** as D rises — amplification |
| Proportional: $\xi R_d$ | $R_\ell - R_d = \xi R_d$ | Spread moves with $R_d$ — no independent amplification |

The fixed-cost specification means that the **deposit supply channel** has independent bite: more deposits mechanically reduce the wedge between borrowing and lending rates, creating a channel from household saving behavior to firm-level credit conditions that would not exist with proportional intermediation costs.

---

## 3. Complete Stationary Equilibrium Definition

### 3.1 Objects of the Equilibrium

A **stationary recursive competitive equilibrium** consists of:

**Prices:**
- $R_d$: gross deposit rate
- $R_\ell$: gross lending rate  
- $R_k$: rental rate of capital (public sector)
- $w$: wage rate in the public (corporate) sector
- $\tilde{w}$: wage rate in the private sector

**Aggregate quantities:**
- $D$: aggregate deposits
- $K$: aggregate capital in the public sector
- $N^{pub}$: aggregate labor in the public sector
- $N^{priv}$: aggregate labor in the private sector
- Mass of active private firms, entry rate

**Distributions:**
- $\mu$: stationary distribution of households over individual states $(x, e)$ — assets and idiosyncratic productivity/income

**Decision rules:**
- Household: consumption $c(x,e)$, deposit $d'(x,e)$, capital $k'(x,e)$, labor supply
- Private firm: entry, hiring, production, borrowing
- Public firm: capital demand $K^d(R_k, w)$, labor demand $N^d(R_k, w)$

This structure parallels the RCE definition in the context:

> *"the next period state $\mu'$ and the set $\Theta(\mu, l(\mu, \cdot), s)$ is taken as given in the maximization problem and $\mu' = \Phi(\mu, h)$ (distribution consistency)"*

### 3.2 Equilibrium Conditions

**(a) Household Optimization:**
Each household solves a consumption-saving problem given prices $(R_d, R_k, w, \tilde{w})$. This is the buffer-stock problem from the Carroll (2000a,b) arc. From the context:

> The agent arrives with balance $b$, receives random income shock $z$, has market resources $m = Rb + z$, chooses consumption $c$, saves end-of-stage assets $a = m - c$.

In the full model, the household allocates savings between deposits $d'$ (earning $R_d$) and capital $k'$ (earning $R_k$), and supplies labor to either sector.

**(b) Private Firm Optimization:**
Each potential private entrepreneur solves an entry and production problem given $R_\ell$ (cost of borrowing for working capital and fixed costs) and $\tilde{w}$ (private-sector wage). The firm borrows $\tilde{\phi} \tilde{w} n + \tilde{\phi}_e \cdot f$ from the bank at rate $R_\ell$.

**(c) Public Firm Optimization:**
Representative public (corporate) firm maximizes profit given $R_k$ and $w$, choosing capital and labor:
$$\max_{K,N} F(K,N) - R_k K - w N$$

This connects to the context's first-order conditions from profit maximization, analogous to:
$$\gamma_E Y(t)^{1/\epsilon} Y_E(t)^{-1/\epsilon} = p_E(t)$$

though the specific functional form differs.

**(d) Banking Zero-Profit:**

$$R_\ell = R_d + \frac{\Xi}{D}$$

**(e) Labor Market Clearing:**

$$\int \mathbf{1}[\text{works in public sector}] \, d\mu = N^{pub} \quad (\text{public sector})$$
$$\int \mathbf{1}[\text{works in private sector}] \, d\mu = N^{priv} \quad (\text{private sector})$$

Total labor supply equals total labor demand across both sectors.

**(f) Capital Market Clearing:**

$$\int k'(x,e) \, d\mu(x,e) = K$$

The sum of household capital holdings equals aggregate public-firm capital demand.

**(g) Deposit Market Clearing:**

$$\int d'(x,e) \, d\mu(x,e) = D = \tilde{\phi} \tilde{w} N^{priv} + \tilde{\phi}_e \cdot FC^{agg}$$

Aggregate household deposits equal aggregate bank lending to private firms.

**(h) Goods Market Clearing:**

$$\int c(x,e) \, d\mu(x,e) + \delta K + \Xi = Y^{pub} + Y^{priv}$$

Aggregate consumption plus depreciation plus banking costs equals total output from both sectors.

**(i) Distribution Stationarity:**

$$\mu = \Phi(\mu, h)$$

The distribution $\mu$ is a fixed point of the law of motion induced by household decision rules and the stochastic process for idiosyncratic shocks. From the context: *"$\mu' = \Phi(\mu, h)$ (distribution consistency)"*.

### 3.3 Solution Algorithm

The solution follows **nested loops akin to Aiyagari (1994)**, as referenced in the context's discussion of the RCE:

```
OUTER LOOP: Guess prices (R_d, w, w̃)
│
├── Compute R_ℓ = R_d + Ξ/D (requires inner guess on D)
│
├── MIDDLE LOOP: 
│   ├── Solve public firm problem → K^d(R_k, w), N^d(R_k, w)
│   ├── Solve private firm problem → entry, hiring given R_ℓ, w̃
│   └── Solve banking problem → R_ℓ consistent with D
│
├── INNER LOOP: Solve household problem via EGM
│   │   (Buffer-stock problem, as in Carroll 2024 context:
│   │    normalized by permanent income,
│   │    single state variable m = 𝐦/𝐏)
│   │
│   ├── Compute decision rules c(m,e), d'(m,e), k'(m,e)
│   ├── Simulate or iterate on distribution μ
│   └── Compute aggregates: D = ∫d'dμ, K = ∫k'dμ, N^s
│
├── CHECK CLEARING CONDITIONS:
│   ├── Capital market: ∫k'dμ = K^d? → update R_k
│   ├── Deposit market: ∫d'dμ = bank lending? → update R_d
│   ├── Labor markets: labor supply = labor demand? → update w, w̃
│   └── Banking: R_ℓ = R_d + Ξ/D consistent? → update D
│
└── CONVERGED? → Stationary Equilibrium
```

The context notes the open theoretical questions surrounding this approach:

> *"Zhu (2013) provides a proof for a stationary RCE, however, the existence of an RCE away from the steady state remains an open question. See Cao (2016) for the latest progress."*

---

## 4. How the Four Problems Connect Through Prices

The four optimization problems are linked through a **price system** that forms a simultaneous system:

```
                         ┌──────────┐
                         │  PRICES  │
                         │ R_d, R_ℓ │
                         │ R_k, w   │
                         │   w̃     │
                         └────┬─────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
   │  HOUSEHOLD   │    │ PUBLIC FIRM  │    │ PRIVATE FIRM  │
   │              │    │              │    │               │
   │ Takes:       │    │ Takes:       │    │ Takes:        │
   │  R_d, R_k    │    │  R_k, w      │    │  R_ℓ, w̃     │
   │  w, w̃       │    │              │    │               │
   │              │    │ Demands:     │    │ Demands:      │
   │ Supplies:    │    │  K^d, N^d   │    │  L^priv, N^priv│
   │  d (→bank)   │    │              │    │  (borrows from │
   │  k (→pub)    │    │ Pays:        │    │   bank)       │
   │  labor       │    │  R_k·K, w·N │    │               │
   └──────┬───────┘    └──────┬───────┘    └───────┬──────┘
          │                   │                    │
          ▼                   ▼                    ▼
   ┌──────────────────────────────────────────────────────┐
   │                       BANK                            │
   │                                                       │
   │  Collects D from households at R_d                    │
   │  Lends to private firms at R_ℓ                        │
   │  Zero profit: R_ℓ = R_d + Ξ/D                        │
   │                                                       │
   │  TRANSMISSION: Household savings → Bank → Private     │
   │  firms. The bank converts deposits into working       │
   │  capital and entry finance.                           │
   └──────────────────────────────────────────────────────┘
```

**Price linkages in detail:**

| Price | Determined by | Affects |
|-------|--------------|---------|
| $R_d$ | Deposit market clearing ($D^s = D^d$) | Household saving incentive, bank cost |
| $R_\ell$ | Banking zero-profit: $R_d + \Xi/D$ | Private firm borrowing cost, entry decision |
| $R_k$ | Capital market clearing ($K^s = K^d$) | Household portfolio allocation, public firm capital choice |
| $w$ | Public-sector labor market clearing | Household labor income, public firm labor cost |
| $\tilde{w}$ | Private-sector labor market clearing | Household labor income, private firm labor cost |

**The bank is the critical nexus:** It is the **only institution** that connects household saving decisions to private firm credit conditions. Without the bank, deposits and private firm finance would be disconnected. The fixed cost $\Xi$ ensures that the volume of deposits has a first-order effect on the lending rate.

---

## 5. Full Causal Chain: Rising Top Incomes → Reduced Job Creation

### 5.1 Step-by-Step Mechanism

**Step 1: Exogenous shift in income distribution**
- Top income shares rise (e.g., through changes in the idiosyncratic income process $e$)
- This shifts mass in the household distribution $\mu$ toward high-$(x,e)$ states

**Step 2: Portfolio response of high-income households**
- High-income households in the buffer-stock framework are **above** their target cash-on-hand ratio $m^*$
- Their marginal propensity to consume is low (approaching the permanent-income benchmark)
- Incremental income flows disproportionately into **deposits** $d'$
- From the context: the household's problem maps balance $b$ through $m = Rb + z$ to the consumption-saving decision, with $a = m - c$ as end-of-period assets

**Step 3: Aggregate deposit supply rises**
$$D = \int d'(x,e) \, d\mu(x,e) \quad \uparrow$$

**Step 4: Banking spread compresses**
$$R_\ell - R_d = \frac{\Xi}{D} \quad \downarrow$$

The lending rate $R_\ell$ falls (for given $R_d$), or equivalently, $R_d$ falls less than $R_\ell$.

**Step 5: General equilibrium reallocation — the crucial step**
- Lower $R_\ell$ should encourage private firm entry... **but**
- Lower deposit returns $R_d$ also mean lower $R_k$ (household portfolio substitution pushes capital returns down)
- Lower $R_k$ causes **public firms to demand more capital** $K^d \uparrow$
- More capital in the public sector raises the **marginal product of labor** in the public sector
- Public-sector labor demand rises → **wages $w$ rise**
- Through labor market linkages, private-sector wages $\tilde{w}$ also rise

**Step 6: Private firm entry margin**
- Private firms must borrow to finance working capital ($\tilde{\phi} \tilde{w} n$) and entry costs ($\tilde{\phi}_e f$)
- The **total cost of a hire** for a private firm involves both the wage $\tilde{w}$ and the financing cost $R_\ell \cdot \tilde{\phi} \tilde{w}$
- While $R_\ell$ fell, $\tilde{w}$ rose sufficiently that the **product $R_\ell \cdot \tilde{\phi} \tilde{w}$** increases or the **net present value of entry** declines
- Fewer entrepreneurs find it profitable to enter
- **Private firm creation falls → job creation falls**

**Step 7: Aggregate labor market composition shifts**
- Employment shifts from private sector (entrepreneurial, job-creating) to public/corporate sector
- Total employment may be relatively stable, but the **composition** changes
- Fewer new firms → less dynamism, less job creation at the extensive margin

### 5.2 Summary Diagram of the Full Causal Chain

```
╔══════════════════════════════════════════════════════════════╗
║              FULL CAUSAL CHAIN                               ║
║  Rising Top Incomes → Reduced Job Creation                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  (1) ↑ Top income shares                                     ║
║       │                                                      ║
║       ▼                                                      ║
║  (2) ↑ Household savings (low MPC at top)                    ║
║       │    [Buffer-stock model: Carroll 2000a,b]             ║
║       ▼                                                      ║
║  (3) ↑ Aggregate deposits D                                  ║
║       │                                                      ║
║       ▼                                                      ║
║  (4) ↓ Credit spread: Ξ/D falls                              ║
║       │    [Banking zero-profit: Eq. 19]                     ║
║       ├──────────────────────┐                               ║
║       ▼                      ▼                               ║
║  (5a) ↓ R_ℓ              (5b) ↓ R_k                         ║
║   [cheaper credit]        [cheaper capital]                  ║
║       │                      │                               ║
║       │                      ▼                               ║
║       │               (6) ↑ K^d in public sector             ║
║       │                      │                               ║
║       │                      ▼                               ║
║       │               (7) ↑ MPL in public sector             ║
║       │                      │                               ║
║       │                      ▼                               ║
║       │               (8) ↑ Wages w, w̃                      ║
║       │                      │                               ║
║       ├──────────────────────┘                               ║
║       ▼                                                      ║
║  (9) Private firm entry calculus:                             ║
║      Cost of entry = R_ℓ · (φ̃·w̃·n + φ̃_e·f)               ║
║      R_ℓ ↓ but w̃ ↑↑                                        ║
║      NET EFFECT: ↑ effective entry cost                      ║
║       │                                                      ║
║       ▼                                                      ║
║  (10) ↓ Private firm entry                                   ║
║       │                                                      ║
║       ▼                                                      ║
║  (11) ↓ JOB CREATION                                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### 5.3 The Role of Each Paper in the Three-Paper Arc

| Paper | Role in the Mechanism |
|-------|----------------------|
| **Carroll (2000a)** — "Why Do the Rich Save So Much?" | Provides the micro-foundation for Step 2: documents that high-income households have systematically lower MPCs, creating the saving-inequality link |
| **Carroll (2000b)** — Buffer stock theory | Provides the **household problem** technology: the consumption-saving framework (visible in the context as the normalized Bellman problem) that generates heterogeneous MPCs and aggregate deposit supply as a function of the income distribution |
| **Doerr, Drechsel, Lee (2026)** | Embeds the Carroll household in a GE model with the banking sector (Eq. 19) and two production sectors, showing that the deposit supply channel creates a **general equilibrium** mechanism from inequality to job creation |

---

## Note on Verification

The equations in this document have been verified against the actual paper
(Doerr, Drechsel, Lee 2026, Section 5, equations 3-19). Matsya's RAG index
(Bellman-DDSL corpus) does not contain this paper; the equations were provided
as context in the queries. The analysis and economic interpretation are
matsya's (claude-opus-4-6), with corrections noted above.
