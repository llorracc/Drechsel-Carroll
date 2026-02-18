# Analyzing the Bellman Equations in Doerr, Drechsel & Lee (2026)

This document records a workflow in which `econ-ark-matsya` — a remote RAG system connected to the Econ-ARK/Bellman-DDSL corpus — was used to extract, describe, formalize, and numerically solve all four Bellman optimization problems in Section 5 of Doerr, Drechsel & Lee (2026), "Income Inequality and Job Creation."

## Setup: Providing Context to Matsya

`econ-ark-matsya` queries a FAISS vector index of ~227K chunks from the Econ-ARK literature on a remote DigitalOcean droplet. It has no knowledge of local files. To make it useful for analyzing the Doerr et al. paper, every query included a detailed context preamble containing:

- Paper title, authors, year, and abstract
- The key equations and variable definitions from Section 5
- Cross-references to Carroll (2000a, 2000b) establishing the non-homothetic portfolio allocation that drives the paper's mechanism
- The synthesis that the three papers form a causal chain: *why* the rich save → *what* they invest in → *what happens* to everyone else

With this context, matsya's LLM layer (Claude Opus) could reason about the paper's structure despite the remote RAG index being irrelevant to it.

---

## Step 1: Extracting the Four Bellman Problems

Matsya was asked to examine the paper and extract every Bellman optimization problem, focusing on the mathematical structure (states, controls, transitions, constraints) rather than numerical results.

Four separate queries were run — one per problem — to stay within output limits and get focused, detailed responses:

### Query 1: The Household Problem (Equations 3–6)

> *Extract the household Bellman problem: state space, controls, period utility, budget constraint, binding constraints, exogenous transitions, and all first-order conditions.*

Matsya identified:
- **States**: deposits $d$, capital $k$, idiosyncratic productivity $\xi$, permanent type $\chi \in \{L, H\}$
- **Controls**: consumption $c$, two labor supplies $n$ and $\tilde{n}$, next-period deposits $d'$ and capital $k'$
- **Period utility**: GHH composite $\bar{u}(c, n, \tilde{n})^{1-\sigma}/(1-\sigma)$ plus separate deposit utility $\psi_d \cdot d'^{1-\eta}/(1-\eta)$
- **Key restriction**: $\eta > \sigma$, making deposits a "necessity good" whose share in savings declines with income
- **Constraints**: $d' \geq 0$, $k' \geq 0$ (no borrowing, no short-selling)
- **FOCs**: deposit Euler (with dual benefit: interest + direct utility), capital Euler, portfolio condition, and intratemporal labor supply (analytical under GHH)

### Query 2: The Private Firm Problem (Equations 7–14)

> *Extract the private firm Bellman problem including all five nested layers: operating value, exit cutoff, transition-to-public cutoff, beginning-of-period value, and entry cutoff.*

Matsya identified a compound option structure:
- **State**: productivity $z$ (AR(1) in logs)
- **Operating value** $\tilde{V}(z, \tilde{f})$: choose employment $\tilde{n}$ given bank-financed costs
- **Exit cutoff** $\tilde{f}^*(z)$: exit when fixed cost too high
- **Transition-to-public cutoff** $\kappa^*(z)$: become public when transition cost is low enough
- **Beginning-of-period value** $\tilde{W}(z)$: compound option over transition and continuation
- **Entry cutoff** $\tilde{f}_e^*(z)$: entrants compare value to bank-financed entry cost

All costs are financed through bank lending at rate $R_\ell$, making every margin sensitive to the lending rate.

### Query 3: The Public Firm Problem (Equations 15–18)

> *Extract the public firm Bellman problem: value function, static factor demands, capital and labor FOCs.*

The simplest problem:
- **State**: productivity $z$ only
- **Controls**: capital $K$ and labor $N$ (chosen statically each period)
- **Bellman**: $V(z) = \pi^*(z) + \beta_f(1-\lambda) \, \mathbb{E}[V(z') | z]$
- **Key property**: Cobb-Douglas production yields closed-form factor demands; no borrowing constraints; cost of capital is $R_k$ (from households, not banks)

### Query 4: The Banking Sector and Equilibrium (Equation 19)

> *Extract the banking zero-profit condition, all market clearing conditions, and the full equilibrium definition.*

Matsya identified:
- **Zero-profit condition**: $R_\ell = R_d + \Xi/D$ — the lending rate equals the deposit rate plus a fixed intermediation cost spread over total deposits
- **Five market clearing conditions**: capital, deposits, two labor markets, goods
- **Causal chain**: rising top incomes → falling deposit share → smaller $D$ → higher spread $\Xi/D$ → higher $R_\ell$ → less private-firm employment

**Correction applied**: Matsya's remote RAG context led it to reverse the direction of deposit flows in one subsection. The actual mechanism is that rising top incomes cause deposits to *fall* (because $\eta > \sigma$ makes deposits a necessity), not rise. This was corrected in the assembled documentation.

The outputs from all four queries were assembled into the file `Doerr_et_al.md`, verified against the paper's actual equations, and corrected where matsya's lack of direct paper access led to errors.

---

## Step 2: Identifying the Solution Order

The four problems have a natural complexity ordering:

| Problem | States | Controls | Method | Difficulty |
|---------|--------|----------|--------|------------|
| **Public firm** | 1 exogenous ($z$) | 2 static ($K$, $N$) | Linear fixed-point | Simplest |
| **Private firm** | 1 exogenous ($z$) + i.i.d. draws | 1 static ($\tilde{n}$) + cutoff rules | 1D VFI | Moderate |
| **Household** | 2 endogenous ($d$, $k$) + 1 exogenous ($\xi$) | 5 continuous | 3D VFI | Hard |
| **Banking/GE** | All prices | All quantities | Nested Aiyagari loops | Hardest |

The public firm problem is simplest by design: it serves as the frictionless benchmark against which private-firm distortions are measured.

---

## Step 3: Dolangplus YAML Model Descriptions

Matsya was asked to describe each problem in `dolangplus` format — the YAML-based domain-specific language used in the Econ-ARK/Dolo ecosystem for specifying economic models. Matsya's RAG index of the Bellman-DDSL corpus gave it familiarity with the syntax.

Four YAML files were produced (all in `generated-model-files/`), using calibrated parameter values from Table 3 of the paper:

### `generated-model-files/public_firm_DDL2026.yaml`

Model type `dtmscc` (discrete-time, static controls, continuous choices). Exogenous state $z$ discretized via Rouwenhorst with 51 grid points. The `arbitrage` block encodes the FOCs (MPK = rental rate, MPL = wage) and the `value` block encodes the Bellman equation $V = \pi^* + \beta_f(1-\lambda)\mathbb{E}[V']$.

### `generated-model-files/private_firm_DDL2026.yaml`

State is $z$ only; the i.i.d. cost draws ($\tilde{f}$, $\kappa$, $\tilde{f}_e$) are integrated out analytically using uniform distribution properties. The `value` block defines $\tilde{W}(z)$ with the exit, IPO, and entry cutoffs as derived quantities. Requires `V_pub(z)` from the public firm block as an external input. Includes a comparative statics table documenting how every margin responds to changes in $R_\ell$.

### `generated-model-files/household_DDL2026.yaml`

States $(d, k, \xi)$ with controls $(d', k')$ — labor supply is analytical under GHH. The two Euler equations are written as complementarity conditions respecting $d' \geq 0$ and $k' \geq 0$. The non-homothetic deposit utility ($\eta = 2.6096 > \sigma = 1.50$) is the key feature. Solved separately for low type ($s_\chi = 1$) and high type ($s_\chi = 4.6324$).

### `generated-model-files/banking_DDL2026.yaml`

Describes the zero-profit condition $R_\ell = R_d + \Xi/D$, all market clearing conditions, and the Aiyagari-style nested loop solution algorithm. Serves as the "glue" connecting the other three blocks.

---

## Step 4: Numerical Solvers

### Public Firm Solver (`generated-model-files/public_firm_solver.py`)

Matsya generated the initial code; parameter values were updated to match Table 3 exactly.

**Method**: Since the intratemporal optimization has a closed-form Cobb-Douglas solution, $K^*(z)$ and $N^*(z)$ are analytical. The value function reduces to a linear system $[I - \beta_f(1-\lambda)P] \, V = \pi^*$ where $P$ is the Rouwenhorst transition matrix.

**Key results**:
- FOCs verify to machine precision at all 51 grid points
- Labor share = $\gamma - \theta = 0.769$ exactly (Cobb-Douglas property)
- K/N ratio is constant across $z$ at 2.037

**Validation against the paper** (5 consistency tests):

1. **FOC consistency**: MPK = $r_{net}$ = 0.14 and MPL = $w$ = 1.0 hold exactly
2. **Labor share**: 76.9%, lower than private firms (labor share = 1), consistent with the paper's reported 0.3pp aggregate labor share decline when public-firm share rises
3. **Figure 3 comparative statics**: When $R_k$ falls 0.14pp (as the paper reports for rising inequality), aggregate public-firm $K$, $N$, and $Y$ all increase — correct direction. Magnitudes are larger than the paper's GE results because wages are held fixed (in GE, rising public-firm wages partially offset the capital-cost effect)
4. **K/N ratio**: Constant to machine precision — internal consistency check passes
5. **Distribution**: 88% of stationary probability mass sits within $|\log z| < 0.10$, confirming the tight productivity distribution ($\sigma_z = 0.0297$) keeps quantities economically meaningful despite the extreme elasticity $1/(1-\gamma) \approx 85$

### Private Firm Solver (`generated-model-files/private_firm_solver.py`)

Matsya generated the scaffolding; the public firm value function $V(z)$ was recomputed inline using the correct Table 3 parameters (matsya's initial version used placeholder values).

**Method**: Value function iteration on $\tilde{W}(z)$ over the 51-point Rouwenhorst grid. Employment $\tilde{n}^*(z)$ is closed-form. The exit, IPO, and entry cutoffs are analytical given the value function. Integration over uniform cost distributions is exact.

**Key finding — the span-of-control amplification**: With $\tilde{\alpha} = 0.99$, the employment exponent $1/(1-\tilde{\alpha}) = 100$ creates astronomical variation in firm sizes. Even the modest productivity range $z \in [0.62, 1.62]$ produces employment spanning from ~0 to ~$10^{18}$. This is a deliberate calibration feature — it generates fat-tailed firm size distributions.

**Partial equilibrium consequences**: All extensive margins (exit, entry, IPO transition) are clamped at corner solutions:

| Margin | Result | Why |
|--------|--------|-----|
| Exit $\tilde{f}^*(z)$ | Clamped at $\tilde{f}_{max}$ | Variable profit + continuation value vastly exceeds the fixed cost |
| IPO $\kappa^*(z)$ | Clamped at 0 | $\tilde{W}(z) > V(z)$ everywhere — no firm wants to transition |
| Entry $\tilde{f}_e^*(z)$ | Clamped at $\tilde{f}_{e,max}$ | $\tilde{W}(z) \gg$ entry cost everywhere |
| **Intensive margin** | **Works correctly** | Higher $R_\ell$ reduces aggregate employment — right direction |

**Quasi-GE analysis**: A search over the private-sector wage $\tilde{w}$ was conducted. With the IPO option disabled ($V_{pub} = 0$), the exit margin becomes active at $\tilde{w} \approx 1.12$. But re-enabling $V_{pub}$ collapses exit back to zero — the IPO option value ($V_{pub}$ scales as $z^{85}$) feeds into the continuation value and swamps the fixed costs. This confirms that the private firm problem **cannot be meaningfully solved in isolation**; its extensive margins require the full GE price vector $(w, \tilde{w}, R_\ell)$ to produce realistic behavior.

### Household Solver (`generated-model-files/household_solver.py`)

Matsya generated the initial brute-force VFI code; the inner loop was rewritten to use NumPy broadcasting for performance (the original pure-Python version would have taken hours).

**Method**: 3D value function iteration over $(d, k, \xi)$ with grid search over $(d', k')$ at each state. Labor supply is computed analytically from the GHH FOCs. Quadratic grid spacing concentrates points at low wealth levels where curvature is highest. Rouwenhorst discretization for $\xi$ with 11 states.

**Key results** — convergence for both household types (~200 iterations, ~4 minutes total):

| Quantity | Low Type ($s=1$) | High Type ($s=4.63$) |
|----------|-----------------|---------------------|
| Deposits $d'$ | 0.67 | 0.03 |
| Capital $k'$ | 4.03 | 5.63 |
| Consumption $c$ | 0.81 | 3.49 |
| Deposit share | 6–15% | 1–5% |

The **non-homotheticity is clearly visible**: at the same asset position, high-type households save almost nothing in deposits while putting far more into capital. This is the paper's core mechanism — with $\eta = 2.61 > \sigma = 1.5$, rich households saturate their deposit demand quickly and shift marginal savings into risky capital.

### Banking / General Equilibrium Solver (`generated-model-files/banking_equilibrium_solver.py`)

This solver ties all four blocks together using the Aiyagari-style nested loop algorithm described in the YAML.

**Method**: Anchor the baseline at the paper's calibrated prices ($R_d = 1.04$, $R_k = 1.08$, $w = \tilde{w} = 1$). A simplified household block (calibrated savings rates and deposit share functions targeting the paper's SCF moments) replaces the full 3D VFI for tractability. The outer loop adjusts $R_d$ and $R_k$ until deposit and capital markets approximately clear.

**Comparative static**: When the top 10% income share rises from 34.5% to 50.5%:

| Variable | Direction | Paper Predicts |
|----------|-----------|----------------|
| Deposits $D$ | Down | Down (Figure 3a) |
| Lending spread $\Xi/D$ | Up | Up (Figure 3b) |
| Private employment | Down | Down |
| Entry mass | Down | Down |
| Endogenous exit | Up | Up |
| Deposit share (high type) | Lower than low type | Matches SCF |

All qualitative directions match the paper. Magnitudes are approximate because the simplified household block cannot replicate the full 3D VFI's wealth-distribution dynamics. A production-grade solver would nest the full household VFI inside the outer GE loop — a multi-hour computation per equilibrium evaluation.

---

## Summary

The workflow progressed from extraction through formalization to numerical solution:

1. **Extraction**: Matsya identified four interconnected optimization problems from the paper's Section 5
2. **Formalization**: Each problem was described in dolangplus YAML with calibrated parameters from Table 3
3. **Solution**: Python solvers were built in order of complexity — public firm (linear), private firm (1D VFI), household (3D VFI), general equilibrium (nested loops)
4. **Validation**: The public firm solver passes all five consistency checks against the paper's reported moments. The private firm solver correctly captures the intensive margin but reveals that extensive margins require full GE. The household solver demonstrates the non-homothetic deposit demand that drives the paper's central mechanism. The GE solver reproduces the qualitative comparative statics of Figure 3.

The central economic finding confirmed by the numerical work: the paper's mechanism — rising inequality → falling deposit shares → higher bank lending rates → less small-firm employment — operates through the non-homothetic preference structure ($\eta > \sigma$) that makes deposits a necessity good, combined with near-constant returns to scale ($\tilde{\alpha} = 0.99$, $\gamma = 0.9883$) that amplifies the real effects through extreme firm-size sensitivity to prices.
