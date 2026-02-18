# How the Carroll and Doerr et al. Papers Relate

## The Three Papers

This repository contains three economics papers that form a coherent intellectual arc, with Christopher D. Carroll as the connective thread:

1. **"Why Do the Rich Save So Much?"** (Carroll, 2000) — in `Carroll-Why-Do-The-Rich-Save-So-Much/`
2. **"Portfolios of the Rich"** (Carroll, 2000) — in `Carroll-Portfolios-of-the-Rich/`
3. **"Income Inequality and Job Creation"** (Doerr, Drechsel & Lee, 2026) — in `Doerr-et-al/`

## The Causal Chain: Paper 1 → Paper 2 → Paper 3

### Paper 1: Why They Save

Carroll (2000a) establishes the foundational puzzle and theoretical innovation. Standard Life-Cycle models predict the rich should dissave in retirement, and Dynastic models cannot match the magnitude of observed wealth accumulation at the top. Carroll's solution — **Capitalist Spirit utility** where wealth enters directly as a luxury good via modified Stone-Geary preferences $v(w) = (w + \gamma)^{1-\alpha}/(1-\alpha)$ with $\alpha < \rho$ — generates saving rates that rise with wealth. This is the *why* of rich households' behavior: wealth is not merely a store of future consumption but yields direct utility, and the marginal utility of wealth declines more slowly than the marginal utility of consumption ($\alpha < \rho$).

### Paper 2: What They Invest In

Carroll (2000b) takes the same theoretical apparatus and discovers an **unanticipated corollary**: the condition $\alpha < \rho$ that generates rising saving rates *also* implies **declining relative risk aversion (DRRA)**. This elegantly explains the empirical fact (from the SCF) that the rich hold approximately 80% of their portfolios in risky assets versus ~40% for the rest — with ~38% concentrated in private business equity. Carroll concludes that *both* Capitalist Spirit preferences *and* capital market imperfections (particularly entrepreneurial self-financing constraints) are needed to explain the full picture. This is the *what* of rich households' behavior: they don't just save more, they save *differently*.

### Paper 3: What Happens to Everyone Else

Doerr, Drechsel & Lee (2026) trace the **macroeconomic consequences** of these portfolio choices. Taking as empirical input the same SCF evidence on portfolio composition that Papers 1–2 established — specifically that deposit shares decline sharply with income — the authors identify a novel transmission channel: as top income shares rise, aggregate household savings shift away from bank deposits toward stocks, bonds, and direct business holdings. This starves banks of cheap deposit funding, raising financing costs for **bank-dependent small firms**, which reduces their job creation. Using US state-level instrumental variable estimation over 1980–2015 and a general equilibrium model, they find rising top incomes explain roughly 13% of the decline in small-firm employment share.

## Shared Theoretical Architecture

All three papers share a common theoretical scaffolding:

1. **Departure from CRRA homothetic preferences.** CRRA with constant relative risk aversion serves as the baseline all three depart from. Paper 1 introduces non-homotheticity in saving; Paper 2 shows it implies non-homotheticity in risk-taking; Paper 3 takes the resulting non-homothetic portfolio allocation as an empirical input to a macro model.

2. **Capital market imperfections.** Paper 2 emphasizes entrepreneurial self-financing (the rich hold private business equity because external financing is costly or unavailable), while Paper 3 focuses on the mirror image — small firms that *cannot* self-finance depend on bank lending, and are therefore vulnerable when deposit funding dries up. These are two sides of the same coin.

3. **Non-homothetic portfolio composition.** The key empirical fact linking all three is that *the composition of savings varies systematically with wealth/income* — not just the level. All three papers rely on the Survey of Consumer Finances to establish this.

## Authorship and Citation Structure

Carroll authored Papers 1 and 2, with Paper 2 explicitly building on Paper 1's Capitalist Spirit model. Doerr, Drechsel, and Lee are different authors working downstream, leveraging the same SCF evidence and theoretical insights that Carroll established. This represents a natural progression from micro-foundations (individual motivation → individual portfolio choice) to macro consequences (aggregate portfolio reallocation → financial intermediation → real economy).

## The Unified Narrative

Read together, the three papers tell a single story: **The rich save more (Paper 1), invest what they save in riskier and less bank-intermediated assets (Paper 2), and when income concentration rises, this portfolio tilt drains deposits from the banking system, raising borrowing costs for small firms and destroying jobs (Paper 3).** The preference structure that makes the rich save so much is inseparable from the preference structure that makes them invest aggressively, and the aggregate implications of those investment choices flow through financial intermediation to the real economy — connecting the psychology of wealth accumulation at the very top to employment outcomes at the very bottom of the firm size distribution.

## Doerr et al.'s Formalization of Carroll's Insight

Doerr et al. formalize the non-homothetic portfolio allocation through a specific preference structure: household utility includes a separate CRRA term over deposits with curvature parameter $\eta > \sigma$ (where $\sigma$ is the standard consumption CRRA). Because $\eta > \sigma$, the marginal utility of deposits falls *faster* than the marginal utility of consumption-savings, making deposits behave as a "necessity good." Rich households saturate their deposit demand at relatively low levels and shift marginal savings into capital (stocks, equity, business ownership). This is a direct formalization of Carroll's original observation that the composition, not just the level, of wealthy households' savings matters for the real economy.

## Further Reading

See `Carroll-Why-Do-The-Rich-Save-So-Much/README.md` for a literature survey of works citing Carroll (2000), tracing the intellectual lineage from the original paper through Straub (2019), Mian-Straub-Sufi (2021), and the HANK literature.

See `Doerr-et-al/` for detailed Bellman equation documentation, and `Doerr-et-al/generated-model-files/` for dolangplus YAML model descriptions and Python solvers for the Doerr et al. general equilibrium model.
