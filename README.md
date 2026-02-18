# Portfolios of the Rich: From Micro Preferences to Macro Consequences

This repository was created on Wednesday, February 18, 2026, after an interesting presentation of "Income Inequality and Job Creation" by Thomas Drechsel at Johns Hopkins University. The talk prompted an exploration of the intellectual connections between the Doerr, Drechsel & Lee paper and two earlier papers by Christopher Carroll on the saving behavior and portfolio composition of the wealthy.

## The Three-Paper Arc

The papers collected here tell a single story across two decades of research:

1. **Why do the rich save so much?** Carroll (2000) argues that standard models fail to explain the saving behavior of the wealthy and proposes a "Capitalist Spirit" framework where wealth yields direct utility as a luxury good.

2. **What do the rich invest in?** Carroll (2000) documents that the wealthy hold ~80% of their portfolios in risky assets, especially private business equity, and shows that the same preference structure implies declining relative risk aversion.

3. **What happens to everyone else?** Doerr, Drechsel & Lee (2026) identifies the macroeconomic consequences through both novel empirical work and a general equilibrium model. Empirically, using US state-level variation and an instrumental variable strategy, they establish that rising top income shares shift household savings from bank deposits toward stocks and equity, draining deposit funding from banks and raising lending rates for bank-dependent small firms. Their GE model — with endogenous public/private firm dynamics, a banking sector, and non-homothetic household preferences — shows this channel accounts for ~13% of the decline in small-firm employment share since 1980.

## What's in this Repository

After the seminar, `econ-ark-matsya` (a RAG system connected to the Econ-ARK/Bellman-DDSL literature corpus) was used to extract, formalize, and numerically solve the Bellman optimization problems in the Doerr et al. general equilibrium model, working from simplest to hardest.

### Directory Structure

```
Carroll/
├── README.md                          How the three papers relate
├── Portfolios-of-the-Rich/
│   ├── Carroll-Portfolios-of-the-Rich.pdf
│   └── Carroll-Portfolios-of-the-Rich.tex
└── Why-Do-The-Rich-Save-So-Much/
    ├── README.md                      Literature survey (631+ citations)
    ├── Carroll-Why-Do-The-Rich-Save-So-Much.pdf
    └── Carroll-Why-Do-The-Rich-Save-So-Much.tex

Doerr-et-al/
├── README.md                          Workflow narrative: extraction → YAML → solvers
├── Doerr-et-al-Income-inequality-and-job-creation.pdf
├── Doerr_et_al.md                     Assembled Bellman equation documentation
└── generated-model-files/
    ├── matsya_household.txt           Raw matsya output: household problem
    ├── matsya_private_firm.txt        Raw matsya output: private firm problem
    ├── matsya_public_firm.txt         Raw matsya output: public firm problem
    ├── matsya_banking_equilibrium.txt Raw matsya output: banking/GE
    ├── public_firm_DDL2026.yaml       Dolangplus model description
    ├── private_firm_DDL2026.yaml      Dolangplus model description
    ├── household_DDL2026.yaml         Dolangplus model description
    ├── banking_DDL2026.yaml           Dolangplus model description
    ├── public_firm_solver.py          Numerical solver + validation
    ├── private_firm_solver.py         Numerical solver + quasi-GE analysis
    ├── household_solver.py            3D VFI solver
    └── banking_equilibrium_solver.py  Simplified GE solver
```

### Key Findings from the Numerical Work

- The **public firm** problem has a closed-form Cobb-Douglas solution. All five validation tests against the paper's reported moments pass.
- The **private firm** problem reveals that with near-constant returns to scale, the extensive margins (exit, entry, IPO transition) require full general equilibrium prices to produce interior solutions — in partial equilibrium, the IPO option value dominates all fixed costs.
- The **household** solver confirms the non-homothetic deposit demand: high-income households hold 1–5% of savings in deposits vs. 6–15% for low-income households, exactly the mechanism driving the paper's results.
- The **GE solver** reproduces the qualitative comparative statics of the paper's Figure 3: rising top incomes lead to falling deposits, rising lending spreads, and declining private-firm employment.

## Tools

- [econ-ark-matsya](https://github.com/econ-ark/econ-ark-tools): Remote RAG system querying the Econ-ARK/Bellman-DDSL corpus, used with Claude Opus for equation extraction and code generation
- Python solvers use `numpy`, `scipy`, and `matplotlib`
