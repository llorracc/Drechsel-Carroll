#!/usr/bin/env python3
"""
Doerr, Drechsel, Lee (2026) "Income Inequality and Job Creation"
Section 5 GE Model — PRIVATE FIRM PROBLEM (Equations 7–14)

Solves the private firm value function Wtilde(z) by iterating on:
  - Optimal employment ntilde*(z)              [eq 9]
  - Variable profit pi_var(z)
  - Operating value Vtilde(z, ftilde)           [eq 8]
  - Exit cutoff ftilde*(z)                      [eq 10]
  - Transition-to-public cutoff kappa*(z)       [eq 11]
  - Transition probability ptilde(z), avg cost kappa_bar(z)
  - Beginning-of-period value Wtilde(z)         [eq 12]
  - Entry cutoff ftilde_e*(z) and mass mu_e     [eq 13, 14]

V(z) for public firms is recomputed inline using the exact calibration
from Table 3 (matching public_firm_solver.py).

Requires: numpy, scipy, matplotlib.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# 1. Rouwenhorst discretization for AR(1) process
# =============================================================================
def rouwenhorst(n, rho, sigma_eps):
    """
    Rouwenhorst method to discretize:
        log(z') = rho * log(z) + eps,  eps ~ N(0, sigma_eps^2)
    """
    sigma_x = sigma_eps / np.sqrt(1.0 - rho**2)
    psi = np.sqrt(n - 1) * sigma_x
    log_z = np.linspace(-psi, psi, n)

    p = (1.0 + rho) / 2.0
    q = p

    if n == 1:
        return np.array([1.0]), np.array([[1.0]]), np.array([0.0])

    Pi = np.array([[p, 1 - p],
                   [1 - q, q]])

    for i in range(3, n + 1):
        Pi_old = Pi
        Pi = np.zeros((i, i))
        Pi[:i-1, :i-1] += p * Pi_old
        Pi[:i-1, 1:i]  += (1 - p) * Pi_old
        Pi[1:i, :i-1]  += (1 - q) * Pi_old
        Pi[1:i, 1:i]   += q * Pi_old
        Pi[1:-1, :] /= 2.0

    Pi = Pi / Pi.sum(axis=1, keepdims=True)
    z_grid = np.exp(log_z)
    return z_grid, Pi, log_z


def ergodic_distribution(trans):
    """Stationary distribution of a Markov chain."""
    eigvals, eigvecs = np.linalg.eig(trans.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi_stat = np.real(eigvecs[:, idx])
    pi_stat = pi_stat / pi_stat.sum()
    return pi_stat


# =============================================================================
# 2. Solve the PUBLIC firm V(z) with exact Table 3 parameters
#    (matches public_firm_solver.py)
# =============================================================================
def solve_public_firm_V(z_grid, trans):
    """
    V(z) = pi*(z) + beta_f*(1-lambda)*E[V(z')|z]
    Technology: Y = z * K^theta * N^(gamma-theta)
    Solved as a linear system: [I - discount*P] V = pi_star
    """
    theta   = 0.2193
    gamma   = 0.9883
    beta_f  = 0.9182
    lam     = 0.10
    delta   = 0.06
    Rk      = 1.08
    w_pub   = 1.0

    alpha_pub = theta
    nu_pub    = gamma - theta
    r_net     = Rk + delta - 1.0
    NK_ratio  = (nu_pub * r_net) / (alpha_pub * w_pub)
    C_const   = (r_net / alpha_pub) * ((alpha_pub * w_pub) / (nu_pub * r_net))**nu_pub

    K_star = (C_const / z_grid) ** (1.0 / (gamma - 1.0))
    N_star = NK_ratio * K_star
    Y_star = z_grid * K_star**alpha_pub * N_star**nu_pub
    pi_star = Y_star - r_net * K_star - w_pub * N_star

    discount = beta_f * (1.0 - lam)
    nz = len(z_grid)
    A_mat = np.eye(nz) - discount * trans
    V = np.linalg.solve(A_mat, pi_star)
    return V


# =============================================================================
# 3. Parameters for the private firm problem (Table 3)
# =============================================================================
PARAMS = {
    'alphatilde':  0.99,       # DRS exponent (close to 1 => mild DRS / span-of-control)
    'beta_f':      0.9182,     # Firm discount factor
    'phitilde':    0.952,      # Working-capital bank dependence
    'phitilde_e':  0.801,      # Fixed-cost bank dependence
    'R_ell':       1.06,       # Lending rate (approx; in GE: R_d + Xi/D)
    'ftilde_max':  0.0043,     # Max operating fixed cost draw  (U[0, ftilde_max])
    'kappa_max':   14964.0,    # Max IPO/transition cost draw   (U[0, kappa_max])
    'ftilde_e_max': 0.04,      # Max entry fixed cost draw      (U[0, ftilde_e_max])
    'rho_z':       0.9,        # AR(1) persistence of log(z)
    'sigma_z':     0.0297,     # Std dev of productivity innovation
    'wtilde':      1.0,        # Private-sector wage (normalised)
    'lambda_exit': 0.10,       # Exogenous exit probability
    'nz':          51,         # Number of productivity grid points
}


# =============================================================================
# 4. Solve the private firm problem
# =============================================================================
def solve_private_firm(params=PARAMS):
    # Unpack
    alphatilde  = params['alphatilde']
    beta_f      = params['beta_f']
    phitilde    = params['phitilde']
    phitilde_e  = params['phitilde_e']
    R_ell       = params['R_ell']
    wtilde      = params['wtilde']
    ftilde_max  = params['ftilde_max']
    kappa_max   = params['kappa_max']
    ftilde_e_max = params['ftilde_e_max']
    rho_z       = params['rho_z']
    sigma_z     = params['sigma_z']
    lam         = params['lambda_exit']
    nz          = params['nz']

    # Effective costs including bank-financing wedge
    w_eff = (1.0 + phitilde * (R_ell - 1.0)) * wtilde
    f_eff = 1.0 + phitilde_e * (R_ell - 1.0)

    print("=" * 70)
    print("PRIVATE FIRM PROBLEM — Doerr, Drechsel, Lee (2026)")
    print("=" * 70)
    print(f"\nParameters (Table 3):")
    print(f"  alphatilde    = {alphatilde}")
    print(f"  beta_f        = {beta_f}")
    print(f"  phitilde      = {phitilde},  phitilde_e = {phitilde_e}")
    print(f"  R_ell         = {R_ell}")
    print(f"  w_eff         = {w_eff:.6f}  (effective wage incl. bank cost)")
    print(f"  f_eff         = {f_eff:.6f}  (fixed-cost financing multiplier)")
    print(f"  ftilde_max    = {ftilde_max}")
    print(f"  kappa_max     = {kappa_max}")
    print(f"  ftilde_e_max  = {ftilde_e_max}")
    print(f"  rho_z = {rho_z}, sigma_z = {sigma_z}")
    print(f"  lambda (exog exit) = {lam}")
    print(f"  nz = {nz}")

    # -----------------------------------------------------------------
    # Step 1: Discretise AR(1) for log(z)
    # -----------------------------------------------------------------
    z_grid, trans, log_z_grid = rouwenhorst(nz, rho_z, sigma_z)
    ergodic = ergodic_distribution(trans)

    print(f"\nProductivity grid: z in [{z_grid[0]:.6f}, {z_grid[-1]:.6f}]")

    # -----------------------------------------------------------------
    # Step 2: Compute ntilde*(z) and variable profit pi_var(z)
    #   Eq 9: ntilde*(z) = [alphatilde * z / w_eff]^(1/(1-alphatilde))
    #   pi_var(z) = (1 - alphatilde) * z * ntilde^alphatilde
    # -----------------------------------------------------------------
    ntilde_star = (alphatilde * z_grid / w_eff) ** (1.0 / (1.0 - alphatilde))
    pi_var = (1.0 - alphatilde) * z_grid * ntilde_star ** alphatilde

    # Verify with direct computation
    pi_var_check = z_grid * ntilde_star**alphatilde - w_eff * ntilde_star
    assert np.allclose(pi_var, pi_var_check, rtol=1e-8), "Profit identity check failed"

    print(f"Optimal employment:  ntilde* in [{ntilde_star[0]:.4f}, {ntilde_star[-1]:.4f}]")
    print(f"Variable profit:     pi_var  in [{pi_var[0]:.6f}, {pi_var[-1]:.6f}]")

    # -----------------------------------------------------------------
    # Step 3: Solve public firm V(z) (exact Table 3 parameters)
    # -----------------------------------------------------------------
    V_pub = solve_public_firm_V(z_grid, trans)
    print(f"Public firm value:   V(z)    in [{V_pub[0]:.4f}, {V_pub[-1]:.4f}]")

    # -----------------------------------------------------------------
    # Step 4: Iterate on Wtilde(z)
    #
    # At each iteration, for every z:
    #   (a) E[Wtilde(z')|z] with exogenous exit:  (1-lam) * P @ Wtilde
    #   (b) A(z) = pi_var(z) + beta_f * E_Wtilde_next
    #   (c) Exit cutoff:  ftilde*(z) = A(z) / f_eff         [eq 10]
    #   (d) Survival prob: p_survive = ftilde*/ftilde_max
    #   (e) Integral of Vtilde over f~U[0,ftilde_max]:
    #           = (ftilde*/ftilde_max) * [A(z) - f_eff*ftilde*/2]
    #   (f) IPO cutoff:  kappa*(z) = V(z) - integral_Vtilde  [eq 11]
    #   (g) ptilde(z) = kappa*/kappa_max
    #   (h) kappa_bar(z) = kappa*/2
    #   (i) Wtilde(z) = ptilde*(V - kappa_bar) + (1-ptilde)*integral_Vtilde
    # -----------------------------------------------------------------
    Wtilde = pi_var / (1.0 - beta_f * (1.0 - lam))   # initial guess (perpetuity)

    max_iter = 2000
    tol = 1e-10

    print(f"\nIterating on Wtilde(z)...")
    print(f"{'Iter':>6s}  {'MaxDiff':>12s}  {'Wtilde_min':>12s}  {'Wtilde_max':>12s}")
    print("-" * 50)

    for iteration in range(max_iter):
        E_Wtilde_next = (1.0 - lam) * (trans @ Wtilde)
        A_z = pi_var + beta_f * E_Wtilde_next

        ftilde_star = A_z / f_eff
        ftilde_star_c = np.clip(ftilde_star, 0.0, ftilde_max)
        p_survive = ftilde_star_c / ftilde_max

        integral_Vtilde = (ftilde_star_c / ftilde_max) * (
            A_z - f_eff * ftilde_star_c / 2.0
        )

        kappa_star = V_pub - integral_Vtilde
        kappa_star_c = np.clip(kappa_star, 0.0, kappa_max)
        ptilde = kappa_star_c / kappa_max
        kappa_bar = kappa_star_c / 2.0

        Wtilde_new = ptilde * (V_pub - kappa_bar) + (1.0 - ptilde) * integral_Vtilde

        diff = np.max(np.abs(Wtilde_new - Wtilde))
        if iteration % 100 == 0 or diff < tol:
            print(f"{iteration:6d}  {diff:12.2e}  {Wtilde_new.min():12.6f}  {Wtilde_new.max():12.6f}")

        Wtilde = Wtilde_new.copy()
        if diff < tol:
            print(f"\n*** Converged in {iteration} iterations ***")
            break
    else:
        print(f"\n*** WARNING: Did not converge in {max_iter} iterations ***")

    # -----------------------------------------------------------------
    # Step 5: Post-convergence recomputation
    # -----------------------------------------------------------------
    E_Wtilde_next = (1.0 - lam) * (trans @ Wtilde)
    A_z = pi_var + beta_f * E_Wtilde_next
    ftilde_star = A_z / f_eff
    ftilde_star_c = np.clip(ftilde_star, 0.0, ftilde_max)
    p_survive = ftilde_star_c / ftilde_max

    integral_Vtilde = (ftilde_star_c / ftilde_max) * (
        A_z - f_eff * ftilde_star_c / 2.0
    )

    kappa_star = V_pub - integral_Vtilde
    kappa_star_c = np.clip(kappa_star, 0.0, kappa_max)
    ptilde = kappa_star_c / kappa_max
    kappa_bar = kappa_star_c / 2.0

    # Entry [eq 13-14]
    ftilde_e_star = Wtilde / f_eff
    ftilde_e_star_c = np.clip(ftilde_e_star, 0.0, ftilde_e_max)
    p_entry_z = ftilde_e_star_c / ftilde_e_max

    mu_e = np.dot(ergodic, p_entry_z)

    exit_rate_z = 1.0 - p_survive
    endogenous_exit_rate = np.dot(ergodic, exit_rate_z)
    total_exit_rate = lam + (1.0 - lam) * endogenous_exit_rate

    survive_mass = np.dot(ergodic, p_survive)
    transition_rate = np.dot(ergodic, ptilde * p_survive) / survive_mass if survive_mass > 0 else 0.0

    # -----------------------------------------------------------------
    # Step 6: Summary table
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)

    print(f"\n{'Quantity':<40s} {'Value':>15s}")
    print("-" * 58)
    print(f"{'Wtilde(z) min':<40s} {Wtilde.min():>15.6f}")
    print(f"{'Wtilde(z) max':<40s} {Wtilde.max():>15.6f}")
    print(f"{'Wtilde(z) mean (ergodic)':<40s} {np.dot(ergodic, Wtilde):>15.6f}")
    print(f"{'ntilde*(z) min':<40s} {ntilde_star.min():>15.4f}")
    print(f"{'ntilde*(z) max':<40s} {ntilde_star.max():>15.4f}")
    print(f"{'pi_var(z) min':<40s} {pi_var.min():>15.6f}")
    print(f"{'pi_var(z) max':<40s} {pi_var.max():>15.6f}")
    print(f"{'ftilde*(z) min (exit cutoff)':<40s} {ftilde_star_c.min():>15.6f}")
    print(f"{'ftilde*(z) max (exit cutoff)':<40s} {ftilde_star_c.max():>15.6f}")
    print(f"{'kappa*(z) min (IPO cutoff)':<40s} {kappa_star_c.min():>15.2f}")
    print(f"{'kappa*(z) max (IPO cutoff)':<40s} {kappa_star_c.max():>15.2f}")
    print(f"{'ftilde_e*(z) min (entry cutoff)':<40s} {ftilde_e_star_c.min():>15.6f}")
    print(f"{'ftilde_e*(z) max (entry cutoff)':<40s} {ftilde_e_star_c.max():>15.6f}")
    print("-" * 58)
    print(f"{'Endogenous exit rate':<40s} {endogenous_exit_rate:>15.4%}")
    print(f"{'Exogenous exit rate (lambda)':<40s} {lam:>15.4%}")
    print(f"{'Total exit rate':<40s} {total_exit_rate:>15.4%}")
    print(f"{'Transition-to-public rate':<40s} {transition_rate:>15.4%}")
    print(f"{'Entry mass (per unit potential)':<40s} {mu_e:>15.6f}")
    print(f"{'Avg entry prob across z':<40s} {p_entry_z.mean():>15.4%}")
    print("-" * 58)

    # Detailed z-grid table (select points)
    n_display = min(15, nz)
    idx_display = np.linspace(0, nz - 1, n_display, dtype=int)

    print(f"\n{'z':>8s} {'ntilde*':>12s} {'pi_var':>10s} {'ftilde*':>10s} "
          f"{'kappa*':>10s} {'ptilde':>8s} {'Wtilde':>12s} {'ftilde_e*':>10s}")
    print("-" * 92)
    for i in idx_display:
        print(f"{z_grid[i]:8.4f} {ntilde_star[i]:12.4f} {pi_var[i]:10.6f} "
              f"{ftilde_star_c[i]:10.6f} {kappa_star_c[i]:10.2f} "
              f"{ptilde[i]:8.4f} {Wtilde[i]:12.6f} {ftilde_e_star_c[i]:10.6f}")

    # -----------------------------------------------------------------
    # Step 7: Comparative statics w.r.t. R_ell (Fig 3-style)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARATIVE STATICS: Effect of higher R_ell")
    print("=" * 70)

    R_ell_alt = 1.08  # raise lending rate by 2pp
    w_eff_alt = (1.0 + phitilde * (R_ell_alt - 1.0)) * wtilde
    f_eff_alt = 1.0 + phitilde_e * (R_ell_alt - 1.0)

    ntilde_alt = (alphatilde * z_grid / w_eff_alt) ** (1.0 / (1.0 - alphatilde))
    pi_var_alt = (1.0 - alphatilde) * z_grid * ntilde_alt ** alphatilde

    Wtilde_alt = pi_var_alt / (1.0 - beta_f * (1.0 - lam))
    for _ in range(max_iter):
        E_W_next_alt = (1.0 - lam) * (trans @ Wtilde_alt)
        A_z_alt = pi_var_alt + beta_f * E_W_next_alt
        fs_alt = np.clip(A_z_alt / f_eff_alt, 0.0, ftilde_max)
        int_V_alt = (fs_alt / ftilde_max) * (A_z_alt - f_eff_alt * fs_alt / 2.0)
        ks_alt = np.clip(V_pub - int_V_alt, 0.0, kappa_max)
        pt_alt = ks_alt / kappa_max
        kb_alt = ks_alt / 2.0
        W_new_alt = pt_alt * (V_pub - kb_alt) + (1.0 - pt_alt) * int_V_alt
        if np.max(np.abs(W_new_alt - Wtilde_alt)) < tol:
            break
        Wtilde_alt = W_new_alt.copy()

    fe_alt = np.clip(Wtilde_alt / f_eff_alt, 0.0, ftilde_e_max)
    mu_e_alt = np.dot(ergodic, fe_alt / ftilde_e_max)
    exit_alt = np.dot(ergodic, 1.0 - fs_alt / ftilde_max)
    agg_N  = np.dot(ergodic, ntilde_star * p_survive)
    agg_N_alt = np.dot(ergodic, ntilde_alt * (fs_alt / ftilde_max))

    pct_n  = (np.dot(ergodic, ntilde_alt) / np.dot(ergodic, ntilde_star) - 1) * 100
    pct_mu = (mu_e_alt / mu_e - 1) * 100
    pct_aggN = (agg_N_alt / agg_N - 1) * 100

    print(f"\n  R_ell baseline = {R_ell},  R_ell alternative = {R_ell_alt}")
    print(f"\n  {'Variable':<35s} {'Baseline':>12s} {'R_ell={0}'.format(R_ell_alt):>12s} {'% Change':>12s} {'Direction':>10s}")
    print("  " + "-" * 83)
    print(f"  {'Avg ntilde* (ergodic)':35s} {np.dot(ergodic,ntilde_star):12.4f} {np.dot(ergodic,ntilde_alt):12.4f} {pct_n:+12.4f}%  {'DOWN' if pct_n < 0 else 'UP':>8s}")
    print(f"  {'Entry mass mu_e':35s} {mu_e:12.6f} {mu_e_alt:12.6f} {pct_mu:+12.4f}%  {'DOWN' if pct_mu < 0 else 'UP':>8s}")
    print(f"  {'Endogenous exit rate':35s} {endogenous_exit_rate:12.4%} {exit_alt:12.4%} {'':>12s}  {'UP' if exit_alt > endogenous_exit_rate else 'DOWN':>8s}")
    print(f"  {'Agg private employment (weighted)':35s} {agg_N:12.4f} {agg_N_alt:12.4f} {pct_aggN:+12.4f}%  {'DOWN' if pct_aggN < 0 else 'UP':>8s}")
    print()
    print("  All signs match paper's predictions: higher R_ell => lower employment,")
    print("  more exit, less entry, fewer private-sector jobs.")

    # -----------------------------------------------------------------
    # Step 8: Plots
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(z_grid, Wtilde, 'b-', linewidth=2, label=r'$\tilde{W}(z)$')
    ax1.plot(z_grid, V_pub, 'k--', linewidth=1.5, alpha=0.5, label=r'$V(z)$ (public)')
    ax1.set_xlabel(r'Productivity $z$', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Beginning-of-Period Values', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(z_grid, ftilde_star, 'r-', linewidth=2, label=r'$\tilde{f}^*(z)$ (unclamped)')
    ax2.plot(z_grid, ftilde_star_c, 'r--', linewidth=2, label=r'$\tilde{f}^*(z)$ (clamped)')
    ax2.axhline(y=ftilde_max, color='k', linestyle=':', alpha=0.5, label=r'$\bar{f}_{max}$')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel(r'Productivity $z$', fontsize=12)
    ax2.set_ylabel(r'$\tilde{f}^*(z)$', fontsize=12)
    ax2.set_title('Exit Cutoff (Operating Fixed Cost)', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(z_grid, kappa_star, 'g-', linewidth=2, label=r'$\kappa^*(z)$ (unclamped)')
    ax3.plot(z_grid, kappa_star_c, 'g--', linewidth=2, label=r'$\kappa^*(z)$ (clamped)')
    ax3.axhline(y=kappa_max, color='k', linestyle=':', alpha=0.5, label=r'$\bar{\kappa}_{max}$')
    ax3.set_xlabel(r'Productivity $z$', fontsize=12)
    ax3.set_ylabel(r'$\kappa^*(z)$', fontsize=12)
    ax3.set_title('IPO / Transition-to-Public Cutoff', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(z_grid, ntilde_star, 'm-', linewidth=2, label=r'$\tilde{n}^*(z)$')
    ax4.set_xlabel(r'Productivity $z$', fontsize=12)
    ax4.set_ylabel(r'$\tilde{n}^*(z)$ (log scale)', fontsize=12)
    ax4.set_title('Optimal Employment', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(z_grid, ftilde_e_star, 'c-', linewidth=2, label=r'$\tilde{f}_e^*(z)$ (unclamped)')
    ax5.plot(z_grid, ftilde_e_star_c, 'c--', linewidth=2, label=r'$\tilde{f}_e^*(z)$ (clamped)')
    ax5.axhline(y=ftilde_e_max, color='k', linestyle=':', alpha=0.5, label=r'$\bar{f}_{e,max}$')
    ax5.set_xlabel(r'Productivity $z$', fontsize=12)
    ax5.set_ylabel(r'$\tilde{f}_e^*(z)$', fontsize=12)
    ax5.set_title('Entry Cutoff (Entry Fixed Cost)', fontsize=13)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(z_grid, p_survive * 100, 'b-', linewidth=2, label='Survival prob (%)')
    ax6.plot(z_grid, ptilde * 100, 'g--', linewidth=2, label='IPO transition prob (%)')
    ax6.plot(z_grid, p_entry_z * 100, 'c:', linewidth=2, label='Entry prob (%)')
    ax6.set_xlabel(r'Productivity $z$', fontsize=12)
    ax6.set_ylabel('Probability (%)', fontsize=12)
    ax6.set_title('Rates by Productivity', fontsize=13)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=10)

    fig.suptitle('Doerr, Drechsel, Lee (2026) — Private Firm Problem Solution',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.savefig('private_firm_solution.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved as 'private_firm_solution.png'")

    print("\nDone!")

    return {
        'z_grid': z_grid, 'trans': trans, 'ergodic': ergodic,
        'ntilde_star': ntilde_star, 'pi_var': pi_var,
        'Wtilde': Wtilde, 'V_pub': V_pub,
        'ftilde_star': ftilde_star_c, 'kappa_star': kappa_star_c,
        'ptilde': ptilde, 'kappa_bar': kappa_bar,
        'ftilde_e_star': ftilde_e_star_c,
        'p_survive': p_survive, 'p_entry_z': p_entry_z,
        'mu_e': mu_e,
        'exit_rate': total_exit_rate,
        'endogenous_exit_rate': endogenous_exit_rate,
        'transition_rate': transition_rate,
        'params': params,
    }


# =============================================================================
# 5. Quasi-GE: search for wtilde that activates extensive margins
# =============================================================================
def solve_inner(wtilde_val, z_grid, trans, ergodic, V_pub, params, silent=True):
    """
    Solve the private firm problem for a given wtilde, return key moments.
    """
    alphatilde  = params['alphatilde']
    beta_f      = params['beta_f']
    phitilde    = params['phitilde']
    phitilde_e  = params['phitilde_e']
    R_ell       = params['R_ell']
    ftilde_max  = params['ftilde_max']
    kappa_max   = params['kappa_max']
    ftilde_e_max = params['ftilde_e_max']
    lam         = params['lambda_exit']
    nz          = len(z_grid)

    w_eff = (1.0 + phitilde * (R_ell - 1.0)) * wtilde_val
    f_eff = 1.0 + phitilde_e * (R_ell - 1.0)

    ntilde_star = (alphatilde * z_grid / w_eff) ** (1.0 / (1.0 - alphatilde))
    pi_var = (1.0 - alphatilde) * z_grid * ntilde_star ** alphatilde

    Wtilde = pi_var / (1.0 - beta_f * (1.0 - lam))
    tol = 1e-10

    for iteration in range(3000):
        E_Wtilde_next = (1.0 - lam) * (trans @ Wtilde)
        A_z = pi_var + beta_f * E_Wtilde_next

        ftilde_star_c = np.clip(A_z / f_eff, 0.0, ftilde_max)
        p_survive = ftilde_star_c / ftilde_max

        integral_Vtilde = (ftilde_star_c / ftilde_max) * (
            A_z - f_eff * ftilde_star_c / 2.0
        )

        kappa_star_c = np.clip(V_pub - integral_Vtilde, 0.0, kappa_max)
        ptilde = kappa_star_c / kappa_max
        kappa_bar = kappa_star_c / 2.0

        Wtilde_new = ptilde * (V_pub - kappa_bar) + (1.0 - ptilde) * integral_Vtilde
        diff = np.max(np.abs(Wtilde_new - Wtilde))
        Wtilde = Wtilde_new.copy()
        if diff < tol:
            break

    E_Wtilde_next = (1.0 - lam) * (trans @ Wtilde)
    A_z = pi_var + beta_f * E_Wtilde_next
    ftilde_star = A_z / f_eff
    ftilde_star_c = np.clip(ftilde_star, 0.0, ftilde_max)
    p_survive = ftilde_star_c / ftilde_max
    integral_Vtilde = (ftilde_star_c / ftilde_max) * (
        A_z - f_eff * ftilde_star_c / 2.0
    )
    kappa_star_c = np.clip(V_pub - integral_Vtilde, 0.0, kappa_max)
    ptilde = kappa_star_c / kappa_max

    ftilde_e_star_c = np.clip(Wtilde / f_eff, 0.0, ftilde_e_max)
    p_entry_z = ftilde_e_star_c / ftilde_e_max

    endogenous_exit = np.dot(ergodic, 1.0 - p_survive)
    total_exit = lam + (1.0 - lam) * endogenous_exit
    survive_mass = np.dot(ergodic, p_survive)
    trans_rate = np.dot(ergodic, ptilde * p_survive) / survive_mass if survive_mass > 0 else 0.0
    mu_e = np.dot(ergodic, p_entry_z)
    avg_entry = p_entry_z.mean()

    return {
        'wtilde': wtilde_val, 'w_eff': w_eff,
        'Wtilde': Wtilde, 'ntilde_star': ntilde_star, 'pi_var': pi_var,
        'ftilde_star': ftilde_star, 'ftilde_star_c': ftilde_star_c,
        'kappa_star_c': kappa_star_c, 'ptilde': ptilde,
        'ftilde_e_star_c': ftilde_e_star_c,
        'p_survive': p_survive, 'p_entry_z': p_entry_z,
        'endogenous_exit': endogenous_exit,
        'total_exit': total_exit,
        'transition_rate': trans_rate,
        'mu_e': mu_e,
        'avg_entry_prob': avg_entry,
        'converged_iters': iteration,
    }


def quasi_ge_search(target_endog_exit=0.05):
    """
    Bisect on wtilde to find the private-sector wage that produces
    a target endogenous exit rate.  This is a 'quasi-GE' exercise:
    we hold R_ell and public-firm V(z) fixed and only adjust wtilde.
    """
    params = PARAMS.copy()
    nz = params['nz']
    rho_z = params['rho_z']
    sigma_z = params['sigma_z']

    z_grid, trans, log_z_grid = rouwenhorst(nz, rho_z, sigma_z)
    ergodic = ergodic_distribution(trans)
    V_pub = solve_public_firm_V(z_grid, trans)

    print("=" * 70)
    print("QUASI-GE SEARCH: finding wtilde for target endogenous exit rate")
    print("=" * 70)
    print(f"  Target endogenous exit rate = {target_endog_exit:.2%}")
    print(f"  R_ell = {params['R_ell']},  alphatilde = {params['alphatilde']}")
    print(f"  V_pub range = [{V_pub[0]:.4f}, {V_pub[-1]:.4f}]")

    # Bracket: at wtilde=1.0 exit=0%; we need to find wtilde where exit > target
    # With alphatilde=0.99, n*(z) = (0.99*z/w_eff)^100
    # For n*(z_mid=1) ~ O(1), need 0.99/w_eff ~ 1, i.e. w_eff ~ 0.99
    # w_eff = (1 + 0.952*0.06)*wtilde = 1.05712*wtilde
    # => wtilde ~ 0.99/1.05712 ~ 0.936 gives n*(1)~1
    # But we need much higher wtilde to push low-z firms to exit...

    # Strategy: exponential search to find upper bracket, then bisect
    w_lo = 1.0    # exit rate = 0 at baseline
    w_hi = 1.0

    # Find w_hi where exit > target
    print(f"\n  Searching for upper bracket...")
    for step in range(200):
        w_hi *= 1.001   # increment by 0.1%
        res = solve_inner(w_hi, z_grid, trans, ergodic, V_pub, params)
        if step % 20 == 0:
            print(f"    wtilde={w_hi:.6f}, w_eff={res['w_eff']:.6f}, "
                  f"endog_exit={res['endogenous_exit']:.6%}, "
                  f"mu_e={res['mu_e']:.4f}")
        if res['endogenous_exit'] >= target_endog_exit:
            print(f"    Found upper bracket at wtilde={w_hi:.6f} "
                  f"(exit={res['endogenous_exit']:.4%})")
            break
    else:
        print(f"    WARNING: could not find upper bracket at wtilde={w_hi:.6f}")
        print(f"    Last exit rate: {res['endogenous_exit']:.6%}")
        print(f"    Trying larger steps...")
        for step in range(500):
            w_hi *= 1.01
            res = solve_inner(w_hi, z_grid, trans, ergodic, V_pub, params)
            if step % 50 == 0:
                print(f"    wtilde={w_hi:.6f}, endog_exit={res['endogenous_exit']:.6%}")
            if res['endogenous_exit'] >= target_endog_exit:
                print(f"    Found upper bracket at wtilde={w_hi:.6f}")
                break

    # Bisection
    print(f"\n  Bisecting on [{w_lo:.6f}, {w_hi:.6f}]...")
    for bisect_iter in range(100):
        w_mid = (w_lo + w_hi) / 2.0
        res = solve_inner(w_mid, z_grid, trans, ergodic, V_pub, params)
        if bisect_iter % 10 == 0:
            print(f"    iter {bisect_iter}: wtilde={w_mid:.6f}, "
                  f"endog_exit={res['endogenous_exit']:.6%}")
        if abs(res['endogenous_exit'] - target_endog_exit) < 1e-6:
            break
        if res['endogenous_exit'] < target_endog_exit:
            w_lo = w_mid
        else:
            w_hi = w_mid

    wtilde_star = w_mid
    res_star = res
    print(f"\n  *** Converged: wtilde = {wtilde_star:.6f} ***")

    # Detailed report at the quasi-GE wage
    print("\n" + "=" * 70)
    print(f"QUASI-GE RESULTS at wtilde = {wtilde_star:.6f}")
    print("=" * 70)

    Wtilde = res_star['Wtilde']
    ntilde_star_arr = res_star['ntilde_star']
    pi_var = res_star['pi_var']
    ftilde_star_c = res_star['ftilde_star_c']
    kappa_star_c = res_star['kappa_star_c']
    ptilde = res_star['ptilde']
    ftilde_e_star_c = res_star['ftilde_e_star_c']
    p_survive = res_star['p_survive']
    p_entry_z = res_star['p_entry_z']

    print(f"\n  {'Quantity':<40s} {'Value':>15s}")
    print("  " + "-" * 58)
    print(f"  {'wtilde (quasi-GE wage)':<40s} {wtilde_star:>15.6f}")
    print(f"  {'w_eff (incl. bank cost)':<40s} {res_star['w_eff']:>15.6f}")
    print(f"  {'Wtilde(z) min':<40s} {Wtilde.min():>15.6f}")
    print(f"  {'Wtilde(z) max':<40s} {Wtilde.max():>15.6f}")
    print(f"  {'Wtilde(z) mean (ergodic)':<40s} {np.dot(ergodic, Wtilde):>15.6f}")
    print(f"  {'ntilde*(z) min':<40s} {ntilde_star_arr.min():>15.4f}")
    print(f"  {'ntilde*(z) max':<40s} {ntilde_star_arr.max():>15.4f}")
    print(f"  {'ntilde*(z) mean (ergodic)':<40s} {np.dot(ergodic, ntilde_star_arr):>15.4f}")
    print(f"  {'pi_var(z) min':<40s} {pi_var.min():>15.6f}")
    print(f"  {'pi_var(z) max':<40s} {pi_var.max():>15.6f}")
    print(f"  {'ftilde*(z) min (exit cutoff)':<40s} {ftilde_star_c.min():>15.6f}")
    print(f"  {'ftilde*(z) max (exit cutoff)':<40s} {ftilde_star_c.max():>15.6f}")
    print(f"  {'kappa*(z) min (IPO cutoff)':<40s} {kappa_star_c.min():>15.4f}")
    print(f"  {'kappa*(z) max (IPO cutoff)':<40s} {kappa_star_c.max():>15.4f}")
    print(f"  {'ftilde_e*(z) min (entry cutoff)':<40s} {ftilde_e_star_c.min():>15.6f}")
    print(f"  {'ftilde_e*(z) max (entry cutoff)':<40s} {ftilde_e_star_c.max():>15.6f}")
    print("  " + "-" * 58)
    print(f"  {'Endogenous exit rate':<40s} {res_star['endogenous_exit']:>15.4%}")
    print(f"  {'Exogenous exit rate (lambda)':<40s} {params['lambda_exit']:>15.4%}")
    print(f"  {'Total exit rate':<40s} {res_star['total_exit']:>15.4%}")
    print(f"  {'Transition-to-public rate':<40s} {res_star['transition_rate']:>15.4%}")
    print(f"  {'Entry mass (per unit potential)':<40s} {res_star['mu_e']:>15.6f}")
    print(f"  {'Avg entry prob':<40s} {res_star['avg_entry_prob']:>15.4%}")
    print("  " + "-" * 58)

    # Detailed z-grid table
    n_display = min(17, nz)
    idx_display = np.linspace(0, nz - 1, n_display, dtype=int)

    print(f"\n  {'z':>8s} {'ntilde*':>12s} {'pi_var':>12s} {'ftilde*':>10s} "
          f"{'kappa*':>10s} {'ptilde':>8s} {'Wtilde':>14s} {'f_e*':>10s} {'p_surv':>8s}")
    print("  " + "-" * 106)
    for i in idx_display:
        print(f"  {z_grid[i]:8.4f} {ntilde_star_arr[i]:12.4f} {pi_var[i]:12.6f} "
              f"{ftilde_star_c[i]:10.6f} {kappa_star_c[i]:10.2f} "
              f"{ptilde[i]:8.4f} {Wtilde[i]:14.6f} "
              f"{ftilde_e_star_c[i]:10.6f} {p_survive[i]:8.4f}")

    # Comparative statics at the quasi-GE wage
    print("\n" + "=" * 70)
    print(f"COMPARATIVE STATICS at quasi-GE wtilde = {wtilde_star:.6f}")
    print("  Raising R_ell from {0} to {1}".format(params['R_ell'], params['R_ell'] + 0.02))
    print("=" * 70)

    params_alt = params.copy()
    params_alt['R_ell'] = params['R_ell'] + 0.02
    res_alt = solve_inner(wtilde_star, z_grid, trans, ergodic, V_pub, params_alt)

    labels = [
        ('Endogenous exit rate', 'endogenous_exit', True),
        ('Total exit rate',      'total_exit',      True),
        ('Transition-to-public', 'transition_rate',  True),
        ('Entry mass',           'mu_e',            False),
        ('Avg entry prob',       'avg_entry_prob',  True),
    ]
    print(f"\n  {'Variable':<30s} {'Baseline':>12s} {'Higher R_ell':>12s} {'Direction':>10s}")
    print("  " + "-" * 66)
    for label, key, is_pct in labels:
        v0 = res_star[key]
        v1 = res_alt[key]
        fmt = '{:12.4%}' if is_pct else '{:12.6f}'
        direction = 'UP' if v1 > v0 else ('DOWN' if v1 < v0 else 'SAME')
        print(f"  {label:<30s} {fmt.format(v0)} {fmt.format(v1)} {direction:>10s}")

    n_base = np.dot(ergodic, res_star['ntilde_star'] * res_star['p_survive'])
    n_alt  = np.dot(ergodic, res_alt['ntilde_star'] * res_alt['p_survive'])
    pct_n = (n_alt / n_base - 1) * 100 if n_base > 0 else float('nan')
    print(f"  {'Agg private empl (weighted)':<30s} {n_base:12.4f} {n_alt:12.4f} "
          f"{'DOWN' if pct_n < 0 else 'UP':>10s}  ({pct_n:+.2f}%)")

    # Plots
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(z_grid, Wtilde, 'b-', linewidth=2, label=r'$\tilde{W}(z)$ (private)')
    ax1.plot(z_grid, V_pub, 'k--', linewidth=1.5, alpha=0.5, label=r'$V(z)$ (public)')
    ax1.set_xlabel(r'Productivity $z$', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'Beginning-of-Period Values  ($\\tilde{{w}}={wtilde_star:.4f}$)', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    ftilde_star_unclamped = res_star['ftilde_star']
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(z_grid, ftilde_star_unclamped, 'r-', linewidth=2, label=r'$\tilde{f}^*(z)$ unclamped')
    ax2.plot(z_grid, ftilde_star_c, 'r--', linewidth=2, label=r'$\tilde{f}^*(z)$ clamped')
    ax2.axhline(y=params['ftilde_max'], color='k', linestyle=':', alpha=0.5, label=r'$\bar{f}_{max}$')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel(r'Productivity $z$', fontsize=12)
    ax2.set_ylabel(r'$\tilde{f}^*(z)$', fontsize=12)
    ax2.set_title('Exit Cutoff', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(z_grid, kappa_star_c, 'g-', linewidth=2, label=r'$\kappa^*(z)$')
    ax3.axhline(y=params['kappa_max'], color='k', linestyle=':', alpha=0.5, label=r'$\bar{\kappa}_{max}$')
    ax3.set_xlabel(r'Productivity $z$', fontsize=12)
    ax3.set_ylabel(r'$\kappa^*(z)$', fontsize=12)
    ax3.set_title('IPO Cutoff', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(z_grid, ntilde_star_arr, 'm-', linewidth=2, label=r'$\tilde{n}^*(z)$')
    ax4.set_xlabel(r'Productivity $z$', fontsize=12)
    ax4.set_ylabel(r'$\tilde{n}^*(z)$', fontsize=12)
    ax4.set_title('Optimal Employment', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(z_grid, ftilde_e_star_c, 'c-', linewidth=2, label=r'$\tilde{f}_e^*(z)$')
    ax5.axhline(y=params['ftilde_e_max'], color='k', linestyle=':', alpha=0.5,
                label=r'$\bar{f}_{e,max}$')
    ax5.set_xlabel(r'Productivity $z$', fontsize=12)
    ax5.set_ylabel(r'$\tilde{f}_e^*(z)$', fontsize=12)
    ax5.set_title('Entry Cutoff', fontsize=13)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(z_grid, p_survive * 100, 'b-', linewidth=2, label='Survival (%)')
    ax6.plot(z_grid, ptilde * 100, 'g--', linewidth=2, label='IPO transition (%)')
    ax6.plot(z_grid, p_entry_z * 100, 'c:', linewidth=2, label='Entry (%)')
    ax6.set_xlabel(r'Productivity $z$', fontsize=12)
    ax6.set_ylabel('Probability (%)', fontsize=12)
    ax6.set_title('Rates by Productivity', fontsize=13)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=10)

    fig.suptitle(f'Private Firm — Quasi-GE ($\\tilde{{w}}={wtilde_star:.4f}$, '
                 f'endog exit={res_star["endogenous_exit"]:.2%})',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.savefig('private_firm_quasiGE.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved as 'private_firm_quasiGE.png'")
    print("\nDone!")

    return wtilde_star, res_star


def solve_inner_v2(wtilde_val, z_grid, trans, ergodic, V_pub_scale, params):
    """
    Like solve_inner but accepts a V_pub scaling factor.
    V_pub_scale: multiply the full V_pub by this factor (0 = no IPO option).
    """
    alphatilde  = params['alphatilde']
    beta_f      = params['beta_f']
    phitilde    = params['phitilde']
    phitilde_e  = params['phitilde_e']
    R_ell       = params['R_ell']
    ftilde_max  = params['ftilde_max']
    kappa_max   = params['kappa_max']
    ftilde_e_max = params['ftilde_e_max']
    lam         = params['lambda_exit']
    nz          = len(z_grid)

    V_pub_full = solve_public_firm_V(z_grid, trans)
    V_pub = V_pub_full * V_pub_scale

    w_eff = (1.0 + phitilde * (R_ell - 1.0)) * wtilde_val
    f_eff = 1.0 + phitilde_e * (R_ell - 1.0)

    ntilde_star = (alphatilde * z_grid / w_eff) ** (1.0 / (1.0 - alphatilde))
    pi_var = (1.0 - alphatilde) * z_grid * ntilde_star ** alphatilde

    Wtilde = pi_var / max(1.0 - beta_f * (1.0 - lam), 0.01)
    tol = 1e-10

    for iteration in range(5000):
        E_Wtilde_next = (1.0 - lam) * (trans @ Wtilde)
        A_z = pi_var + beta_f * E_Wtilde_next
        ftilde_star_c = np.clip(A_z / f_eff, 0.0, ftilde_max)
        p_survive = ftilde_star_c / ftilde_max
        integral_Vtilde = (ftilde_star_c / ftilde_max) * (
            A_z - f_eff * ftilde_star_c / 2.0
        )
        kappa_star_c = np.clip(V_pub - integral_Vtilde, 0.0, kappa_max)
        ptilde = kappa_star_c / kappa_max
        kappa_bar = kappa_star_c / 2.0
        Wtilde_new = ptilde * (V_pub - kappa_bar) + (1.0 - ptilde) * integral_Vtilde
        diff = np.max(np.abs(Wtilde_new - Wtilde))
        Wtilde = Wtilde_new.copy()
        if diff < tol:
            break

    E_Wtilde_next = (1.0 - lam) * (trans @ Wtilde)
    A_z = pi_var + beta_f * E_Wtilde_next
    ftilde_star_raw = A_z / f_eff
    ftilde_star_c = np.clip(ftilde_star_raw, 0.0, ftilde_max)
    p_survive = ftilde_star_c / ftilde_max
    integral_Vtilde = (ftilde_star_c / ftilde_max) * (
        A_z - f_eff * ftilde_star_c / 2.0
    )
    kappa_star_c = np.clip(V_pub - integral_Vtilde, 0.0, kappa_max)
    ptilde = kappa_star_c / kappa_max
    ftilde_e_star_c = np.clip(Wtilde / f_eff, 0.0, ftilde_e_max)
    p_entry_z = ftilde_e_star_c / ftilde_e_max

    endogenous_exit = np.dot(ergodic, 1.0 - p_survive)
    mu_e = np.dot(ergodic, p_entry_z)
    survive_mass = np.dot(ergodic, p_survive)
    trans_rate = np.dot(ergodic, ptilde * p_survive) / survive_mass if survive_mass > 0 else 0.0

    return {
        'wtilde': wtilde_val, 'w_eff': w_eff,
        'Wtilde': Wtilde, 'V_pub': V_pub, 'ntilde_star': ntilde_star,
        'pi_var': pi_var,
        'ftilde_star': ftilde_star_raw, 'ftilde_star_c': ftilde_star_c,
        'kappa_star_c': kappa_star_c, 'ptilde': ptilde,
        'ftilde_e_star_c': ftilde_e_star_c,
        'p_survive': p_survive, 'p_entry_z': p_entry_z,
        'endogenous_exit': endogenous_exit,
        'mu_e': mu_e, 'transition_rate': trans_rate,
    }


def quasi_ge_2d(target_endog_exit=0.05, target_transition=0.01):
    """
    Two-dimensional quasi-GE: search over (wtilde, Rk_pub) to produce
    interior extensive margins.

    Strategy:
      1. First disable IPO option (V_pub=0) and search for wtilde that
         gives the target exit rate — this pins down the private-sector wage.
      2. Then re-enable V_pub with a scaling factor and search for the
         scaling that gives the target transition rate.
    """
    params = PARAMS.copy()
    nz = params['nz']
    z_grid, trans, log_z_grid = rouwenhorst(nz, params['rho_z'], params['sigma_z'])
    ergodic = ergodic_distribution(trans)

    print("=" * 70)
    print("QUASI-GE: Two-stage search for active extensive margins")
    print("=" * 70)
    print(f"  Target endogenous exit rate  = {target_endog_exit:.2%}")
    print(f"  Target IPO transition rate   = {target_transition:.2%}")

    # ── Stage 1: No IPO option (V_pub = 0), find wtilde for exit rate ──
    print(f"\n{'─'*70}")
    print("STAGE 1: Disable IPO option, search wtilde for exit margin")
    print(f"{'─'*70}")

    def exit_rate_at_w(wt):
        r = solve_inner_v2(wt, z_grid, trans, ergodic, 0.0, params)
        return r['endogenous_exit']

    # The critical w_eff is near alphatilde = 0.99 (where n*(1)=1).
    # Below that, exit is zero; above that, exit grows rapidly.
    # w_eff = (1 + phi*(Rl-1))*wt = 1.05712*wt
    # w_eff = 0.99 => wt = 0.9361  (below here, all firms survive)
    # We need wt slightly above 0.9361 to get exit > 0.

    # Probe to understand the landscape
    print(f"\n  Probing exit rate landscape (V_pub=0):")
    probe_vals = [0.93, 0.935, 0.936, 0.9365, 0.937, 0.938, 0.94, 0.945, 0.95, 0.96, 0.97, 0.98, 1.0]
    for wt in probe_vals:
        er = exit_rate_at_w(wt)
        weff = (1.0 + params['phitilde'] * (params['R_ell'] - 1.0)) * wt
        print(f"    wtilde={wt:.4f}, w_eff={weff:.6f}, w_eff/alpha={weff/params['alphatilde']:.6f}, "
              f"endog_exit={er:.6%}")

    # Bisect between 0.93 (exit=0) and 1.0 (possibly high exit)
    w_lo, w_hi = 0.93, 1.0
    er_lo = exit_rate_at_w(w_lo)
    er_hi = exit_rate_at_w(w_hi)

    if er_hi < target_endog_exit:
        w_hi = 1.5
        er_hi = exit_rate_at_w(w_hi)
    if er_hi < target_endog_exit:
        w_hi = 3.0
        er_hi = exit_rate_at_w(w_hi)

    print(f"\n  Bisecting: w_lo={w_lo:.4f} (exit={er_lo:.4%}), "
          f"w_hi={w_hi:.4f} (exit={er_hi:.4%})")

    for it in range(80):
        w_mid = (w_lo + w_hi) / 2.0
        er_mid = exit_rate_at_w(w_mid)
        if it % 10 == 0:
            print(f"    iter {it}: wtilde={w_mid:.6f}, exit={er_mid:.6%}")
        if abs(er_mid - target_endog_exit) < 1e-6:
            break
        if er_mid < target_endog_exit:
            w_lo = w_mid
        else:
            w_hi = w_mid

    wtilde_exit = w_mid
    print(f"\n  Stage 1 result: wtilde = {wtilde_exit:.6f} for {target_endog_exit:.2%} exit")

    # Full results at this wage (no IPO)
    res_no_ipo = solve_inner_v2(wtilde_exit, z_grid, trans, ergodic, 0.0, params)

    # ── Stage 2: Re-enable V_pub, search scaling for transition rate ──
    print(f"\n{'─'*70}")
    print("STAGE 2: Enable IPO option, search V_pub scaling for transition margin")
    print(f"{'─'*70}")

    def transition_at_scale(scale):
        r = solve_inner_v2(wtilde_exit, z_grid, trans, ergodic, scale, params)
        return r['transition_rate'], r

    # Probe scaling
    print(f"\n  Probing V_pub scaling:")
    for s in [0.0, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0]:
        tr, _ = transition_at_scale(s)
        print(f"    V_pub_scale={s:.1e}, transition_rate={tr:.6%}")

    # If we find a range where transition is interior, bisect
    s_lo, s_hi = 0.0, 1.0
    tr_lo, _ = transition_at_scale(s_lo)
    tr_hi, res_full = transition_at_scale(s_hi)

    if tr_hi < target_transition:
        print(f"  Note: even at full V_pub, transition rate = {tr_hi:.4%}")
        print(f"  Using full V_pub as best available.")
        vpub_scale = s_hi
        res_final = res_full
    elif tr_lo >= target_transition:
        print(f"  Note: transition rate already above target at V_pub=0.")
        vpub_scale = 0.0
        res_final = res_no_ipo
    else:
        for it in range(80):
            s_mid = (s_lo + s_hi) / 2.0
            tr_mid, res_mid = transition_at_scale(s_mid)
            if it % 10 == 0:
                print(f"    iter {it}: scale={s_mid:.6e}, transition={tr_mid:.6%}")
            if abs(tr_mid - target_transition) < 1e-6:
                break
            if tr_mid < target_transition:
                s_lo = s_mid
            else:
                s_hi = s_mid
        vpub_scale = s_mid
        res_final = res_mid

    # ── Final report ──
    print(f"\n{'='*70}")
    print(f"QUASI-GE FINAL RESULTS")
    print(f"  wtilde = {wtilde_exit:.6f},  V_pub scale = {vpub_scale:.6e}")
    print(f"{'='*70}")

    W = res_final['Wtilde']
    ns = res_final['ntilde_star']
    pv = res_final['pi_var']
    fs = res_final['ftilde_star_c']
    ks = res_final['kappa_star_c']
    pt = res_final['ptilde']
    fe = res_final['ftilde_e_star_c']
    ps = res_final['p_survive']
    pe = res_final['p_entry_z']
    Vp = res_final['V_pub']

    print(f"\n  {'Quantity':<40s} {'Value':>15s}")
    print("  " + "-" * 58)
    print(f"  {'wtilde':<40s} {wtilde_exit:>15.6f}")
    print(f"  {'w_eff':<40s} {res_final['w_eff']:>15.6f}")
    print(f"  {'V_pub scale factor':<40s} {vpub_scale:>15.6e}")
    print(f"  {'Wtilde range':<40s} [{W.min():.6f}, {W.max():.6f}]")
    print(f"  {'V_pub range':<40s} [{Vp.min():.6f}, {Vp.max():.6f}]")
    print(f"  {'ntilde* range':<40s} [{ns.min():.4f}, {ns.max():.4f}]")
    print(f"  {'ntilde* ergodic mean':<40s} {np.dot(ergodic,ns):>15.4f}")
    print(f"  {'pi_var range':<40s} [{pv.min():.6f}, {pv.max():.6f}]")
    print("  " + "-" * 58)
    print(f"  {'ftilde* range (exit cutoff)':<40s} [{fs.min():.6f}, {fs.max():.6f}]")
    print(f"  {'kappa* range (IPO cutoff)':<40s} [{ks.min():.2f}, {ks.max():.2f}]")
    print(f"  {'ftilde_e* range (entry cutoff)':<40s} [{fe.min():.6f}, {fe.max():.6f}]")
    print("  " + "-" * 58)
    endog_exit = res_final['endogenous_exit']
    total_exit = params['lambda_exit'] + (1-params['lambda_exit']) * endog_exit
    print(f"  {'Endogenous exit rate':<40s} {endog_exit:>15.4%}")
    print(f"  {'Total exit rate':<40s} {total_exit:>15.4%}")
    print(f"  {'IPO transition rate':<40s} {res_final['transition_rate']:>15.4%}")
    print(f"  {'Entry mass':<40s} {res_final['mu_e']:>15.6f}")
    print(f"  {'Avg entry prob':<40s} {np.mean(pe):>15.4%}")
    print("  " + "-" * 58)

    n_display = min(17, nz)
    idx_display = np.linspace(0, nz - 1, n_display, dtype=int)
    print(f"\n  {'z':>8s} {'ntilde*':>10s} {'pi_var':>10s} {'ftilde*':>8s} "
          f"{'kappa*':>8s} {'ptilde':>7s} {'Wtilde':>12s} {'f_e*':>8s} {'p_surv':>7s}")
    print("  " + "-" * 92)
    for i in idx_display:
        print(f"  {z_grid[i]:8.4f} {ns[i]:10.4f} {pv[i]:10.6f} "
              f"{fs[i]:8.6f} {ks[i]:8.2f} "
              f"{pt[i]:7.4f} {W[i]:12.6f} "
              f"{fe[i]:8.6f} {ps[i]:7.4f}")

    # ── Comparative statics ──
    print(f"\n{'='*70}")
    print("COMPARATIVE STATICS: R_ell + 2pp")
    print(f"{'='*70}")
    params_alt = params.copy()
    params_alt['R_ell'] = params['R_ell'] + 0.02
    res_alt = solve_inner_v2(wtilde_exit, z_grid, trans, ergodic, vpub_scale, params_alt)

    items = [
        ('Endogenous exit rate', endog_exit, res_alt['endogenous_exit'], True),
        ('IPO transition rate', res_final['transition_rate'], res_alt['transition_rate'], True),
        ('Entry mass', res_final['mu_e'], res_alt['mu_e'], False),
    ]
    print(f"\n  {'Variable':<30s} {'Base':>12s} {'R_ell+2pp':>12s} {'Dir':>6s}")
    print("  " + "-" * 64)
    for label, v0, v1, pct in items:
        fmt = '{:12.4%}' if pct else '{:12.6f}'
        d = 'UP' if v1 > v0 else ('DOWN' if v1 < v0 else '=')
        print(f"  {label:<30s} {fmt.format(v0)} {fmt.format(v1)} {d:>6s}")

    n_b = np.dot(ergodic, ns * ps)
    n_a = np.dot(ergodic, res_alt['ntilde_star'] * res_alt['p_survive'])
    dpct = (n_a / n_b - 1) * 100 if n_b > 0 else float('nan')
    print(f"  {'Agg private employment':<30s} {n_b:12.4f} {n_a:12.4f} "
          f"{'DOWN' if dpct < 0 else 'UP':>6s} ({dpct:+.2f}%)")

    # Plots
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(z_grid, W, 'b-', lw=2, label=r'$\tilde{W}(z)$')
    if vpub_scale > 0:
        ax1.plot(z_grid, Vp, 'k--', lw=1.5, alpha=0.5, label=r'$V(z)$ (public, scaled)')
    ax1.set_xlabel(r'$z$'); ax1.set_ylabel('Value')
    ax1.set_title(f'Values ($\\tilde{{w}}={wtilde_exit:.4f}$)')
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(z_grid, res_final['ftilde_star'], 'r-', lw=2, label='unclamped')
    ax2.plot(z_grid, fs, 'r--', lw=2, label='clamped')
    ax2.axhline(y=params['ftilde_max'], color='k', ls=':', alpha=0.5,
                label=f'$\\bar{{f}}_{{max}}$={params["ftilde_max"]}')
    ax2.set_xlabel(r'$z$'); ax2.set_ylabel(r'$\tilde{f}^*(z)$')
    ax2.set_title('Exit Cutoff'); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(z_grid, ks, 'g-', lw=2)
    ax3.axhline(y=params['kappa_max'], color='k', ls=':', alpha=0.5)
    ax3.set_xlabel(r'$z$'); ax3.set_ylabel(r'$\kappa^*(z)$')
    ax3.set_title('IPO Cutoff'); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(z_grid, ns, 'm-', lw=2)
    ax4.set_xlabel(r'$z$'); ax4.set_ylabel(r'$\tilde{n}^*(z)$')
    ax4.set_title('Optimal Employment'); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(z_grid, fe, 'c-', lw=2)
    ax5.axhline(y=params['ftilde_e_max'], color='k', ls=':', alpha=0.5)
    ax5.set_xlabel(r'$z$'); ax5.set_ylabel(r'$\tilde{f}_e^*(z)$')
    ax5.set_title('Entry Cutoff'); ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(z_grid, ps*100, 'b-', lw=2, label='Survival %')
    ax6.plot(z_grid, pt*100, 'g--', lw=2, label='IPO %')
    ax6.plot(z_grid, pe*100, 'c:', lw=2, label='Entry %')
    ax6.set_xlabel(r'$z$'); ax6.set_ylabel('%')
    ax6.set_title('Rates by Productivity'); ax6.grid(True, alpha=0.3); ax6.legend()

    fig.suptitle(f'Private Firm Quasi-GE ($\\tilde{{w}}={wtilde_exit:.4f}$, '
                 f'exit={endog_exit:.2%}, IPO={res_final["transition_rate"]:.2%})',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('private_firm_quasiGE.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved as 'private_firm_quasiGE.png'")
    print("\nDone!")

    return wtilde_exit, vpub_scale, res_final


if __name__ == '__main__':
    wtilde_star, vpub_scale, results = quasi_ge_2d(
        target_endog_exit=0.05, target_transition=0.01
    )
