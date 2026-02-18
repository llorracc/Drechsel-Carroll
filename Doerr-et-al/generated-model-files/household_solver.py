#!/usr/bin/env python3
"""
Doerr, Drechsel, Lee (2026) "Income Inequality and Job Creation"
Section 5 — HOUSEHOLD BELLMAN PROBLEM (Equations 3-6)

Solves via Value Function Iteration on (d, k, xi) for types chi in {L, H}.
GHH preferences => analytical labor supply; the problem reduces to
choosing (d', k') given total resources m.

Vectorised numpy implementation — no Python inner loops in VFI.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# 1. CALIBRATION (Table 3)
# =============================================================================
sigma      = 1.50
nu         = 3.0
rho_xi     = 0.92
sigma_eps  = 0.12
beta       = 0.9182
psi_n      = 1.2871
psitilde_n = 1.2349
psi_d      = 0.0632
eta        = 2.6096
s_L, s_H   = 1.0, 4.6324
Rd, Rk     = 1.04, 1.08
w, wtilde  = 1.0, 1.0
Pi, T      = 0.0, 0.0

# =============================================================================
# 2. ANALYTICAL LABOR SUPPLY (GHH)
# =============================================================================
n_pub  = (w / psi_n) ** nu
n_priv = (wtilde / psitilde_n) ** nu
labor_income_base = w * n_pub + wtilde * n_priv
exp_lab = 1.0 + 1.0 / nu
labor_disutil_per_s = (psi_n * n_pub**exp_lab / exp_lab +
                       psitilde_n * n_priv**exp_lab / exp_lab)

# =============================================================================
# 3. ROUWENHORST
# =============================================================================
def rouwenhorst(n, rho, sigma_e):
    sigma_x = sigma_e / np.sqrt(1 - rho**2)
    psi_val = np.sqrt(n - 1) * sigma_x
    log_z = np.linspace(-psi_val, psi_val, n)
    p = (1 + rho) / 2.0
    q = p
    if n == 1:
        return np.exp(log_z), np.array([[1.0]]), log_z
    Pi_mat = np.array([[p, 1-p], [1-q, q]])
    for i in range(3, n+1):
        Po = Pi_mat
        Pn = np.zeros((i, i))
        Pn[:i-1, :i-1] += p * Po
        Pn[:i-1, 1:i]  += (1-p) * Po
        Pn[1:i, :i-1]  += (1-q) * Po
        Pn[1:i, 1:i]   += q * Po
        Pn[1:-1, :] /= 2.0
        Pi_mat = Pn
    Pi_mat /= Pi_mat.sum(axis=1, keepdims=True)
    return np.exp(log_z), Pi_mat, log_z

nxi = 11
xi_grid, Pi_xi, logxi_grid = rouwenhorst(nxi, rho_xi, sigma_eps)

# =============================================================================
# 4. ASSET GRIDS  (quadratic spacing, denser near zero)
# =============================================================================
nd, nk = 50, 50
d_max, k_max = 20.0, 20.0
d_grid = np.linspace(0, 1, nd)**2 * d_max
k_grid = np.linspace(0, 1, nk)**2 * k_max
d_grid[0] = 0.0
k_grid[0] = 0.0

# =============================================================================
# 5. UTILITY HELPERS
# =============================================================================
def u_crra(ubar):
    ubar = np.maximum(ubar, 1e-12)
    if sigma == 1.0:
        return np.log(ubar)
    return ubar**(1 - sigma) / (1 - sigma)

v_d_grid = psi_d * np.maximum(d_grid, 1e-12)**(1 - eta) / (1 - eta)  # (nd,)

# =============================================================================
# 6. VECTORISED VFI
# =============================================================================
def solve_household(s_chi, label, max_iter=600, tol=1e-5):
    L_s = s_chi * labor_disutil_per_s
    li  = s_chi * labor_income_base

    # m(d, k, xi) = li*xi + Rk*k + Rd*d + Pi - T
    m_grid = (li * xi_grid[None, None, :]
              + Rk * k_grid[None, :, None]
              + Rd * d_grid[:, None, None]
              + Pi - T)                      # (nd, nk, nxi)

    # Precompute the flow payoff for every (state, d', k') combination.
    # For each state point, the feasible set is d' + k' < m, and
    #   ubar = m - d' - k' - L_s > 0.
    # We precompute the d'-dependent part: deposit utility v_d(d')
    # and then vectorise over k' for each (state, d') pair.

    # Initialise V
    V = u_crra(np.maximum(m_grid * 0.5 - L_s, 0.01)) / (1 - beta)

    pol_jd = np.zeros((nd, nk, nxi), dtype=np.int32)
    pol_jk = np.zeros((nd, nk, nxi), dtype=np.int32)

    print(f"\n{'='*60}")
    print(f"Solving for type {label} (s = {s_chi})")
    print(f"  State grid: {nd}×{nk}×{nxi} = {nd*nk*nxi:,} points")
    print(f"  Choice grid: {nd}×{nk} = {nd*nk:,} pairs per state")
    print(f"{'='*60}")

    for it in range(max_iter):
        V_old = V.copy()

        # E[V(d', k', xi') | xi] for all (d', k', xi)
        EV = np.einsum('ij,abj->abi', Pi_xi, V)  # (nd, nk, nxi)

        # For each xi state, build the full payoff matrix over (jd, jk)
        # and maximise.  To avoid a Python loop over (id, ik), we loop
        # only over xi (11 iterations) and vectorise over states.
        V_new = np.full_like(V, -1e18)
        pjd_new = np.zeros_like(pol_jd)
        pjk_new = np.zeros_like(pol_jk)

        for i_xi in range(nxi):
            m_slice = m_grid[:, :, i_xi]            # (nd, nk)
            EV_slice = EV[:, :, i_xi]               # (nd_choice, nk_choice)

            # For every state (id, ik), we want:
            #   max_{jd, jk}  u_crra(m[id,ik] - d'[jd] - k'[jk] - L_s)
            #                 + v_d[jd]  + beta * EV[jd, jk, i_xi]
            # subject to m - d' - k' - L_s > 0.
            #
            # Strategy: loop over jd (50 iters), vectorise over
            # (id, ik, jk) using broadcasting.

            for jd in range(nd):
                d_next = d_grid[jd]
                vd     = v_d_grid[jd]

                # Resources left after choosing d':  remaining = m - d' - L_s
                rem = m_slice - d_next - L_s          # (nd, nk)

                # For each jk, consumption composite ubar = rem - k'[jk]
                # rem[:,:,None] - k_grid[None,None,:]  -> (nd, nk, nk_choice)
                ubar = rem[:, :, None] - k_grid[None, None, :]  # (nd, nk, nk_choice)

                # Mask infeasible
                feasible = ubar > 1e-8
                ubar_safe = np.where(feasible, ubar, 1e-8)

                payoff = (u_crra(ubar_safe)
                          + vd
                          + beta * EV_slice[jd, :][None, None, :])  # broadcast EV[jd, jk]

                payoff = np.where(feasible, payoff, -1e18)

                # Best jk for this jd, at every (id, ik)
                best_jk = np.argmax(payoff, axis=2)           # (nd, nk)
                best_val = np.take_along_axis(
                    payoff, best_jk[:, :, None], axis=2
                ).squeeze(axis=2)                              # (nd, nk)

                improve = best_val > V_new[:, :, i_xi]
                V_new[:, :, i_xi] = np.where(improve, best_val,
                                              V_new[:, :, i_xi])
                pjd_new[:, :, i_xi] = np.where(improve, jd,
                                                pjd_new[:, :, i_xi])
                pjk_new[:, :, i_xi] = np.where(improve, best_jk,
                                                pjk_new[:, :, i_xi])

        V = V_new
        pol_jd = pjd_new
        pol_jk = pjk_new

        diff = np.max(np.abs(V - V_old))
        if it % 20 == 0 or diff < tol:
            print(f"  iter {it:4d}  max|ΔV| = {diff:.3e}")
        if diff < tol:
            print(f"  *** Converged in {it} iterations ***")
            break
    else:
        print(f"  WARNING: not converged (diff={diff:.2e})")

    # Policy functions in levels
    pol_d = d_grid[pol_jd]
    pol_k = k_grid[pol_jk]
    pol_c = m_grid - pol_d - pol_k

    return V, pol_d, pol_k, pol_c, m_grid

# =============================================================================
# 7. SOLVE BOTH TYPES
# =============================================================================
print("=" * 70)
print("HOUSEHOLD PROBLEM — Doerr, Drechsel, Lee (2026)")
print("=" * 70)
print(f"  n_pub = {n_pub:.4f},  n_priv = {n_priv:.4f}")
print(f"  labor_income_base = {labor_income_base:.4f}")
for sc, lab in [(s_L,'Low'),(s_H,'High')]:
    li = sc * labor_income_base
    ld = sc * labor_disutil_per_s
    print(f"  Type {lab}: labor inc (xi=1)={li:.4f}, disutility={ld:.4f}, net={li-ld:.4f}")

V_L, pd_L, pk_L, pc_L, m_L = solve_household(s_L, "Low (s=1.0)")
V_H, pd_H, pk_H, pc_H, m_H = solve_household(s_H, "High (s=4.63)")

# =============================================================================
# 8. PLOTS
# =============================================================================
i_xi_med = nxi // 2

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: d' vs m  (k=0 slice)
ax = axes[0, 0]
ax.plot(m_L[:, 0, i_xi_med], pd_L[:, 0, i_xi_med], 'b-o', ms=2, label='Low type')
ax.plot(m_H[:, 0, i_xi_med], pd_H[:, 0, i_xi_med], 'r-s', ms=2, label='High type')
ax.set_xlabel('Resources $m$'); ax.set_ylabel("$d'$")
ax.set_title("Deposit Policy $d'(m)$"); ax.legend(); ax.grid(True, alpha=0.3)

# Panel 2: k' vs m
ax = axes[0, 1]
ax.plot(m_L[:, 0, i_xi_med], pk_L[:, 0, i_xi_med], 'b-o', ms=2, label='Low type')
ax.plot(m_H[:, 0, i_xi_med], pk_H[:, 0, i_xi_med], 'r-s', ms=2, label='High type')
ax.set_xlabel('Resources $m$'); ax.set_ylabel("$k'$")
ax.set_title("Capital Policy $k'(m)$"); ax.legend(); ax.grid(True, alpha=0.3)

# Panel 3: c vs m
ax = axes[1, 0]
ax.plot(m_L[:, 0, i_xi_med], pc_L[:, 0, i_xi_med], 'b-o', ms=2, label='Low type')
ax.plot(m_H[:, 0, i_xi_med], pc_H[:, 0, i_xi_med], 'r-s', ms=2, label='High type')
ax.set_xlabel('Resources $m$'); ax.set_ylabel("$c$")
ax.set_title("Consumption $c(m)$"); ax.legend(); ax.grid(True, alpha=0.3)

# Panel 4: Deposit share d'/(d'+k') — NON-HOMOTHETICITY
ax = axes[1, 1]
for pd_t, pk_t, color, label in [
    (pd_L, pk_L, 'blue', 'Low type'), (pd_H, pk_H, 'red', 'High type')
]:
    total_sav = []
    dep_share = []
    for i_d in range(nd):
        for i_k in range(nk):
            dn = pd_t[i_d, i_k, i_xi_med]
            kn = pk_t[i_d, i_k, i_xi_med]
            tot = dn + kn
            if tot > 0.05:
                total_sav.append(tot)
                dep_share.append(dn / tot)
    ts = np.array(total_sav); ds = np.array(dep_share)
    order = np.argsort(ts)
    ax.scatter(ts[order], ds[order], s=4, alpha=0.3, c=color, label=label)
    if len(ts) > 10:
        nbins = 15
        edges = np.linspace(ts.min(), ts.max(), nbins+1)
        centres = 0.5*(edges[:-1]+edges[1:])
        means = np.array([ds[(ts>=edges[i])&(ts<edges[i+1])].mean()
                          if ((ts>=edges[i])&(ts<edges[i+1])).sum()>0 else np.nan
                          for i in range(nbins)])
        valid = ~np.isnan(means)
        ax.plot(centres[valid], means[valid], c=color, lw=2.5, label=f'{label} (avg)')

ax.set_xlabel("Total savings $d'+k'$"); ax.set_ylabel("Deposit share")
ax.set_title("NON-HOMOTHETICITY: Deposit Share vs Savings")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.05)

fig.suptitle("Household Problem — Doerr, Drechsel, Lee (2026)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("household_policies.png", dpi=150, bbox_inches='tight')
print("\nFigure saved: household_policies.png")

# =============================================================================
# 9. SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Quantity':<40s} {'Low Type':>12s} {'High Type':>12s}")
print("-" * 66)
print(f"{'V(d=0,k=0,xi_med)':<40s} {V_L[0,0,i_xi_med]:>12.4f} {V_H[0,0,i_xi_med]:>12.4f}")
i2d = min(np.searchsorted(d_grid, 2.0), nd-1)
i2k = min(np.searchsorted(k_grid, 2.0), nk-1)
print(f"{'V(d≈2,k≈2,xi_med)':<40s} {V_L[i2d,i2k,i_xi_med]:>12.4f} {V_H[i2d,i2k,i_xi_med]:>12.4f}")
print(f"{'d_next at (d≈2,k≈2,xi_med)':<40s} {pd_L[i2d,i2k,i_xi_med]:>12.4f} {pd_H[i2d,i2k,i_xi_med]:>12.4f}")
print(f"{'k_next at (d≈2,k≈2,xi_med)':<40s} {pk_L[i2d,i2k,i_xi_med]:>12.4f} {pk_H[i2d,i2k,i_xi_med]:>12.4f}")
print(f"{'c at (d≈2,k≈2,xi_med)':<40s} {pc_L[i2d,i2k,i_xi_med]:>12.4f} {pc_H[i2d,i2k,i_xi_med]:>12.4f}")

print(f"\n--- Deposit Share d'/(d'+k') at median xi, k=0 ---")
print(f"{'Wealth (d)':>12s} {'m_L':>8s} {'share_L':>10s} {'m_H':>8s} {'share_H':>10s}")
print("-" * 52)
for td in [0.5, 1.0, 3.0, 5.0, 8.0, 12.0, 16.0]:
    i_d = min(np.searchsorted(d_grid, td), nd-1)
    dn_L = pd_L[i_d, 0, i_xi_med]; kn_L = pk_L[i_d, 0, i_xi_med]
    dn_H = pd_H[i_d, 0, i_xi_med]; kn_H = pk_H[i_d, 0, i_xi_med]
    sL = dn_L/(dn_L+kn_L) if (dn_L+kn_L)>0.01 else float('nan')
    sH = dn_H/(dn_H+kn_H) if (dn_H+kn_H)>0.01 else float('nan')
    print(f"{d_grid[i_d]:12.2f} {m_L[i_d,0,i_xi_med]:8.2f} {sL:10.3f} "
          f"{m_H[i_d,0,i_xi_med]:8.2f} {sH:10.3f}")

print(f"\neta={eta:.4f} > sigma={sigma:.2f}: deposits are a NECESSITY good.")
print("Rich households hold lower deposit share => non-homothetic portfolios.")
print("\nDone!")
