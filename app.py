"""
Interactive version of the model. Run with:
    streamlit run app.py
Deploy to Streamlit Community Cloud for a public URL.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from model import (mu, V, delta, visibility_gain, filter_saving, delta_U,
                   theta_mu_peak, theta_V_peak, equilibrium_pi)

st.set_page_config(page_title="Dating Platform Model", layout="wide")

st.title("Premium Visibility and Matching: Interactive Model")
st.markdown(
    "Companion to the paper. Adjust the sliders to explore how the "
    "matching environment and premium surplus respond to model parameters. "
    "The adoption rate π is solved endogenously as the fixed point at which "
    "the mass of users with ΔU > 0 equals π."
)

# ---- Sidebar: parameters ----
st.sidebar.header("Parameters")
alpha = st.sidebar.slider("α (weight on attractiveness)", 0.05, 0.95, 0.5, 0.05)
gamma = st.sidebar.slider("γ (attractiveness threshold)", 0.05, 0.95, 0.5, 0.05)
t     = st.sidebar.slider("t (visibility boost)", 0.0, 10.0, 5.0, 0.5)
c     = st.sidebar.slider("c (swiping cost)", 0.0, 0.5, 0.05, 0.01)
p     = st.sidebar.slider("p (premium price)", 0.0, 1.0, 0.15, 0.01)
rho   = st.sidebar.slider("ρ (sex ratio N_{-s}/N_s)", 0.25, 4.0, 1.0, 0.25)
K     = st.sidebar.slider("K (attention budget)", 0.1, 5.0, 1.0, 0.1)

# ---- Solve for equilibrium π endogenously ----
pi_eq, e_bar_eq = equilibrium_pi(p, alpha, gamma, c, rho, K, t)

# ---- Compute curves ----
theta = np.linspace(0.001, 1.0, 500)

mu_vals = mu(theta, alpha, gamma)
V_vals  = V(theta, alpha, gamma)
vis     = visibility_gain(theta, alpha, gamma, t, e_bar_eq, rho, K)
filt    = filter_saving(theta, alpha, gamma, c, K)
total   = delta_U(theta, alpha, gamma, c, e_bar_eq, rho, K, t, p)

theta_mu = theta_mu_peak(gamma)
theta_V  = theta_V_peak(gamma)

# ---- Equilibrium readout banner ----
st.info(
    f"**Equilibrium:** π_s = {pi_eq:.3f} · ē_s = {e_bar_eq:.3f} "
    f"(solved as fixed point of adoption given price p = {p})"
)

# ---- Layout ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Matching environment")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(theta, mu_vals, label=r"$\mu(\theta_i)$ (match probability)",
             color="black", linestyle="--")
    ax1.plot(theta, V_vals,  label=r"$V(\theta_i)$ (expected match value)",
             color="black")
    if 0 < theta_mu < 1:
        ax1.axvline(theta_mu, color="grey", linestyle=":", alpha=0.7)
        ax1.text(theta_mu, ax1.get_ylim()[1] * 0.95,
                 r"$\theta_i^\mu$", ha="center", fontsize=9)
    if 0 < theta_V < 1:
        ax1.axvline(theta_V, color="grey", linestyle="-.", alpha=0.7)
        ax1.text(theta_V, ax1.get_ylim()[1] * 0.85,
                 r"$\theta_i^V$", ha="center", fontsize=9)
    ax1.set_xlabel(r"$\theta_i$")
    ax1.set_xlim(0, 1)
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    st.pyplot(fig1)

with col2:
    st.subheader("Premium surplus decomposition")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(theta, total, color="black", linewidth=2.0,
             label=r"$\Delta U_i$")
    ax2.plot(theta, vis,  color="black", linestyle="--",
             label="Visibility gain")
    ax2.plot(theta, filt, color="black", linestyle=":",
             label="Filter saving")
    ax2.axhline(0, color="grey", linewidth=0.7)
    ax2.fill_between(theta, 0, total, where=(total > 0),
                     color="black", alpha=0.08,
                     label=f"Adoption region (π = {pi_eq:.2f})")
    if 0 < theta_V < 1:
        ax2.axvline(theta_V, color="grey", linewidth=0.7, linestyle="-.", alpha=0.7)
        ax2.text(theta_V, 0, r"$\theta_i^V$", ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel(r"$\theta_i$")
    ax2.set_xlim(0, 1)
    ax2.legend(frameon=False, loc="lower center")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    st.pyplot(fig2)

# ---- Live readouts ----
st.markdown("---")
st.subheader("Key quantities")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("θ_μ (match prob. peak)", f"{theta_mu:.3f}")
c2.metric("θ_V (match value peak)", f"{theta_V:.3f}")
c3.metric("Equilibrium π_s", f"{pi_eq:.3f}")
c4.metric("ē_s = 1 + π·t", f"{e_bar_eq:.3f}")
c5.metric("Adoption fraction check", f"{float(np.mean(total > 0)):.3f}")

# ---- Comparative statics: π vs price ----
st.markdown("---")
st.subheader("Comparative statics: equilibrium adoption vs. price")
p_grid = np.linspace(0.001, float(
    (visibility_gain(theta, alpha, gamma, t, 1.0, rho, K)
     + filter_saving(theta, alpha, gamma, c, K)).max()
) * 1.1, 80)

pi_curve = []
for p_val in p_grid:
    pi_val, _ = equilibrium_pi(p_val, alpha, gamma, c, rho, K, t)
    pi_curve.append(pi_val)
pi_curve = np.array(pi_curve)

fig3, ax3 = plt.subplots(figsize=(6, 3.5))
ax3.plot(p_grid, pi_curve, color="black")
ax3.axvline(p, color="grey", linestyle="--", alpha=0.8,
            label=f"Current p = {p}")
ax3.axhline(pi_eq, color="grey", linestyle=":", alpha=0.8,
            label=f"π_eq = {pi_eq:.3f}")
ax3.set_xlabel("Premium price p")
ax3.set_ylabel("Equilibrium π_s")
ax3.set_xlim(0, p_grid[-1])
ax3.set_ylim(0, 1.05)
ax3.legend(frameon=False)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
st.pyplot(fig3)
