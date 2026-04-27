"""
Interactive version of the model. Run with:
    streamlit run app.py
Deploy to Streamlit Community Cloud for a public URL.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from model import (mu, V, delta, visibility_gain, filter_saving, delta_U,
                   theta_mu_peak, theta_V_peak)

st.set_page_config(page_title="Dating Platform Model", layout="wide")

st.title("Premium Visibility and Matching: Interactive Model")
st.markdown(
    "Companion to the paper. Use the sliders to explore how the "
    "matching environment and premium surplus respond to model parameters."
)

# ---- Sidebar: parameters ----
st.sidebar.header("Parameters")
alpha = st.sidebar.slider("α (weight on attractiveness)", 0.05, 0.95, 0.5, 0.05)
gamma = st.sidebar.slider("γ (attractiveness threshold)", 0.05, 0.95, 0.5, 0.05)
t     = st.sidebar.slider("t (visibility boost)", 0.0, 10.0, 5.0, 0.5)
c     = st.sidebar.slider("c (swiping cost)", 0.0, 0.5, 0.05, 0.01)
p     = st.sidebar.slider("p (premium price)", 0.0, 1.0, 0.15, 0.01)
E_s   = st.sidebar.slider("E_s (total side exposure)", 1.0, 200.0, 50.0, 1.0)

# ---- Compute ----
theta = np.linspace(0.001, 1.0, 500)

mu_vals = mu(theta, alpha, gamma)
V_vals  = V(theta, alpha, gamma)
vis     = visibility_gain(theta, alpha, gamma, t, E_s)
filt    = filter_saving(theta, alpha, gamma, c)
total   = delta_U(theta, alpha, gamma, c, E_s, t, p)

theta_mu = theta_mu_peak(gamma)
theta_V  = theta_V_peak(gamma)

# Adoption fraction (assuming uniform theta)
adopt_mass = float(np.mean(total > 0))

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
                     color="black", alpha=0.08)
    ax2.set_xlabel(r"$\theta_i$")
    ax2.set_xlim(0, 1)
    ax2.legend(frameon=False, loc="lower center")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    st.pyplot(fig2)

# ---- Live readouts ----
st.markdown("---")
st.subheader("Key quantities")
c1, c2, c3 = st.columns(3)
c1.metric(r"θ_μ (match probability peak)", f"{theta_mu:.3f}")
c2.metric(r"θ_V (expected match value peak)", f"{theta_V:.3f}")
c3.metric("Adoption fraction (uniform θ)", f"{adopt_mass:.1%}")