"""
Closed-form expressions from the dating platform model.
Shared by both the static figure script and the Streamlit app.
"""

import numpy as np


def mu(theta, alpha, gamma):
    """Match probability per encounter."""
    theta = np.asarray(theta, dtype=float)
    out = np.zeros_like(theta)

    low = theta <= gamma
    high = ~low

    out[low] = (theta[low]**2 * (1 - gamma)**2 * (1 + gamma)) / (2 * gamma)
    out[high] = (theta[high] - gamma/2
                 - 0.5 * (1 + gamma - gamma**2) * theta[high]**2)

    return (alpha / (1 - alpha)) * out


def V(theta, alpha, gamma):
    """Expected match value per encounter."""
    theta = np.asarray(theta, dtype=float)
    out = np.zeros_like(theta)

    low = theta <= gamma
    high = ~low

    out[low] = (alpha**2 * theta[low]**3 * (1 - gamma)**3 * (1 + gamma)**2
                / (6 * (1 - alpha) * gamma**2))

    bracket = ((1 + gamma + gamma**2) * theta[high]
               - (1 + 2*gamma) * theta[high]**2
               + (1 + 2*gamma + gamma**2 - gamma**3) / 3 * theta[high]**3
               - gamma * (2 + gamma) / 3)
    out[high] = alpha**2 / (2 * (1 - alpha)) * bracket

    return out


def delta(theta, alpha, gamma):
    """Compatibility filter threshold."""
    return (alpha / (1 - alpha)) * theta * (1 - gamma)


def theta_mu_peak(gamma):
    """Interior peak of mu (in the high-type regime)."""
    return 1.0 / (1 + gamma - gamma**2)


def theta_V_peak(gamma):
    """Interior peak of V (in the high-type regime)."""
    num = (1 + 2*gamma) - np.sqrt(gamma) * (1 - gamma) * (1 + gamma)
    den = 1 + 2*gamma + gamma**2 - gamma**3
    return num / den


def U_free(theta, alpha, gamma, c, e_bar, rho, K):
    return (rho * K / e_bar) * V(theta, alpha, gamma) - c * K * (1 - mu(theta, alpha, gamma))


def U_premium(theta, alpha, gamma, c, e_bar, rho, K, t, p):
    return ((rho * K * (1 + t)) / e_bar * V(theta, alpha, gamma)
            - c * K * (delta(theta, alpha, gamma) - mu(theta, alpha, gamma))
            - p)


def visibility_gain(theta, alpha, gamma, t, e_bar, rho, K):
    return (rho * K * t / e_bar) * V(theta, alpha, gamma)


def filter_saving(theta, alpha, gamma, c, K):
    return c * K * (1 - delta(theta, alpha, gamma))


def delta_U(theta, alpha, gamma, c, e_bar, rho, K, t, p):
    return (visibility_gain(theta, alpha, gamma, t, e_bar, rho, K)
            + filter_saving(theta, alpha, gamma, c, K)
            - p)

def adoption_rate(p, theta_grid, alpha, gamma, c, e_bar, rho, K, t):
    """Fraction of users for whom Delta U > 0 at price p."""
    surplus = delta_U(theta_grid, alpha, gamma, c, e_bar, rho, K, t, p)
    return float(np.mean(surplus > 0))



def equilibrium_pi(p, alpha, gamma, c, rho, K, t, tol=1e-8, n_theta=500):
    """
    Find equilibrium premium adoption rate as fixed point of
        F(pi) = fraction of users with delta_U > 0 at e_bar = 1 + pi*t.

    F is non-increasing in pi (higher pi → higher e_bar → lower visibility gain).
    g(pi) = F(pi) - pi satisfies g(0) >= 0 and g(1) <= 0, so bisection
    finds the unique fixed point reliably without oscillation.
    Returns (pi_eq, e_bar_eq).
    """
    theta_grid = np.linspace(0.001, 1.0, n_theta)

    def F(pi):
        e_bar = 1.0 + pi * t
        surplus = delta_U(theta_grid, alpha, gamma, c, e_bar, rho, K, t, p)
        return float(np.mean(surplus > 0))

    # Corner solution: no adoption at all even when e_bar = 1
    if F(0.0) == 0.0:
        return 0.0, 1.0

    # Full adoption at pi=1 (unusual but handle gracefully)
    if F(1.0) >= 1.0:
        return 1.0, 1.0 + t

    # g(0) > 0, g(1) < 0 → unique root via bisection
    lo, hi = 0.0, 1.0
    for _ in range(60):          # 60 halvings → error < 1e-18
        mid = 0.5 * (lo + hi)
        if F(mid) - mid > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    pi_eq = 0.5 * (lo + hi)
    return pi_eq, 1.0 + pi_eq * t
