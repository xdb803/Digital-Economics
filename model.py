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
    return (rho * K / e_bar) * V(theta, alpha, gamma) - c * (1 - mu(theta, alpha, gamma))


def U_premium(theta, alpha, gamma, c, e_bar, rho, K, t, p):
    return ((rho * K * (1 + t)) / e_bar * V(theta, alpha, gamma)
            - c * (delta(theta, alpha, gamma) - mu(theta, alpha, gamma))
            - p)


def visibility_gain(theta, alpha, gamma, t, e_bar, rho, K):
    return (rho * K * t / e_bar) * V(theta, alpha, gamma)


def filter_saving(theta, alpha, gamma, c):
    return c * (1 - delta(theta, alpha, gamma))


def delta_U(theta, alpha, gamma, c, e_bar, rho, K, t, p):
    return (visibility_gain(theta, alpha, gamma, t, e_bar, rho, K)
            + filter_saving(theta, alpha, gamma, c)
            - p)