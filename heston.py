import scipy as sp
import numpy as np
import scipy.integrate as spi
import scipy.stats as sps
import matplotlib.pyplot as plt


def char_func(phi, kappa, theta, sigma, rho, v0, r, t, s0, func_type):
    if func_type == 1:
        u = 0.5
        b = kappa + theta - rho * sigma
    else:
        u = -0.5
        b = kappa + theta
    x = np.log(s0)
    a = kappa * theta
    d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 - sigma ** 2 * (2 * u * phi * 1j - phi ** 2))
    b_plus = b - rho * sigma * phi * 1j + d
    b_minus = b - rho * sigma * phi * 1j - d
    g = b_plus / b_minus
    C = r * phi * 1j * t + (a / sigma ** 2) * (b_plus * t - 2 * np.log((1 - g * np.exp(d * t)) / (1 - g)))
    D = (b_plus / sigma ** 2) * ((1 - np.exp(d * t)) / (1 - g * np.exp(d * t)))
    return np.exp(C + D * v0 + 1j * phi * x)


def prob_integral(kappa, theta, sigma, rho, v0, r, t, s0, K, func_type):
    f_phi = lambda phi: (np.exp(-1 * 1j * phi * np.log(K)) * char_func(phi, kappa, theta, sigma, rho, v0, r, t, s0,
                                                                       func_type)) / (1j * phi)
    p = 0.5 + (1 / np.pi) * spi.quad(f_phi, 0, 100)[0]
    return p


def heston_call(kappa, theta, sigma, rho, v0, r, t, s0, K):
    a = s0 * prob_integral(kappa, theta, sigma, rho, v0, r, t, s0, K, 1)
    b = K * np.exp(-r * t) * prob_integral(kappa, theta, sigma, rho, v0, r, t, s0, K, 2)
    return a - b


def bls_call(s0, K, t, r, sigma):
    d1 = (np.log(s0 / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s0 / K) + (r - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    call = (s0 * sps.norm.cdf(d1) - K * np.exp(-r * t) * sps.norm.cdf(d2))
    return call


if __name__ == "__main__":
    kappa = 2.0
    theta = 0.01
    sigma = 0.1
    # sigma = 0.2
    rho = 0
    v0 = 0.01
    r = 0
    t = 0.5
    # s0 = 100
    K = 100

    diff_01 = []
    for s0 in np.arange(70, 141, 1):
        hst_call = heston_call(kappa, theta, 0.1, rho, v0, r, t, s0, K)
        bs_call = bls_call(s0, K, t, r, sigma)
        diff_01.append(hst_call - bs_call)

    diff_02 = []
    for s0 in np.arange(70, 141, 1):
        hst_call = heston_call(kappa, theta, 0.2, rho, v0, r, t, s0, K)
        bs_call = bls_call(s0, K, t, r, sigma)
        diff_02.append(hst_call - bs_call)

    spot_price = np.arange(70, 141, 1)
    plt.figure()
    plt.plot(spot_price, diff_01)
    plt.plot(spot_price, diff_02)
    plt.show()
