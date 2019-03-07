import scipy as sp
import numpy as np
import scipy.integrate as spi


def Heston_Call_Value_Int(kappa, theta, sigma, rho, v0, r, T, s0, K):
    a = s0 * Heston_P_Value(kappa, theta, sigma, rho, v0, r, T, s0, K, 1)
    b = K * np.exp(-r * T) * Heston_P_Value(kappa, theta, sigma, rho, v0, r, T, s0, K, 2)
    # print (a,b)
    return a - b


def Heston_P_Value(kappa, theta, sigma, rho, v0, r, T, s0, K, type):
    ifun = lambda phi: Int_Function_1(phi, kappa, theta, sigma, rho, v0, r, T, s0, K, type)
    return 0.5 + (1 / np.pi) * spi.quad(ifun, 0, 100)[0]


def Int_Function_1(phi, kappa, theta, sigma, rho, v0, r, T, s0, K, type):
    temp = (np.exp(-1 * 1j * phi * np.log(K)) * Int_Function_2(phi, kappa, theta, sigma, rho, v0, r, T, s0, K, type))
    return temp


def Int_Function_2(phi, kappa, theta, sigma, rho, v0, r, T, s0, K, type):
    if type == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa
    a = kappa * theta
    x = np.log(s0)
    d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 - sigma ** 2 * (2 * u * phi * 1j - phi ** 2))
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)
    D = r * phi * 1j * T + (a / sigma ** 2) * (
            (b - rho * sigma * phi * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    E = ((b - rho * sigma * phi * 1j + d) / sigma ** 2) * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))

    return np.exp(D + E * v0 + 1j * phi * x)


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
    f_phi = lambda phi: char_func(phi, kappa, theta, sigma, rho, v0, r, t, s0, func_type)
    p = 0.5 + (1 / np.pi) * spi.quad(sp.real(f_phi), 0, np.inf)
    return p
