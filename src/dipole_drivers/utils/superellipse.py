import numpy as np

from scipy.integrate import quad

__all__ = [
    "calculate_superellipse_coeff_a_m",
    "calculate_superellipse_fourier_coeffs",
    "calculate_superellipse_r",
]

def calculate_superellipse_coeff_a_m(
    a: float,
    b: float,
    n: float,
    m: int
) -> float:
    integrand = lambda x: calculate_superellipse_r(a, b, x, n) * np.cos(m*x)
    multiplier = 1 / ((2 if m == 0 else 1) * np.pi)
    coeff, _ = quad(integrand, 0, 2*np.pi)
    return multiplier * coeff

def calculate_superellipse_fourier_coeffs(
    a: float,
    b: float,
    n: float,
    mmax: int
) -> dict[int, float]:
    coeffs = {}
    for m in range(mmax + 1):
        if m % 2 != 0: continue
        coeffs[m] = calculate_superellipse_coeff_a_m(a, b, n, m)
    return coeffs

def calculate_superellipse_r(
    a: float,
    b: float,
    theta: float,
    n: float
) -> float:
    if (a == 0) or (b == 0):
        raise ValueError(f"a and b must be non-zero.")
    acos: float = np.abs(np.cos(theta)/a)**n
    bsin: float = np.abs(np.sin(theta)/b)**n
    npow: float = -1/n
    return (acos + bsin)**npow
