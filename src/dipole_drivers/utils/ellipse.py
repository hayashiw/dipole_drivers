import numpy as np

from scipy.integrate import quad
from scipy.optimize import root_scalar

__all__ = [
    "calculate_arc_length_residual",
    "calculate_ellipse_arc_length",
    "calculate_even_ellipse_spacing",
    "calculate_total_arc_length",
    "fit_ellipse_to_points",
]

def calculate_arc_length_residual(
    a: float,
    b: float,
    theta: float,
    arc_length_target: float
) -> float:
    calculate_differential_arc_length = \
        lambda x: calculate_ellipse_arc_length(a, b, x)
    arc_length, _ = quad(calculate_differential_arc_length, 0, theta)
    return arc_length - arc_length_target

def calculate_ellipse_arc_length(a: float, b: float, theta: float) -> float:
    asin2: float = (a*np.sin(theta))**2
    bcos2: float = (b*np.cos(theta))**2
    return np.sqrt( asin2 + bcos2 )

def calculate_even_ellipse_spacing(
    a: float,
    b: float,
    ntheta: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    total_arc_length = calculate_total_arc_length(a, b)
    arc_lengths = np.linspace(0, total_arc_length, ntheta, endpoint=False)
    calculate_residual = lambda x, y: calculate_arc_length_residual(a, b, x, y)
    thetas = np.zeros(ntheta)
    for i, arc_length in enumerate(arc_lengths):
        if i == 0: continue
        result = root_scalar(
            calculate_residual,
            args=(arc_length,),
            bracket=(float(thetas[i - 1]), float(2 * np.pi)),
        )
        thetas[i] = result.root
    return thetas

def calculate_total_arc_length(a: float, b: float) -> float:
    calculate_differential_arc_length = \
        lambda x: calculate_ellipse_arc_length(a, b, x)
    total_arc_length, _ = quad(calculate_differential_arc_length, 0, 2*np.pi)
    return total_arc_length

def fit_ellipse_to_points(
    x: np.ndarray[tuple[int], np.dtype[np.float64]],
    y: np.ndarray[tuple[int], np.dtype[np.float64]]
) -> tuple[float, float, float]:
    A = np.column_stack([x**2, x, y**2])
    b = np.ones_like(x)
    c1, c2, c3 = np.linalg.lstsq(A, b, rcond=None)[0]

    ellipse_x0 = -c2 / (2 * c1)
    factor = 1 + (c2**2) / (4*c1)
    ellipse_minor_radius_x = np.sqrt(factor / c1)
    ellipse_minor_radius_y = np.sqrt(factor / c3)

    return ellipse_x0, ellipse_minor_radius_x, ellipse_minor_radius_y
