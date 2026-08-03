import numpy as np
import jax.numpy as jnp
from jax import grad

from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import derivative_dec, Derivative
from simsopt.geo.jit import jit

from ..defaults import TINY

__all__ = ['LpCurrentPenalty', 'CurrentPenalty']


def Lp_current_pure(currents, p, threshold):
    r"""
    This function returns the Lp penalty for currents exceeding a threshold.

    Args:
        currents (array, shape (N,)): Coil currents.
        p (float): The Lp norm exponent.
        threshold (float): The current above which coils are penalized.

    Returns:
        The penalty value.
    """
    return jnp.sum(jnp.maximum(jnp.abs(currents) - threshold, 0) ** p)


def current_pnorm_pure(currents, p):
    r"""
    This function returns the p-norm of the current vector. The largest
    current is factored out before exponentiating, so the result cannot
    overflow however large p or the currents are.

    Args:
        currents (array, shape (N,)): Coil currents.
        p (float): The norm exponent.

    Returns:
        The norm value.
    """
    abs_currents = jnp.abs(currents)
    scale = jnp.max(abs_currents) + TINY
    return scale * jnp.sum((abs_currents / scale) ** p) ** (1 / p)


class BaseCurrentObjective(Optimizable):
    r"""
    Base class for objectives over a list of currents. Gathers the current
    values into a vector for the pure function and scatters the resulting
    gradient back through each current's own vjp, so the objective composes
    with ``ScaledCurrent`` and with symmetrized coil arrays.
    """

    def __init__(self, currents):
        self.currents = list(currents)
        Optimizable.__init__(self, depends_on=self.currents)

    def _values(self):
        return jnp.asarray([current.get_value() for current in self.currents])

    def _scatter(self, dJ_dcurrents):
        dJ_dcurrents = np.asarray(dJ_dcurrents, dtype=float)
        deriv = Derivative({}) # pyright: ignore[reportArgumentType]
        for current, dJ_dcurrent in zip(self.currents, dJ_dcurrents):
            deriv += current.vjp(np.atleast_1d(dJ_dcurrent))
        return deriv

    def J(self):
        r"""
        This returns the value of the quantity.
        """
        return float(self.J_jax(self._values()))

    @derivative_dec
    def dJ(self):
        r"""
        This returns the derivative of the quantity with respect to the
        current dofs.
        """
        return self._scatter(self.dJ_jax(self._values()))

    return_fn_map = {'J': J, 'dJ': dJ}


class LpCurrentPenalty(BaseCurrentObjective):
    r"""
    Penalize coil currents that exceed a threshold,

    .. math::
        J = \sum_i \max(|I_i| - I_0, 0)^p

    where :math:`I_0` is ``threshold``. The penalty and its gradient are
    exactly zero for every coil under the threshold, so it enforces a current
    rating without disturbing the optimum once the rating is met.

    Args:
        currents: List of :obj:`simsopt.field.Current` objects to penalize.
        p (float): The exponent. Must be at least 2 for a continuous gradient
            at the threshold.
        threshold (float): The current above which coils are penalized.
    """

    def __init__(self, currents, p=2.0, threshold=0.0):
        if p < 2:
            raise ValueError(
                f"LpCurrentPenalty requires p >= 2 for a continuous gradient "
                f"at the threshold; got p={p}.")
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative; got {threshold}.")
        self.p = float(p)
        self.threshold = float(threshold)
        self.J_jax = jit(lambda currents:
                         Lp_current_pure(currents, self.p, self.threshold))
        self.dJ_jax = jit(grad(self.J_jax))
        BaseCurrentObjective.__init__(self, currents)


class CurrentPenalty(BaseCurrentObjective):
    r"""
    Penalize coil currents through a p-norm of the current vector,

    .. math::
        J = \left(\sum_i |I_i|^p\right)^{1/p}

    which is smooth for :math:`p > 1` and approaches :math:`\max_i |I_i|` as
    :math:`p` grows, so it acts as a differentiable proxy for the largest
    current in an array. Unlike :obj:`LpCurrentPenalty` it needs no target,
    which makes it the natural objective for a Pareto sweep against field
    quality.

    Larger ``p`` tracks the maximum more closely but concentrates the gradient
    on fewer coils; at :math:`p = 12` on a typical dipole array the value is
    within a few percent of the maximum and over half the gradient falls on a
    single coil.

    Args:
        currents: List of :obj:`simsopt.field.Current` objects to penalize.
        p (float): The norm exponent. Must exceed 1.
    """

    def __init__(self, currents, p=2.0):
        if not np.isfinite(p) or p <= 1:
            raise ValueError(
                f"CurrentPenalty requires p > 1 for a well-behaved gradient; "
                f"got p={p}.")
        self.p = float(p)
        self.J_jax = jit(lambda currents: current_pnorm_pure(currents, self.p))
        self.dJ_jax = jit(grad(self.J_jax))
        BaseCurrentObjective.__init__(self, currents)
