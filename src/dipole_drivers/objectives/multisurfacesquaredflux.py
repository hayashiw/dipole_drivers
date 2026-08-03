import numpy as np

from simsopt.objectives import SquaredFlux

__all__ = ['MultiSurfaceSquaredFlux']


class _StackedSurfaces:
    def __init__(self, surfaces):
        self.surfaces = list(surfaces)
        ntheta = {s.gamma().shape[1] for s in self.surfaces}
        if len(ntheta) != 1:
            raise ValueError(f"all surfaces must share ntheta; got {sorted(ntheta)}.")

    def gamma(self):
        return np.concatenate([s.gamma() for s in self.surfaces])

    def normal(self):
        return np.concatenate([s.normal() for s in self.surfaces])


def MultiSurfaceSquaredFlux(surfaces, field, targets=None, **kwargs):
    target = None if (targets is None) or (all(t is None for t in targets)) else np.concatenate(targets)
    return SquaredFlux(_StackedSurfaces(surfaces), field, target=target, **kwargs)
