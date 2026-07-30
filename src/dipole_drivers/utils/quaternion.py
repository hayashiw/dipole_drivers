import numpy as np

from scipy.spatial.transform import Rotation
from typing import Sequence

__all__ = ["calculate_quaternion_alignment"]

def calculate_quaternion_alignment(
    normal: Sequence[float],
    tangent: Sequence[float]
) -> np.typing.NDArray[np.float64]:
    estimated_rotation, _ = Rotation.align_vectors(
        [normal, tangent],
        [[0,0,1], [1,0,0]],
        weights=[np.inf, 1])
    return estimated_rotation.as_quat(scalar_first=True)