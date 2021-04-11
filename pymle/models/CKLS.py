from typing import Union
import numpy as np

from pymle.Model import Model1D


class CKLS(Model1D):
    """
    Model for CKLS
    Parameters: theta_1, theta_2, theta_3, theta_4
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] + self._params[1] * x

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[2] * x ** self._params[3]
