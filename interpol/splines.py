"""Weights and derivatives of spline orders 0 to 7."""
from enum import Enum

import torch

from .jit_utils import cube, pow4, pow5, pow6, pow7, square


class InterpolationType(Enum):
    nearest = zeroth = 0
    linear = first = 1
    quadratic = second = 2
    cubic = third = 3
    fourth = 4
    fifth = 5
    sixth = 6
    seventh = 7


@torch.jit.script
class Spline:
    def __init__(self, order: int = 1):
        self.order = order

    def weight(self, x):
        w = self.fastweight(x)
        zero = torch.zeros([1], dtype=x.dtype, device=x.device)
        w = torch.where(x.abs() >= (self.order + 1) / 2, zero, w)
        return w

    def fastweight(self, x):
        if self.order == 0:
            return torch.ones(x.shape, dtype=x.dtype, device=x.device)
        x = x.abs()
        if self.order == 1:
            return 1 - x
        if self.order == 2:
            x_low = 0.75 - square(x)
            x_up = 0.5 * square(1.5 - x)
            return torch.where(x < 0.5, x_low, x_up)
        if self.order == 3:
            x_low = (x * x * (x - 2.0) * 3.0 + 4.0) / 6.0
            x_up = cube(2.0 - x) / 6.0
            return torch.where(x < 1.0, x_low, x_up)
        if self.order == 4:
            x_low = square(x)
            x_low = x_low * (x_low * 0.25 - 0.625) + 115.0 / 192.0
            x_mid = x * (x * (x * (5.0 - x) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0
            x_up = pow4(x - 2.5) / 24.0
            return torch.where(x < 0.5, x_low, torch.where(x < 1.5, x_mid, x_up))
        if self.order == 5:
            x_low = square(x)
            x_low = x_low * (x_low * (0.25 - x / 12.0) - 0.5) + 0.55
            x_mid = x * (x * (x * (x * (x / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425
            x_up = pow5(3 - x) / 120.0
            return torch.where(x < 1.0, x_low, torch.where(x < 2.0, x_mid, x_up))
        if self.order == 6:
            x_low = square(x)
            x_low = x_low * (x_low * (7.0 / 48.0 - x_low / 36.0) - 77.0 / 192.0) + 5887.0 / 11520.0
            x_mid_low = (
                x
                * (x * (x * (x * (x * (x / 48.0 - 7.0 / 48.0) + 0.328125) - 35.0 / 288.0) - 91.0 / 256.0) - 7.0 / 768.0)
                + 7861.0 / 15360.0
            )
            x_mid_up = (
                x
                * (x * (x * (x * (x * (7.0 / 60.0 - x / 120.0) - 0.65625) + 133.0 / 72.0) - 2.5703125) + 1267.0 / 960.0)
                + 1379.0 / 7680.0
            )
            x_up = pow6(x - 3.5) / 720.0
            return torch.where(x < 0.5, x_low, torch.where(x < 1.5, x_mid_low, torch.where(x < 2.5, x_mid_up, x_up)))
        if self.order == 7:
            x_low = square(x)
            x_low = x_low * (x_low * (x_low * (x / 144.0 - 1.0 / 36.0) + 1.0 / 9.0) - 1.0 / 3.0) + 151.0 / 315.0
            x_mid_low = (
                x * (x * (x * (x * (x * (x * (0.05 - x / 240.0) - 7.0 / 30.0) + 0.5) - 7.0 / 18.0) - 0.1) - 7.0 / 90.0)
                + 103.0 / 210.0
            )
            x_mid_up = (
                x
                * (
                    x
                    * (
                        x * (x * (x * (x * (x / 720.0 - 1.0 / 36.0) + 7.0 / 30.0) - 19.0 / 18.0) + 49.0 / 18.0)
                        - 23.0 / 6.0
                    )
                    + 217.0 / 90.0
                )
                - 139.0 / 630.0
            )
            x_up = pow7(4 - x) / 5040.0
            return torch.where(x < 1.0, x_low, torch.where(x < 2.0, x_mid_low, torch.where(x < 3.0, x_mid_up, x_up)))
        raise NotImplementedError

    def grad(self, x):
        if self.order == 0:
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        g = self.fastgrad(x)
        zero = torch.zeros([1], dtype=x.dtype, device=x.device)
        g = torch.where(x.abs() >= (self.order + 1) / 2, zero, g)
        return g

    def fastgrad(self, x):
        if self.order == 0:
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        return self._fastgrad(x.abs()).mul(x.sign())

    def _fastgrad(self, x):
        if self.order == 1:
            return torch.ones(x.shape, dtype=x.dtype, device=x.device)
        if self.order == 2:
            return torch.where(x < 0.5, -2 * x, x - 1.5)
        if self.order == 3:
            g_low = x * (x * 1.5 - 2)
            g_up = -0.5 * square(2 - x)
            return torch.where(x < 1, g_low, g_up)
        if self.order == 4:
            g_low = x * (square(x) - 1.25)
            g_mid = x * (x * (x * (-2.0 / 3.0) + 2.5) - 2.5) + 5.0 / 24.0
            g_up = cube(2.0 * x - 5.0) / 48.0
            return torch.where(x < 0.5, g_low, torch.where(x < 1.5, g_mid, g_up))
        if self.order == 5:
            g_low = x * (x * (x * (x * (-5.0 / 12.0) + 1.0)) - 1.0)
            g_mid = x * (x * (x * (x * (5.0 / 24.0) - 1.5) + 3.75) - 3.5) + 0.625
            g_up = pow4(x - 3.0) / (-24.0)
            return torch.where(x < 1, g_low, torch.where(x < 2, g_mid, g_up))
        if self.order == 6:
            g_low = square(x)
            g_low = x * (g_low * (7.0 / 12.0) - square(g_low) / 6.0 - 77.0 / 96.0)
            g_mid_low = x * (x * (x * (x * (x * 0.125 - 35.0 / 48.0) + 1.3125) - 35.0 / 96.0) - 0.7109375) - 7.0 / 768.0
            g_mid_up = (
                x * (x * (x * (x * (x / (-20.0) + 7.0 / 12.0) - 2.625) + 133.0 / 24.0) - 5.140625) + 1267.0 / 960.0
            )
            g_up = pow5(2 * x - 7) / 3840.0
            return torch.where(x < 0.5, g_low, torch.where(x < 1.5, g_mid_low, torch.where(x < 2.5, g_mid_up, g_up)))
        if self.order == 7:
            g_low = square(x)
            g_low = x * (g_low * (g_low * (x * (7.0 / 144.0) - 1.0 / 6.0) + 4.0 / 9.0) - 2.0 / 3.0)
            g_mid_low = (
                x * (x * (x * (x * (x * (x * (-7.0 / 240.0) + 3.0 / 10.0) - 7.0 / 6.0) + 2.0) - 7.0 / 6.0) - 1.0 / 5.0)
                - 7.0 / 90.0
            )
            g_mid_up = (
                x
                * (
                    x * (x * (x * (x * (x * (7.0 / 720.0) - 1.0 / 6.0) + 7.0 / 6.0) - 38.0 / 9.0) + 49.0 / 6.0)
                    - 23.0 / 3.0
                )
                + 217.0 / 90.0
            )
            g_up = pow6(x - 4) / (-720.0)
            return torch.where(x < 1, g_low, torch.where(x < 2, g_mid_low, torch.where(x < 3, g_mid_up, g_up)))
        raise NotImplementedError

    def hess(self, x):
        if self.order == 0:
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        h = self.fasthess(x)
        zero = torch.zeros([1], dtype=x.dtype, device=x.device)
        h = torch.where(x.abs() >= (self.order + 1) / 2, zero, h)
        return h

    def fasthess(self, x):
        if self.order in (0, 1):
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        x = x.abs()
        if self.order == 2:
            one = torch.ones([1], dtype=x.dtype, device=x.device)
            return torch.where(x < 0.5, -2 * one, one)
        if self.order == 3:
            return torch.where(x < 1, 3.0 * x - 2.0, 2.0 - x)
        if self.order == 4:
            return torch.where(
                x < 0.5,
                3.0 * square(x) - 1.25,
                torch.where(x < 1.5, x * (-2.0 * x + 5.0) - 2.5, square(2.0 * x - 5.0) / 8.0),
            )
        if self.order == 5:
            h_low = square(x)
            h_low = -h_low * (x * (5.0 / 3.0) - 3.0) - 1.0
            h_mid = x * (x * (x * (5.0 / 6.0) - 9.0 / 2.0) + 15.0 / 2.0) - 7.0 / 2.0
            h_up = 9.0 / 2.0 - x * (x * (x / 6.0 - 3.0 / 2.0) + 9.0 / 2.0)
            return torch.where(x < 1, h_low, torch.where(x < 2, h_mid, h_up))
        if self.order == 6:
            h_low = square(x)
            h_low = -h_low * (h_low * (5.0 / 6) - 7.0 / 4.0) - 77.0 / 96.0
            h_mid_low = x * (x * (x * (x * (5.0 / 8.0) - 35.0 / 12.0) + 63.0 / 16.0) - 35.0 / 48.0) - 91.0 / 128.0
            h_mid_up = -(x * (x * (x * (x / 4.0 - 7.0 / 3.0) + 63.0 / 8.0) - 133.0 / 12.0) + 329.0 / 64.0)
            h_up = x * (x * (x * (x / 24.0 - 7.0 / 12.0) + 49.0 / 16.0) - 343.0 / 48.0) + 2401.0 / 384.0
            return torch.where(x < 0.5, h_low, torch.where(x < 1.5, h_mid_low, torch.where(x < 2.5, h_mid_up, h_up)))
        if self.order == 7:
            h_low = square(x)
            h_low = h_low * (h_low * (x * (7.0 / 24.0) - 5.0 / 6.0) + 4.0 / 3.0) - 2.0 / 3.0
            h_mid_low = -(
                x * (x * (x * (x * (x * (7.0 / 40.0) - 3.0 / 2.0) + 14.0 / 3.0) - 6.0) + 7.0 / 3.0) + 1.0 / 5.0
            )
            h_mid_up = (
                x * (x * (x * (x * (x * (7.0 / 120.0) - 5.0 / 6.0) + 14.0 / 3.0) - 38.0 / 3.0) + 49.0 / 3.0)
                - 23.0 / 3.0
            )
            h_up = -(x * (x * (x * (x * (x / 120.0 - 1.0 / 6.0) + 4.0 / 3.0) - 16.0 / 3.0) + 32.0 / 3.0) - 128.0 / 15.0)
            return torch.where(x < 1, h_low, torch.where(x < 2, h_mid_low, torch.where(x < 3, h_mid_up, h_up)))
        raise NotImplementedError
