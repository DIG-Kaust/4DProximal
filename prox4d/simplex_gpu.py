import numpy as np
from pyproximal import ProxOperator
from prox4d.bisection_gpu import *


class _Simplex_numba_gpu(ProxOperator):
    """Simplex operator (numba version)
   """

    def __init__(self, n, radius, dims=None, axis=-1,
                 maxiter=100, ftol=1e-8, xtol=1e-8, call=True):
        super().__init__(None, False)
        if dims is not None and len(dims) != 2:
            raise ValueError('provide only 2 dimensions, or None')
        self.n = n
        # self.coeffs = cuda.to_device(np.ones(self.n if dims is None else dims[axis]))
        self.coeffs = np.ones(self.n if dims is None else dims[axis])
        self.radius = radius
        self.dims = dims
        self.axis = axis
        self.otheraxis = 1 if axis == 0 else 0
        self.maxiter = maxiter
        self.ftol = ftol
        self.xtol = xtol
        self.call = call

    def __call__(self, x):
        return 0

    def prox(self, x, tau):
        x = x.reshape(self.dims)
        if self.axis == 0:
            x = x.T

        num_threads_per_blocks = 1024
        num_blocks = (x.shape[0] + num_threads_per_blocks - 1) // num_threads_per_blocks
        y = np.empty_like(x)
        # y = cuda.device_array_like(x)

        simplex_jit_gpu[num_blocks, num_threads_per_blocks](x, self.coeffs, self.radius, 0, 10000000000, self.maxiter,
                                                            self.ftol, self.xtol, y)

        if self.axis == 0:
            y = y.T
        return y.ravel()


def Simplex_gpu(n, radius, dims=None, axis=-1, maxiter=100,
                ftol=1e-8, xtol=1e-8, call=True):
    s = _Simplex_numba_gpu(n, radius, dims=dims, axis=axis,
                           maxiter=maxiter, ftol=ftol, xtol=xtol, call=call)
    return s