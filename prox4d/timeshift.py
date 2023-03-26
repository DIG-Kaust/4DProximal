import numpy as np
from pylops import FirstDerivative, Laplacian, Diagonal
from pylops.optimization.leastsquares import regularized_inversion as RegularizedInversion
from scipy.interpolate import splrep, BSpline


def linearized_timeshift(b, m, dt, epsRs=1e3, outeriter=1, inneriter=5):
    """"
    Performs timeshift inversion of 2D or 3D seismic by linearizing the nonlinear time-shift equation.

    m:  monitor seismic
    b:  baseline seismic
    dt: seismic sampling rate
    """""
    if b.ndim == 2:
        nt, nx = b.shape
        shift_est = np.zeros((nx, nt))
        R = Laplacian(dims=(nx, nt), axes=(0,1))
        Fd = FirstDerivative(dims=(nt, nx), axis=0, sampling=dt)
    elif b.ndim == 3:
        nt, nx, ny = b.shape
        shift_est = np.zeros((ny, nx, nt))
        R = Laplacian(dims=(ny, nx, nt), axes=(0, 1, 2),
                      weights=(1, 1, 1), sampling=(1, 1, 1), edge=True)
        Fd = FirstDerivative(dims=(nt, nx, ny), axis=0, sampling=dt)
    t = np.arange(nt) * dt
    mshift = m.copy()

    for iiter in range(outeriter):
        # Data
        mbdiff = b.T.ravel() - mshift.T.ravel()

        if b.ndim == 2:
            deriv = (Fd * mshift.ravel()).reshape(nt, nx)
            # plt.imshow(deriv)
            # plt.colorbar()
            # plt.show()
            # Jabobian
            Dm = -Diagonal((deriv).T.ravel())
            shift_est += RegularizedInversion(Dm, mbdiff,[R], epsRs=[epsRs], dataregs=[-R * shift_est.ravel()],
                                              **dict(iter_lim=inneriter, damp=1e-5))[0].reshape(nx, nt)
            # Interpolate baseline to current time shift estimate
            iavas = (t[np.newaxis, :] - shift_est) / dt
            for i in range(nx):
                # t_, c_, k_ = splrep(np.arange(-3, nt + 3), np.pad(m[:, i], (3, 3), 'edge'), s=0, k=3)
                t_, c_, k_ = splrep(np.arange(nt), m[:, i], s=0, k=3)
                spline = BSpline(t_, c_, k_, extrapolate=True)
                mshift[:, i] = spline(iavas[i, :])
            # print('new')
            # SI1op = BlockDiag([Interp(nt, i, kind='sinc', dtype='float64')[0] for i in iavas])
            # mshift = (SI1op * m.T.ravel()).reshape(nt, nx)


        elif b.ndim == 3:
            deriv = (Fd * mshift.ravel()).reshape(nt, nx, ny)

            # Jabobian
            Dm = -Diagonal((deriv).T.ravel())
            shift_est += RegularizedInversion(Dm,  mbdiff, [R], epsRs=[epsRs], dataregs=[-R * shift_est.ravel()],
                                              **dict(iter_lim=inneriter, damp=1e-5))[0].reshape(ny, nx, nt)
            # Interpolate baseline to current time shift estimate
            iavas = (t[np.newaxis, np.newaxis, :] - shift_est) / dt
            for i in range(nx):
                for j in range(ny):
                    t_, c_, k_ = splrep(np.arange(-3, nt + 3), np.pad(m[:, i, j], (3, 3), 'edge'), s=0, k=2)
                    spline = BSpline(t_, c_, k_, extrapolate=False)
                    mshift[:, i, j] = spline(iavas[j, i, :])

    return -shift_est, mshift
