import matplotlib.pyplot as plt
from pylops.avo.poststack import *
from pylops.optimization.sparsity import *
from pylops.basicoperators import VStack as VStacklop
from pyproximal.proximal import *
from pyproximal.optimization.segmentation import *
from pylops import FirstDerivative, Laplacian, Diagonal
from pylops.basicoperators import HStack as HStacklop
from pylops.basicoperators import Zero, Block
from prox4d.utils_ import *


def jis_4D(d_base, d_monitor, mback, cl, Op_base, Op_mon, model_tv,
                                     alpha, beta, delta, tau, mu, alpha_dif=None,
                                     niter=4, l2niter=20, pdniter=100,
                                     segmentniter=10, bisectniter=30, tolstop=0.,
                                     mtrue=None, plotflag=True, show=False):
    r"""Joint inversion-segmentation in 4D with Primal-Dual solver

    Parameters
    ----------
    d_base : :obj:`np.ndarray`
        Base Seimic Data (must have 2 or 3 dimensions with depth/time along first axis)
    d_monitor : :obj:`np.ndarray`
        Monitor Seimic Data (must have 2 or 3 dimensions with depth/time along first axis)
    mback : :obj:`np.ndarray`
        Background model (must have 2 or 3 dimensions with depth/time along
        first axis)
    cl : :obj:`numpy.ndarray`
        Classes
    Op_base : :obj:`pylops.avo.poststack.PoststackLinearModelling`
        Modelling operator baseline seismic
    Op_mon : :obj:`pylops.avo.poststack.PoststackLinearModelling`
        Modelling operator monitor seismic
    model_TV : :obj:`str`
        Type of TV regularization of the model ('anisotropic' or 'isotropic')
    alpha : :obj:`float`
        Scaling factor of the TV regularization of the model
    alpha_dif : :obj:`float`
        Scaling factor of the TV regularization of the monitor-baseline difference
    beta : :obj:`float`
        Scaling factor of the TV regularization of the segmentation
    delta : :obj:`float`
        Positive scalar weight of the segmentation misfit term
    tau : :obj:`float`
        Stepsize of subgradient of :math:`f`
    mu : :obj:`float`
        Stepsize of subgradient of :math:`g^*`
    niter : :obj:`int`, optional
        Number of iterations of joint scheme
    l2niter : :obj:`int`, optional
        Number of iterations of l2 proximal
    pdniter : :obj:`int`, optional
        Number of iterations of Primal-Dual solver
    segmentniter : :obj:`int`, optional
        Number of iterations of Segmentation solve
    bisectniter : :obj:`int`, optional
        Number of iterations of bisection used in the simplex proximal
    tolstop : :obj:`float`, optional
        Stopping criterion based on the segmentation update
    mtrue : :obj:`np.ndarray`, optional
        True model (must have 2 or 3 dimensions with depth/time along
        first axis). When available use to compute metrics
    plotflag : :obj:`bool`, optional
        Display intermediate steps
    show : :obj:`bool`, optional
        Print solvers iterations log

    Returns
    -------
    minv : :obj:`numpy.ndarray`
        Inverted model.
    v : :obj:`numpy.ndarray`
        Classes probabilities.
    vcl : :obj:`numpy.ndarray`
        Estimated classes.
    rre : :obj:`list`
        RRE metric through iterations (only if ``mtrue`` is provided)
    snr : :obj:`list`
        PSNR metric through iterations (only if ``mtrue`` is provided)
    minv_hist : :obj:`numpy.ndarray`
        History of inverted model through iterations
    v_hist : :obj:`numpy.ndarray`
        History of classes probabilities through iterations

    """

    d = np.hstack([d_base.ravel(), d_monitor.ravel()])
    mshape = mback.shape  # original model shape
    mprod = mback.size
    # background model augmented (vertical stack baseline | monitor)
    m_aug_back = np.hstack([mback.ravel(), mback.ravel()])
    ncl = len(cl)
    Op = BlockDiag([Op_base, Op_mon])
    print('Working with alpha=%f,  beta=%f,  delta=%f' % (alpha, beta, delta))

    # TV regularization term
    if model_tv == 'anisotropic':
        Dop = Gradient(dims=mshape, edge=True, dtype=Op_base.dtype, kind='forward')
        Dop = BlockDiag([Dop, Dop])
        l1 = L1(sigma=alpha)

    elif model_tv == 'isotropic':
        if len(mshape) == 3:
            Dzop = FirstDerivative(dims=mshape, axis=0, edge=True, dtype=Op_base.dtype, kind='forward')
            Dxop = FirstDerivative(dims=mshape, axis=1, edge=True, dtype=Op_base.dtype, kind='forward')
            Dyop = FirstDerivative(dims=mshape, axis=2, edge=True, dtype=Op_base.dtype, kind='forward')
            Zop = Zero(mprod, dtype='float64')
            Dop = Block([[Dzop, Zop],
                         [Zop, Dzop],
                         [Dxop, Zop],
                         [Zop, Dxop],
                         [Dyop, Zop],
                         [Zop, Dyop]])
            l1 = L21(ndim=len(mshape), sigma=alpha)

        if len(mshape) == 2:
            Dzop = FirstDerivative(dims=mshape, axis=0, edge=True, dtype=Op_base.dtype, kind='forward')
            Dxop = FirstDerivative(dims=mshape, axis=1, edge=True, dtype=Op_base.dtype, kind='forward')
            Zop = Zero(mprod, dtype='float64')
            Dop = Block([[Dzop, Zop],
                         [Zop, Dzop],
                         [Dxop, Zop],
                         [Zop, Dxop]])
            l1 = L21(ndim=len(mshape), sigma=alpha)

    elif model_tv == 'isotropic_dif' and alpha_dif:
        if len(mshape) == 3:
            Dzop = FirstDerivative(dims=mshape, axis=0, edge=True, dtype=Op_base.dtype, kind='forward')
            Dxop = FirstDerivative(dims=mshape, axis=1, edge=True, dtype=Op_base.dtype, kind='forward')
            Dyop = FirstDerivative(dims=mshape, axis=2, edge=True, dtype=Op_base.dtype, kind='forward')
            Zop = Zero(mprod, dtype='float64')

            Dop = Block([[Dzop, Zop],
                         [Zop, Dzop],
                         [-alpha_dif * Dzop, alpha_dif * Dzop],
                         [Dxop, Zop],
                         [Zop, Dxop],
                         [alpha_dif * Dxop, -alpha_dif * Dxop],
                         [Dyop, Zop],
                         [Zop, Dyop],
                         [alpha_dif * Dyop, -alpha_dif * Dyop]])
            l1 = L21(ndim=len(mshape), sigma=alpha)

        elif len(mshape) == 3:
            Dzop = FirstDerivative(dims=mshape, axis=0, edge=True, dtype=Op_base.dtype, kind='forward')
            Dxop = FirstDerivative(dims=mshape, axis=1, edge=True, dtype=Op_base.dtype, kind='forward')
            Zop = Zero(mprod, dtype='float64')
            Dop = Block([[Dzop, Zop],
                         [Zop, Dzop],
                         [-alpha_dif * Dzop, alpha_dif * Dzop],
                         [Dxop, Zop],
                         [Zop, Dxop],
                         [alpha_dif * Dxop, -alpha_dif * Dxop]])
            l1 = L21(ndim=len(mshape), sigma=alpha)

    p = np.zeros(m_aug_back.size)
    q = np.zeros(ncl * mprod)
    v = np.zeros(ncl * mprod)
    minv = m_aug_back.copy().ravel()
    minv_hist = []
    v_hist = []

    rre = snr = None
    if mtrue is not None:
        rre = np.zeros((niter, 2))
        snr = np.zeros((niter, 2))

    if plotflag:
        fig, axs = plt.subplots(2, niter, figsize=(4 * niter, 10))

    for iiter in range(niter):
        print('Iteration %d...' % iiter)
        minvold = minv.copy()
        vold = v.copy()

        #############
        # Inversion #
        #############
        if iiter == 0:
            # define misfit term
            l2 = L2(Op=Op, b=d.ravel(), niter=l2niter, warm=True)
            # solve
            minv = PrimalDual(l2, l1, Dop, x0=m_aug_back.ravel(),
                              tau=tau, mu=mu, theta=1., niter=pdniter,
                              show=show)
            minv = np.real(minv)

            # Update p
            l2_grad = L2(Op=Op, b=d.ravel())
            dp = (1. / alpha) * l2_grad.grad(minv)
            p -= np.real(dp)

        else:
            # define misfit term
            v = v.reshape((mprod, ncl))
            DD = VStacklop([Diagonal(np.sqrt(2. * delta) * np.sqrt(v[:, icl]))
                            for icl in range(ncl)])
            DD2 = HStacklop([-DD] + [DD])
            L1op = VStacklop([Op] + [DD2])
            d1 = np.hstack([d.ravel(), (np.sqrt(2. * delta) * (np.sqrt(v) * cl[np.newaxis, :]).T).ravel()])
            l2 = L2(Op=L1op, b=d1, niter=l2niter, warm=True, q=p, alpha=-alpha)

            # solve
            minv = PrimalDual(l2, l1, Dop, x0=m_aug_back.ravel(),
                              tau=tau, mu=mu, theta=1., niter=pdniter,
                              show=show)
            minv = np.real(minv)

            # Update p
            l2_grad = L2(Op=L1op, b=d1)
            dp = l2_grad.grad(minv)
            dm = minv.ravel()[minv.ravel().shape[0]//2:] - minv.ravel()[:minv.ravel().shape[0]//2]
            dp2 = np.sum([v[:, icl] * (dm-cl[icl]) for icl in range(ncl)], axis=0)
            dp2 = np.hstack([dp2*-1, dp2])
            p -= (1. / alpha) * (np.real(dp+2*delta*dp2))

        minv_hist.append(minv)
        base = np.exp(minv[:minv.shape[0] // 2])
        moni = np.exp(minv[minv.shape[0] // 2:])
        dif = ((moni - base) / moni).reshape(mshape)

        if plotflag:
            if len(mshape) == 3:
                axs[0, iiter].imshow(dif[:, :, 2], 'gray')
                axs[0, iiter].axis('tight')
            elif len(mshape) == 2:
                axs[0, iiter].imshow(dif, 'gray')
                axs[0, iiter].axis('tight')

        ################
        # Segmentation #
        ################
        v, vcl = Segment(dif, cl, 2 * delta, 2 * beta, z=-beta * q,
                         niter=segmentniter, callback=None, show=show,
                         kwargs_simplex=dict(engine='numba',
                                             maxiter=bisectniter, call=False))
        v_hist.append(v)

        # Update q
        dq = (delta / beta) * ((dif.ravel() - cl[:, np.newaxis]) ** 2).ravel()
        q -= dq

        if plotflag:
            if len(mshape) == 3:
                axs[1, iiter].imshow(vcl.reshape(mshape)[:, :, 2], 'gray')
                axs[1, iiter].axis('tight')
            elif len(mshape) == 2:
                axs[1, iiter].imshow(vcl.reshape(mshape), 'gray')
                axs[1, iiter].axis('tight')

        # Monitor cost functions
        print('f=', L2(Op=Op, b=d.ravel())(minv))
        print('||v-v_old||_2=', np.linalg.norm(v.ravel() - vold.ravel()))
        print('||m-m_old||_2=', np.linalg.norm(minv.ravel() - minvold.ravel()))

        # Monitor quality of reconstruction
        if mtrue is not None:
            rre[iiter, 0] = RRE(mtrue[0].ravel(), minv[:minv.shape[0] // 2].ravel())
            rre[iiter, 1] = RRE(mtrue[1].ravel(), minv[minv.shape[0] // 2:].ravel())
            snr[iiter, 0] = SNR(mtrue[0].ravel(), minv[:minv.shape[0] // 2].ravel())
            snr[iiter, 1] = SNR(mtrue[0].ravel(), minv[minv.shape[0] // 2:].ravel())

        # Check stopping criterion
        if np.linalg.norm(v.ravel() - vold.ravel()) < tolstop:
            break

    plt.show()

    return minv, v, vcl, rre, snr, minv_hist, v_hist
