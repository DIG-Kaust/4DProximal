import matplotlib.pyplot as plt
from pylops.avo.poststack import *
from pylops.optimization.sparsity import *
from pylops.basicoperators import VStack as VStacklop
from pyproximal.proximal import *
from pyproximal.optimization.segmentation import *
from pylops.basicoperators import HStack as HStacklop
from pylops.basicoperators import Zero, Block
from prox4d.segment_gpu import Segment_gpu
from pylops import Diagonal
from pnpseismic.DnCNN_models import *
from pnpseismic.Denoising_scalings import *
from pnpseismic.PnP_seismic import *
from pylops import Diagonal, Identity

def jis_4D_pnp(d_base, d_monitor, mback, cl, Op_base, Op_mon, model_tv,
                                     alpha, beta, delta, tau, mu, sigma,
                                     niter=4, l2niter=20, pdniter=100,
                                     segmentniter=10, bisectniter=30, tolstop=0.,
                                     mtrue=None, plotflag=True, show=False, gpu=False):

 

    d = np.hstack([d_base.ravel(), d_monitor.ravel()])
    mshape = mback.shape  # original model shape
    mprod = mback.size
    # background model augmented (vertical stack baseline | monitor)
    m_aug_back = np.hstack([mback.ravel(), mback.ravel()])
    ncl = len(cl)

    print('Working with alpha=%f,  beta=%f,  delta=%f' % (alpha, beta, delta))


    Op = BlockDiag([Op_base, Op_mon])

    p = np.zeros(m_aug_back.size)
    q = np.zeros(ncl * mprod)
    v = np.zeros(ncl * mprod)
    minv = m_aug_back.copy().ravel()
    minv_hist = []
    v_hist = []

    
    #pnp
    Iop = Identity(d.size)
    model_02 = UNetRes(in_nc=2)
    model_02.load_state_dict(torch.load('../../models/drunet_gray.pth'))
    model_02.eval().to(cuda0);

#     sigma = .001
    denoiser_dn = lambda x, mu_: scaling_DRUNET(model_02, torch.from_numpy(x).float(), mu_, sigma, verb=False)

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
            minv = PlugAndPlay_PrimalDual(l2, denoiser_dn, Iop, np.vstack([mback,mback]).shape,
                                               tau=tau, x0=m_aug_back.ravel(), mu=mu,
                                               niter=pdniter, show=show)
            
            minv = np.real(minv)
            # dinv = Op * minv

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

            
            minv = PlugAndPlay_PrimalDual(l2, denoiser_dn, Iop, np.vstack([mback,mback]).shape,
                                               tau=tau, x0=m_aug_back.ravel(), mu=mu,
                                               niter=pdniter, show=show)
            
            minv = np.real(minv)
            # dinv = Op * minv

            # Update p
            l2_grad = L2(Op=L1op, b=d1)
            dp = (1. / alpha) * l2_grad.grad(minv)

            p -= np.real(dp)

        minv_hist.append(minv)
        base = np.exp(minv[:minv.shape[0] // 2])
        moni = np.exp(minv[minv.shape[0] // 2:])
        dif = ((moni - base) / moni).reshape(mshape)

        if plotflag:
            if len(mshape) == 3:
                axs[0, iiter].imshow(dif[:, :, 10], 'gray')
                axs[0, iiter].axis('tight')
            elif len(mshape) == 2:
                axs[0, iiter].imshow(dif, 'gray')
                axs[0, iiter].axis('tight')

        ################
        # Segmentation #
        ################
        if gpu:
            print('Running on GPU')
            v, vcl = Segment_gpu(dif, cl, 2 * delta, 2 * beta, z=-beta * q,
                                 niter=segmentniter, callback=None, show=show,
                                 kwargs_simplex=dict(maxiter=bisectniter, call=False))
        else:
            v, vcl = Segment(dif, cl, 2 * delta, 2 * beta, z=-beta * q,
                             niter=segmentniter, callback=None, show=show,
                             kwargs_simplex=dict(engine='numba',
                                                 maxiter=bisectniter, call=False))
        # xtol=1e-3, ftol=1e-3))
        v_hist.append(v)

        # Update q
        dq = (delta / beta) * ((dif.ravel() - cl[:, np.newaxis]) ** 2).ravel()
        q -= dq

        if plotflag:
            if len(mshape) == 3:
                axs[1, iiter].imshow(vcl.reshape(mshape)[:, :, 10], 'gray')
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