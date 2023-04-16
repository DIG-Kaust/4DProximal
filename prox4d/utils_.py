import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import matplotlib.gridspec as gridspec


def callback0(x, xtrue, xhist, errhist):
    xhist.append(x)
    errhist.append(np.linalg.norm(x - xtrue)/np.linalg.norm(xtrue))
    
def callback(x, y, xtrue, xhist, yhist, errhistx, errhisty):
    xhist.append(x)
    yhist.append(y)
    errhistx.append(np.linalg.norm(x - xtrue) / np.linalg.norm(xtrue))
    errhisty.append(np.linalg.norm(y - xtrue) / np.linalg.norm(xtrue))


def callback(x, xhist):
    xhist.append(x)


def RRE(x, xinv):
    return np.linalg.norm(x - xinv) / np.linalg.norm(x)


def SNR(xref, xest):
    xrefv = np.mean(np.abs(xref) ** 2)
    return 10. * np.log10(xrefv / np.mean(np.abs(xref - xest)**2))


def plotter_4D(b, m, dt=1, type='impedance', perc=1, dif_scale=0.01, ref=None,
               height=4, width=15, mtrue=None, cmap='seismic_r', vline=None):
    if ref is not None:
        vmin, vmax = np.percentile(ref, [perc, 100 - perc])
    else:
        vmin, vmax = np.percentile(b, [perc, 100 - perc])

    fig = plt.figure(figsize=(width, height))
    if type == 'seismic':
        gs = gridspec.GridSpec(5, 4, width_ratios=(1, 1, 1, .05), height_ratios=(.1, .5, .5, .5, .1),
                               left=0.1, right=0.9, bottom=0.1, top=0.9,
                               wspace=0.05, hspace=0.05)
        ax0 = fig.add_subplot(gs[:, 0])
        base = ax0.imshow(b, vmin=-vmax, vmax=vmax, cmap=cmap, extent=[0, m.shape[1], m.shape[0] * dt, 0])
        ax0.set_ylabel('TWT $[s]$')
        ax0.set_title('a) Baseline Seismic')
        ax0.axis('tight');
        ax1 = fig.add_subplot(gs[:, 1])
        ax1.imshow(m, vmin=-vmax, vmax=vmax, cmap=cmap)
        ax1.set_yticklabels([])
        ax1.set_title('b) Monitor Seismic')
        ax1.axis('tight');
        ax2 = fig.add_subplot(gs[:, 2])
        ax2.imshow(m - b, vmin=-vmax, vmax=vmax, cmap=cmap)
        ax2.set_yticklabels([])
        ax2.set_title('c) Monitor - Baseline')
        ax2.axis('tight')
        ax3 = fig.add_subplot(gs[2, 3])
        ax3.set_title('Amplitude', loc='left')
        Colorbar(ax=ax3, mappable=base)

    if type == 'impedance':
        gs = gridspec.GridSpec(5, 4, width_ratios=(1, 1, 1, .05), height_ratios=(.1, .5, .3, .5, .1),
                               left=0.1, right=0.9, bottom=0.1, top=0.9,
                               wspace=0.05, hspace=0.05)
        ax0 = fig.add_subplot(gs[:, 0])
        base = ax0.imshow(b, vmin=vmin, vmax=vmax, cmap='terrain', extent=[0, m.shape[1], m.shape[0] * dt, 0])
        ax0.set_ylabel('TWT $[s]$')
        ax0.set_title('a) Baseline')
        ax0.axis('tight');
        ax1 = fig.add_subplot(gs[:, 1])
        mon = ax1.imshow(m, vmin=vmin, vmax=vmax, cmap='terrain')
        ax1.set_yticklabels([])
        ax1.set_title('b) Monitor')
        ax1.axis('tight');
        ax2 = fig.add_subplot(gs[:, 2])
        dif = ax2.imshow((m - b) / m, vmin=-dif_scale, vmax=dif_scale, cmap='seismic_r')
        ax2.set_yticklabels([])
        ax2.set_title('c) Monitor - Baseline')
        ax2.axis('tight')
        if vline is not None:
            plt.vlines(vline, 0, m.shape[0], 'k')
        ax3 = fig.add_subplot(gs[1, 3])
        ax3.set_title('Impedance \n $[m/s*g/cm^3]$', loc='left')
        Colorbar(ax=ax3, mappable=base)
        ax3 = fig.add_subplot(gs[3, 3])
        ax3.set_title('Difference \n [%]', loc='left')
        Colorbar(ax=ax3, mappable=dif)

        if mtrue is not None:
            rre1 = RRE(mtrue[0], b)
            snr1 = SNR(mtrue[0], b)
            rre2 = RRE(mtrue[1], m)
            snr2 = SNR(mtrue[1], m)
            rre3 = RRE(mtrue[1]-mtrue[0], m-b)
            # snr3 = SNR(mtrue[1]-mtrue[0], m-b)
            ax0.set_title('a) Baseline \n RRE = %.2f ' % rre1 + 'SNR = %.2f' % snr1)
            ax1.set_title('b) Monitor \n RRE = %.2f ' % rre2 + 'SNR = %.2f' % snr2)
            ax2.set_title('c) Monitor - Difference \n RRE = %.2f ' % rre3)




def plotter_4D_comparison(b1, m1, b2, m2, dt=1, perc=1, height=4, width=12, dif_scale=0.01):

    vmin, vmax = np.percentile(b1, [perc, 100 - perc])

    fig = plt.figure(figsize=(width, height))

    gs = gridspec.GridSpec(5, 5, width_ratios=(1, 1, 1, 1., .05), height_ratios=(.2, .5, .4, .5, .1),
                           left=0.1, right=0.9, bottom=0.1, top=0.9,
                           wspace=0.05, hspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])
    mon = ax0.imshow(m1, vmin=vmin, vmax=vmax, cmap='terrain', extent=[0, m1.shape[1], m1.shape[0] * dt, 0])
    ax0.set_ylabel('TWT $[s]$')
    ax0.set_title('L$_2$ Reg Inversion \n Monitor')
    ax0.axis('tight');
    ax1 = fig.add_subplot(gs[:, 1])
    dif = ax1.imshow((m1 - b1) / m1, vmin=-dif_scale, vmax=dif_scale, cmap='seismic_r')
    ax1.set_yticklabels([])
    ax1.set_title('L$_2$ Reg Inversion \n Monitor - Baseline')
    ax1.axis('tight');
    ax2 = fig.add_subplot(gs[:, 2])
    ax2.imshow(m2, vmin=vmin, vmax=vmax, cmap='terrain', extent=[0, m1.shape[1], m1.shape[0] * dt, 0])
    ax2.set_title('JIS \n Monitor')
    ax2.axis('tight');
    ax2.set_yticklabels([])
    ax3 = fig.add_subplot(gs[:, 3])
    ax3.imshow((m2 - b2) / m2, vmin=-dif_scale, vmax=dif_scale, cmap='seismic_r')
    ax3.set_yticklabels([])
    ax3.set_title('JIS  \n Monitor - Baseline')
    ax3.axis('tight');
    ax4 = fig.add_subplot(gs[1, 4])
    ax4.set_title('Impedance \n $[m/s*g/cm^3]$', loc='left')
    Colorbar(ax=ax4, mappable=mon)
    ax4 = fig.add_subplot(gs[3, 4])
    ax4.set_title('Difference \n [%]', loc='left')
    Colorbar(ax=ax4, mappable=dif)

def plotter_4D_seg(V, nt, nx, dt, width=15, height=4):
    vmin, vmax = 0, 1

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(5, 4, width_ratios=(1, 1, 1, .05), height_ratios=(.1, .5, .5, .5, .1),
                           left=0.1, right=0.9, bottom=0.1, top=0.9,
                           wspace=0.05, hspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])
    base = ax0.imshow(V[:, 0].reshape(nt, nx), vmin=vmin, vmax=vmax, cmap='inferno', extent=[0, nx, nt * dt, 0])
    ax0.set_ylabel('TWT $[s]$')
    ax0.set_title('Class 1')
    ax0.axis('tight');
    ax1 = fig.add_subplot(gs[:, 1])
    ax1.imshow(V[:, 1].reshape(nt, nx), vmin=vmin, vmax=vmax, cmap='inferno', extent=[0, nx, nt * dt, 0])
    ax1.set_title('Class 2')
    ax1.axis('tight');
    ax1.set_yticklabels([])
    ax2 = fig.add_subplot(gs[:, 2])
    ax2.imshow(V[:, 2].reshape(nt, nx), vmin=vmin, vmax=vmax, cmap='inferno', extent=[0, nx, nt * dt, 0])
    ax2.set_title('Class 3')
    ax2.axis('tight');
    ax2.set_yticklabels([])
    ax3 = fig.add_subplot(gs[2, 3])
    ax3.set_title('Probability', loc='left')
    Colorbar(ax=ax3, mappable=base)


def plotter_timeshift(b, m, shift, mshift, dt=1, perc=1, height=5, width=15, dif_scale=1., alpha=0.7, years=[1994, 2001]):

    vmin, vmax = np.percentile(b, [perc, 100 - perc])

    fig = plt.figure(figsize=(width, height))

    gs = gridspec.GridSpec(5, 5, width_ratios=(1, 1, 1, 1., .05), height_ratios=(.1, .5, .5, .5, .1),
                           left=0.1, right=0.9, bottom=0.1, top=0.9,
                           wspace=0.05, hspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])
    base = ax0.imshow(b, vmin=-vmax, vmax=vmax, cmap='gray_r', extent=[0, m.shape[1], m.shape[0] * dt, 0])
    ax0.set_ylabel('TWT $[s]$')
    ax0.set_title('Baseline (%d)'% years[0])
    ax0.axis('tight');
    ax1 = fig.add_subplot(gs[:, 1])
    ax1.imshow(m, vmin=-vmax, vmax=vmax, cmap='gray_r')
    ts = ax1.imshow(shift*1000, vmin=-10*dif_scale, vmax=10*dif_scale, cmap='seismic_r', alpha=alpha)
    ax1.set_yticklabels([])
    ax1.set_title('Monitor (%d)'% years[1])
    ax1.axis('tight');
    ax2 = fig.add_subplot(gs[:, 2])
    ax2.imshow(m - b, vmin=-vmax, vmax=vmax, cmap='gray_r')
    ax2.set_yticklabels([])
    ax2.set_title('Monitor - Baseline')
    ax2.axis('tight')
    ax3 = fig.add_subplot(gs[:, 3])
    ax3.imshow(mshift - b, vmin=-vmax, vmax=vmax, cmap='gray_r')
    ax3.set_yticklabels([])
    ax3.set_title('Monitor(s) - Baseline')
    ax3.axis('tight')
    ax4 = fig.add_subplot(gs[1, 4])
    ax4.set_title('Amplitude', loc='left')
    Colorbar(ax=ax4, mappable=base)
    ax4 = fig.add_subplot(gs[3, 4])
    ax4.set_title('Time-shift \n [ms]', loc='left')
    Colorbar(ax=ax4, mappable=ts)

