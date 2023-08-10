import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain


def dat2dim(dim1, dim2, dat, dimnm_dim1, unit_dim1, dimnm_dim2, unit_dim2, valnm, unit_val,
            lim_dim1=False, lim_dim2=False, datlim=False, cmap='jet'):
    ddim1 = dim1[1] - dim1[0]
    ddim2 = dim2[1] - dim2[0]

    label = f'{valnm} [{unit_val}]'

    if datlim:
        bottom, top = datlim
    else:
        top, bottom = maxAndMin(dat)
    fig, ax = plt.subplots()
    pcf = ax.pcolorfast((dim1.min() - 0.5 * ddim1, dim1.max() + 0.5 * ddim1),
                        (dim2.min() - 0.5 * ddim2, dim2.max() + 0.5 * ddim2),
                        dat.T, cmap=cmap, vmin=bottom, vmax=top)
    fig.colorbar(pcf, ax=ax, label=label)
    if lim_dim1:
        ax.set_xlim(lim_dim1)
    else:
        ax.set_xlim(dim1.min() - 0.5 * ddim1, dim1.max() + 0.5 * ddim1)
    if lim_dim2:
        ax.set_ylim(lim_dim2)
    else:
        ax.set_ylim(dim2.min() - 0.5 * ddim2, dim2.max() + 0.5 * ddim2)

    ax.set_xlabel(f'{dimnm_dim1} [{unit_dim1}]')
    ax.set_ylabel(f'{dimnm_dim2} [{unit_dim2}]')

    return fig, ax


def maxAndMin(dat):
    max = np.max([np.nanmax(dat) * 1.2, 0])
    min = np.min([np.nanmin(dat) * 1.2, 0])
    return max, min


def set(context, style):
    sns.set_context(context)  # 'talk', 'paper', ...
    sns.set_style(style)   # 'ticks', ...
    return


def check(duration):
    if duration == 0:
        plt.show()
    else:
        plt.pause(duration)
        plt.clf()
        plt.close()

    return


def close(fig):
    plt.clf()
    plt.close(fig)
    return

def caption(fig, fig_title, hspace=0.1, wspace=0.1):
    fig.suptitle(fig_title)
    fig.tight_layout()
    plt.subplots_adjust(hspace=hspace)
    plt.subplots_adjust(wspace=wspace)
    return fig

def capsave(fig, fig_title, fnm, path):
    fig.text(1, 0.01, fnm, ha='right', color='grey', size='xx-small')
    fig.savefig(path)
    return fig


def errorbar_single(x, y, yerr, xlabel, ylabel, xlim=False, ylim=False,
                    color='black', lw=1, fmt='.', label=False):

    fig, ax = plt.subplots()
    if label:
        ax.errorbar(x, y, yerr,
                    color=color, ecolor='grey',
                    lw=lw, elinewidth=lw, fmt=fmt, label=label)
    else:
        ax.errorbar(x, y, yerr,
                    color=color, ecolor='grey',
                    lw=lw, elinewidth=lw, fmt=fmt)
    # ax.axhline(0, color='grey', ls='--')
    # ax.axvline(0, color='grey', ls='--')
    if xlim:
        ax.set_xlim(xlim)
    else:
        xmax, xmin = maxAndMin(x)
        ax.set_xlim((xmin, xmax))

    if ylim:
        ax.set_ylim(ylim)
    else:
        if xlim:
            xmin, xmax = xlim
            idxs = np.where(~((x >= xmin) & (x <= xmax)))
            y[idxs] = np.nan
            ymax, ymin = maxAndMin(y)
        else:
            ymax, ymin = maxAndMin(y)
        ax.set_ylim((ymin, ymax))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return fig, ax


def errorbar_hue(li_x, li_y, li_lab_hue, xlabel, ylabel, li_xerr=None, li_yerr=None, xlim=None, ylim=None, fmt='.', li_huecolor=None, figsize=(8, 4)):

    if li_xerr == None:
        li_xerr = [0]*len(li_x)
    if li_yerr == None:
        li_yerr = [0] * len(li_y)

    fig, axs = plt.subplots(figsize=figsize)
    if li_huecolor != None:
        for j in range(len(li_x)):
            axs.errorbar(li_x[j], li_y[j], li_yerr[j], li_xerr[j],
                         color=li_huecolor[j], ecolor='grey', fmt=fmt,
                         mfc=li_huecolor[j], mew=0.1, lw=1)
    else:
        for j in range(len(li_x)):
            axs.errorbar(li_x[j], li_y[j], li_yerr[j], li_xerr[j],
                         ecolor='grey', fmt=fmt,
                         mew=0.1, lw=1)
    axs.legend(li_lab_hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    axs.axhline(0, color='grey', ls='--', label=None)
    axs.axvline(0, color='grey', ls='--', label=None)
    if xlim != None:
        axs.set_xlim(xlim)
    if ylim != None:
        axs.set_ylim(ylim)
    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)

    return fig, axs


def errorbar_vert(li_x, li_y, li_yerr, xlabel, li_lab_y, xlim=False, ylims=False, color='black', lw=1, fmt='-', includezero=False):

    fig, axs = plt.subplots(len(li_lab_y))
    for i in range(len(li_lab_y)):
        axs[i].errorbar(li_x[i], li_y[i], li_yerr[i],
                         color=color, ecolor='grey',
                         lw=lw, elinewidth=lw, fmt=fmt)
        if includezero:
            axs[i].axhline(0, color='grey', ls='--')
            axs[i].axvline(0, color='grey', ls='--')
        if xlim:
            axs[i].set_xlim(xlim)
        if ylims:
            axs[i].set_ylim(ylims)
        axs[i].set_ylabel(li_lab_y[i])
        if i == len(li_lab_y) - 1:
            axs[i].set_xlabel(xlabel)
        else:
            axs[i].set_xticklabels([])

    return fig, axs


def errorbar_vert_hue(li_x, li_y, li_yerr, li_huecolor, li_lw, li_lab_y, li_lab_hue, xlabel, xlim=None, ylims=None, fmt='.'):

    fig, axs = plt.subplots(len(li_lab_y))
    for i in range(len(li_lab_y)):
        for j in range(len(li_lab_hue[i])):
            axs[i].errorbar(li_x[i][j], li_y[i][j], li_yerr[i][j],
                             color=li_huecolor[j], ecolor='grey',
                             lw=li_lw[j], elinewidth=li_lw[j], label=li_lab_hue[i][j], fmt=fmt)
            axs[i].legend()
        axs[i].axhline(0, color='grey', ls='--', label=None)
        # axs[i].axvline(0, color='grey', ls='--')
        if xlim != None:
            axs[i].set_xlim(xlim)
        if ylims != None:
            axs[i].set_ylim(ylims)
        axs[i].set_ylabel(li_lab_y[i])
        if i == len(li_lab_y) - 1:
            axs[i].set_xlabel(xlabel)
        else:
            axs[i].set_xticklabels([])

    return fig, axs


def errorbar_horz_hue(li_x, li_y, li_yerr, li_huecolor, li_lw, li_lab_y, li_lab_hue, xlabel, xlim=None, ylims=None, fmt='.'):
    # If fmt == 'fill', fill error regions with hue colors.

    fig, axs = plt.subplots(1, len(li_lab_y))
    for i in range(len(li_lab_y)):
        for j in range(len(li_lab_hue[i])):
            if fmt == 'fill':
                axs[i].plot(li_x[i][j], li_y[i][j], color=li_huecolor[j], label=li_lab_hue[i][j])
                axs[i].fill_between(li_x[i][j], li_y[i][j] - li_yerr[i][j],
                                    li_y[i][j] + li_yerr[i][j],
                                    color=li_huecolor[j], alpha=0.5, lw=0)
            else:
                axs[i].errorbar(li_x[i][j], li_y[i][j], li_yerr[i][j],
                                 color=li_huecolor[j], ecolor='grey',
                                 lw=li_lw[j], elinewidth=li_lw[j], label=li_lab_hue[i][j], fmt=fmt)
            axs[i].legend()
        axs[i].axhline(0, color='grey', ls='--', label=None)
        # axs[i].axvline(0, color='grey', ls='--')
        if xlim != None:
            axs[i].set_xlim(xlim)
        if ylims != None:
            axs[i].set_ylim(ylims)
        axs[i].set_xlabel(xlabel)
        if i != 0:
            axs[i].set_yticklabels([])
        axs[i].set_title(li_lab_y[i])

    return fig, axs


def errorbar_t_anyRows_hue(xList, yList, yErrList, yLabList, yHueLabList):
    fig, axs = plt.subplots(len(xList), 1)
    xMin = min([dat.min() for dat in list(chain.from_iterable(xList))])
    xMax = max([dat.max() for dat in list(chain.from_iterable(xList))])
    for ii, xDat in enumerate(xList):
        yDat = yList[ii]
        yErr = yErrList[ii]
        if isinstance(yDat, list):
            for jj, xDatDet in enumerate(xDat):
                yDatDet = yDat[jj]
                yErrDet = yErr[jj]
                axs[ii].errorbar(xDatDet, yDatDet, yErrDet, fmt='.', label=yHueLabList[ii][jj])
                axs[ii].legend()
        else:
            axs[ii].errorbar(xDat, yDat, yErr, fmt='.')
        axs[ii].set_ylabel(yLabList[ii])
        axs[ii].set_xlim(xMin, xMax)
        if ii != (len(xList) - 1):
            axs[ii].set_xticklabels([])
        else:
            axs[ii].set_xlabel('Time [s]')

    return fig, axs


def spectrogram_1s_fk(time, freq_k, flim_k, any, bottom, top, label, cmap):
    
    dt = time[1] - time[0]
    dfk = freq_k[1] - freq_k[0]

    fig, ax = plt.subplots()
    pcf = ax.pcolorfast((time.min() - 0.5*dt, time.max() + 0.5*dt), (freq_k.min() - dfk/2, freq_k.max() + dfk/2),
                         any, cmap=cmap, vmin=bottom, vmax=top)
    fig.colorbar(pcf, ax=ax, label=label)
    ax.set_ylim(0, flim_k)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [kHz]')
    ax.set_xlim(time.min() - 0.5*dt, time.max() + 0.5*dt)

    return fig, ax


def spectrogram_2s_fk(time, freq_k, flim_k, any, bottom, top, label, cmap):

    dt = time[1] - time[0]
    dfk = freq_k[1] - freq_k[0]

    fig, ax = plt.subplots()
    pcf = ax.pcolorfast((time.min() - 0.5 * dt, time.max() + 0.5 * dt),
                        (freq_k.min() - dfk / 2, freq_k.max() + dfk / 2),
                         any, cmap=cmap, vmin=bottom, vmax=top)
    fig.colorbar(pcf, ax=ax, label=label)
    ax.set_ylim(-flim_k, flim_k)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [kHz]')
    ax.set_xlim(time.min() - 0.5 * dt, time.max() + 0.5 * dt)

    return fig, ax


def ax0_spectrogram_2s_fk(time0, dat0, err0, label0, time, freq_k, flim_k, spec, bottom, top, label, cmap):

    dt = time[1] - time[0]
    dfk = freq_k[1] - freq_k[0]

    # fig, ax = plt.subplots()
    fig = plt.figure()
    widths = [8, 1]
    heights = [2, 5]
    gs = fig.add_gridspec(ncols=2, nrows=2, hspace=0,
                          width_ratios=widths, height_ratios=heights)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.errorbar(time0, dat0, err0, color='black', ecolor='grey')
    ax0.set_xlim(time.min() - 0.5 * dt, time.max() + 0.5 * dt)
    ax0.set_ylim(np.nanmin([0, np.nanmin(dat0)]), np.nanmax([0, np.nanmax(dat0)]))
    ax0.set_xticklabels([])
    ax0.set_ylabel(label0)

    ax1 = fig.add_subplot(gs[1, 0])
    cax1 = fig.add_subplot(gs[1, 1])
    pcf = ax1.pcolorfast((time.min() - 0.5 * dt, time.max() + 0.5 * dt),
                        (freq_k.min() - 0.5 * dfk, freq_k.max() + 0.5 * dfk),
                        spec, cmap=cmap, vmin=bottom, vmax=top)
    fig.colorbar(pcf, cax=cax1, label=label)
    ax1.set_xlim(time.min() - 0.5 * dt, time.max() + 0.5 * dt)
    ax1.set_ylim(-flim_k, flim_k)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [kHz]')

    return fig, ax0, ax1, cax1


def any_t_fk_2s(time, dt, freq_k, dfk, flim_k, any, bottom, top, label, cmap):

    fig, ax = plt.subplots()
    pcf = ax.pcolorfast((time.min(), time.max() + dt), (freq_k.min() - dfk/2, freq_k.max() + dfk/2),
                         any, cmap=cmap, vmin=bottom, vmax=top)
    fig.colorbar(pcf, ax=ax, label=label)
    ax.set_ylim(-flim_k, flim_k)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [kHz]')
    ax.set_xlim(time.min(), time.max() + dt)

    return fig, ax


def errorbar_fk_2s(freq_k, any, any_err, c='red', label=False, flim_k=False, level=False, fmt='-'): # level=(bottom, top)

    fig, ax = plt.subplots()
    ax.errorbar(freq_k, any, any_err, color=c, ecolor='black', fmt=fmt)
    if flim_k:
        ax.set_xlim(-flim_k, flim_k)
    if level:
        bottom, top = level
        ax.set_ylim(bottom, top)
    ax.set_xlabel('Frequency [kHz]')
    if label:
        ax.set_ylabel(label)

    return fig, ax


def errorbar_t_mul(time, huelabels, any, any_err, bottom, top, label, fmt, palette):

    fig, ax = plt.subplots()
    for i, huelabel in enumerate(huelabels):
        ax.errorbar(time, any[i], any_err[i], label=huelabel,
                    color=sns.color_palette(palette)[i], ecolor='grey', fmt=fmt)
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(bottom, top)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(label)
    ax.legend()

    return fig, ax


def errorbar2_t_mul(time, huelabels, any1, any1_err, any2, any2_err, range1, range2, label1, label2, fmt1, fmt2, palette):

    fig, axs = plt.subplots(2)
    for i, huelabel in enumerate(huelabels):
        axs[0].errorbar(time, any1[i], any1_err[i], label=huelabel,
                        color=sns.color_palette(palette)[i], ecolor='grey', fmt=fmt1)
    axs[0].set_xlim(time.min(), time.max())
    axs[0].set_ylim(range1)
    # axs[0].set_xlabel('Time [s]')
    axs[0].set_xticklabels([])
    axs[0].set_ylabel(label1)

    for i, huelabel in enumerate(huelabels):
        axs[1].errorbar(time, any2[i], any2_err[i],
                        color=sns.color_palette(palette)[i], ecolor='grey', fmt=fmt2)
    axs[1].set_xlim(time.min(), time.max())
    axs[1].set_ylim(range2)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel(label2)

    fig.legend()

    return fig, axs


def any_t(time, dat, datLabel, ylim=False, fmt='-'):

    tLabel = 'Time [s]'
    fig, ax = plt.subplots()
    ax.plot(time, dat, fmt)
    ax.set_ylabel(datLabel)
    ax.set_xlabel(tLabel)

    if ylim:
        ax.set_ylim(ylim)

    return fig, ax


def any(x, y, xLabel, yLabel, xlim=False, ylim=False, fmt='.'):

    fig, ax = plt.subplots()
    ax.plot(x, y, fmt)
    ax.set_ylabel(yLabel)
    ax.set_xlabel(xLabel)

    if xlim:
        ax.set_xlim(xlim)
    else:
        xmax, xmin = maxAndMin(x)
        ax.set_xlim((xmin, xmax))

    if ylim:
        ax.set_ylim(ylim)
    else:
        if xlim:
            xmin, xmax = xlim
            idxs = np.where(~((x >= xmin) & (x <= xmax)))
            y[idxs] = np.nan
            ymax, ymin = maxAndMin(y)
        else:
            ymax, ymin = maxAndMin(y)
        ax.set_ylim((ymin, ymax))

    return fig, ax


def any2_t(time, any1, any2, range1, range2, label1, label2):

    fig, axs = plt.subplots(2)
    axs[0].plot(time, any1,
                color='black')
    axs[0].set_xlim(time.min(), time.max())
    axs[0].set_ylim(range1)
    # axs[0].set_xlabel('Time [s]')
    axs[0].set_xticklabels([])
    axs[0].set_ylabel(label1)

    axs[1].plot(time, any2,
                color='black')
    axs[1].set_xlim(time.min(), time.max())
    axs[1].set_ylim(range2)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel(label2)

    return fig, axs


def any2_t_mul(time, huelabels, any1, any2, range1, range2, label1, label2, colors):

    fig, axs = plt.subplots(2)
    for i, huelabel in enumerate(huelabels):
        axs[0].errorbar(time, any1[i], label=huelabel,
                        color=colors[i])
    axs[0].set_xlim(time.min(), time.max())
    axs[0].set_ylim(range1)
    # axs[0].set_xlabel('Time [s]')
    axs[0].set_xticklabels([])
    axs[0].set_ylabel(label1)

    for i, huelabel in enumerate(huelabels):
        axs[1].errorbar(time, any2[i],
                        color=colors[i])
    axs[1].set_xlim(time.min(), time.max())
    axs[1].set_ylim(range2)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel(label2)

    fig.legend()

    return fig, axs
