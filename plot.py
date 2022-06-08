import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('notebook')
sns.set_style('ticks')


def check(duration):
    if duration == 0:
        plt.show()
    else:
        plt.pause(duration)
        plt.clf()
        plt.close()

    return


def capsave(fig, fig_title, fnm, path):
    fig.suptitle(fig_title)
    fig.text(1, 0.01, fnm, ha='right', color='grey', size='xx-small')
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    fig.savefig(path)
    return fig


def errorbar_single(x, y, yerr, xlabel, ylabel, xlim, ylim,
                    color='black', lw=1):

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr,
                color=color, ecolor='grey',
                lw=lw, elinewidth=lw)
    ax.axhline(0, color='grey', ls='--')
    ax.axvline(0, color='grey', ls='--')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return fig, ax


def errorbar_hue(li_x, li_y, li_yerr, li_huecolor, li_lw, li_lab_hue, xlabel, ylabel, xlim, ylim):

    fig, axs = plt.subplots()
    for j in range(len(li_lab_hue)):
        axs.errorbar(li_x[j], li_y[j], li_yerr[j],
                         color=li_huecolor[j], ecolor='grey',
                         lw=li_lw[j], elinewidth=li_lw[j])
    axs.legend(li_lab_hue)
    axs.axhline(0, color='grey', ls='--')
    axs.axvline(0, color='grey', ls='--')
    axs.set_xlim(xlim)
    axs.set_ylim(ylim)
    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)

    return fig, axs


def errorbar_vert(li_x, li_y, li_yerr, xlabel, li_lab_y, xlim, ylims, color='black', lw=1):

    fig, axs = plt.subplots(len(li_lab_y))
    for i in range(len(li_lab_y)):
        axs[i].errorbar(li_x[i], li_y[i], li_yerr[i],
                         color=color, ecolor='grey',
                         lw=lw, elinewidth=lw)
        axs[i].axhline(0, color='grey', ls='--')
        axs[i].axvline(0, color='grey', ls='--')
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylims)
        axs[i].set_ylabel(li_lab_y[i])
        if i == len(li_lab_y) - 1:
            axs[i].set_xlabel(xlabel)
        else:
            axs[i].set_xticklabels([])

    return fig, axs


def errorbar_vert_hue(li_x, li_y, li_yerr, li_huecolor, li_lw, li_lab_y, li_lab_hue, xlabel, xlim, ylims):

    fig, axs = plt.subplots(len(li_lab_y))
    for i in range(len(li_lab_y)):
        for j in range(len(li_lab_hue)):
            axs[i].errorbar(li_x[i][j], li_y[i][j], li_yerr[i][j],
                             color=li_huecolor[j], ecolor='grey',
                             lw=li_lw[j], elinewidth=li_lw[j])
        if i == 0:
            fig.legend(li_lab_hue)
        axs[i].axhline(0, color='grey', ls='--')
        axs[i].axvline(0, color='grey', ls='--')
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylims)
        axs[i].set_ylabel(li_lab_y[i])
        if i == len(li_lab_y) - 1:
            axs[i].set_xlabel(xlabel)
        else:
            axs[i].set_xticklabels([])

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


def errorbar_fk_2s(freq_k, flim_k, any, any_err, bottom, top, label, fmt):

    fig, ax = plt.subplots()
    ax.errorbar(freq_k, any, any_err, color='red', ecolor='blue', fmt=fmt)
    ax.set_xlim(-flim_k, flim_k)
    ax.set_ylim(bottom, top)
    ax.set_xlabel('Frequency [kHz]')
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
