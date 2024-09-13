from nasu import read, plot, proc, calc, get_eg, eg_mwrm
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import matplotlib
import os
from scipy.signal import welch, find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import fft
from scipy import ndimage
import numpy as np
import time

plot.set("talk", "ticks")
pathCalib = os.path.join("C:\\pythonProject\\nasu\\calib_table.csv")


def gauss(x, a, b, c, d):
    return a * (np.exp(-(x-c)**2/b**2)) + d

def symgauss(x, a, b, d):
    return gauss(x, a, b, 0, d)

def oddgauss(x, a, b, c):
    return a * (np.exp(-(x-c)**2/b**2) - np.exp(-(x+c)**2/b**2))

def oddgauss2(x, a1, b1, c1, a2, b2, c2):
    return oddgauss(x, a1, b1, c1) + oddgauss(x, a2, b2, c2)

def oddgauss3(x, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    return oddgauss(x, a1, b1, c1) + oddgauss(x, a2, b2, c2) + oddgauss(x, a3, b3, c3)

def oddgauss4(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
    return oddgauss(x, a1, b1, c1) + oddgauss(x, a2, b2, c2) + oddgauss(x, a3, b3, c3) + oddgauss(x, a4, b4, c4)

def evengauss(x, a, b, c):
    return a * (np.exp(-(x-c)**2/b**2) + np.exp(-(x+c)**2/b**2))

def symgauss2(x, a1, b1, a2, b2, c2, d):
    return symgauss(x, a1, b1, d) + evengauss(x, a2, b2, c2)

def symgauss3(x, a1, b1, a2, b2, c2, a3, b3, c3, d):
    return symgauss(x, a1, b1, d) + evengauss(x, a2, b2, c2) + evengauss(x, a3, b3, c3)

def symgauss4(x, a1, b1, a2, b2, c2, a3, b3, c3, a4, b4, c4, d):
    return symgauss(x, a1, b1, d) + evengauss(x, a2, b2, c2) + evengauss(x, a3, b3, c3) + evengauss(x, a4, b4, c4)

def gauss1(x, a0, b0, d, a1, b1, c1):
    return gauss(x, a0, b0, 0, d) + gauss(x, a1, b1, c1, 0)

def gauss2(x, a0, b0, d, a1, b1, c1, a2, b2, c2):
    return gauss(x, a0, b0, 0, d) + gauss(x, a1, b1, c1, 0) + gauss(x, a2, b2, c2, 0)

def gauss3(x, a0, b0, d, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    return gauss(x, a0, b0, 0, d) + gauss(x, a1, b1, c1, 0) + gauss(x, a2, b2, c2, 0) + gauss(x, a3, b3, c3, 0)

def gauss4(x, a0, b0, d, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
    return gauss(x, a0, b0, 0, d) + gauss(x, a1, b1, c1, 0) + \
           gauss(x, a2, b2, c2, 0) + gauss(x, a3, b3, c3, 0) + gauss(x, a4, b4, c4, 0)

def cauchy_lorentz(x, a, b, c, d):
    return a * b**2 / (b**2 + (x - c)**2) + d

def gauss_and_lorentz(x, ag, bg, cg, al, bl, cl, d):
    return gauss(x, ag, bg, cg, 0) + cauchy_lorentz(x, al, bl, cl, 0) + d


class randomIQ:

    def __init__(self, Fs, mu, sigma, tstart, tend):

        self.Fs = Fs
        self.dT = 1./Fs
        self.mu = mu
        self.sigma = sigma
        self.tstart = tstart
        self.tend = tend

        self.t = np.arange(self.tstart, self.tend + self.dT, self.dT)
        self.size = self.t.size
        self.IQ = np.random.normal(self.mu, self.sigma, self.size) \
                  + 1.j * np.random.normal(self.mu, self.sigma, self.size)


class single:

    def __init__(self, sn=187570, subsn=1, tstart=3., tend=6., diagname="MWRM-PXI", ch=1):

        self.sn = sn
        self.subsn = subsn
        self.tstart = tstart
        self.tend = tend
        self.diagname = diagname
        self.ch = ch

        self.t, self.d, self.dT, self.Fs, self.size, self.tprms, self.dprms = \
            read.LHD_et_v2(sn=sn, subsn=subsn, diagname=diagname, ch=ch, et=(tstart, tend))

    def specgram(self, NFFT=2**10, ovr=0.5, window="hann", dT=5e-3,
                 cmap="viridis", magnify=True, pause=0.1, display=True):

        self.spg = calc.struct()
        self.spg.NFFT = NFFT
        self.spg.ovr = ovr
        self.spg.NOV = int(self.spg.NFFT * self.spg.ovr)
        self.spg.window = window
        self.spg.dT = dT
        self.spg.Fssp = 1./self.spg.dT
        self.spg.NSamp = int(self.Fs / self.spg.Fssp)
        self.spg.Nsp = self.size // self.spg.NSamp
        self.spg.cmap = cmap

        self.spg.t = self.t[:self.spg.Nsp * self.spg.NSamp].reshape((self.spg.Nsp, self.spg.NSamp)).mean(axis=-1)
        self.spg.darray = self.d[:self.spg.Nsp * self.spg.NSamp].reshape((self.spg.Nsp, self.spg.NSamp))
        self.spg.f, self.spg.psd = welch(x=self.spg.darray, fs=self.Fs, window="hann",
                                                   nperseg=self.spg.NFFT, noverlap=self.spg.NOV,
                                                   detrend="constant", scaling="density",
                                                   axis=-1, average="mean")
        self.spg.dF = self.dT * self.spg.NFFT

        figdir = "Retrieve_MWRM"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.ch}"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.ch}"
        fig, axs = plt.subplots(nrows=2, sharex=True,
                                figsize=(5, 8), gridspec_kw={'height_ratios': [1, 3]},
                                num=fnm)

        axs[0].plot(self.t, self.d, c="black", lw=0.1)
        axs[0].set_ylabel("[V]")
        if magnify:
            p2p = self.d.max() - self.d.min()
            axs[0].set_ylim(self.d.min() - p2p * 0.05, self.d.max() + p2p * 0.05)
        else:
            axs[0].set_ylim(float(self.dprms["RangeLow"][0]), float(self.dprms["RangeHigh"][0]))

        if not display:
            matplotlib.use('Agg')

        axs[1].pcolorfast(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                          np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                          10*np.log10(self.spg.psd.T), cmap=self.spg.cmap)
        axs[1].set_ylabel("Frequency [Hz]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)


class IQ:

    def __init__(self, sn=187570, subsn=1, tstart=3., tend=6.,
                 diagname="MWRM-PXI", chI=11, chQ=12, phase_unwrap=True, save=True):   # LHD version

        self.sn = sn
        self.subsn = subsn
        self.tstart = tstart
        self.tend = tend
        self.diagname = diagname
        self.chI = chI
        self.chQ = chQ

        self.t, self.I, self.Q, self.dT, self.size, self.tprms, self.Iprms, self.Qprms = \
            read.LHD_IQ_et_v3(sn=self.sn, subsn=self.subsn,
                              diagname=self.diagname, chs=[self.chI, self.chQ],
                              et=(self.tstart, self.tend))
        self.Fs = 1. / self.dT

        if diagname == "MWRM-COMB2" and chI in np.arange(1, 16 + 1, 2):

            dictIF = {1: 40, 3: 80, 5: 120, 7: 160,
                      9: 200, 11: 240, 13: 300, 15: 340}
            self.frLO = dictIF[chI]
            calibPrms_df = read.calibPrms_df_v2(pathCalib)
            self.ampratioIQ, self.offsetI, self.offsetQ, self.phasediffIQ = calibPrms_df.loc[self.frLO]

            self.I, self.Q = calc.calibIQComp2(self.I, self.Q,
                                               self.ampratioIQ, self.offsetI, self.offsetQ, self.phasediffIQ)

        elif diagname == "MWRM-COMB" and chI in [1, 3, 5, 7]:
            self.Q *= -1

        self.IQ = self.I + 1.j * self.Q
        self.ampIQ = np.abs(self.IQ)
        if phase_unwrap:
            self.phaseIQ = np.unwrap(np.angle(self.IQ), 1.5 * np.pi)
        else:
            self.phaseIQ = np.angle(self.IQ)

        if save:
            self.dirbase = "Retrieve_MWRM"
            proc.ifNotMake(self.dirbase)
            self.fnm_init = f"{sn}_{subsn}_{tstart}_{tend}_{diagname}_{chI}_{chQ}"
            self.figtitle = f"#{self.sn}-{self.subsn}\n" \
                            f"1: {self.diagname} {self.chI} {self.chQ}"

        self.spg = calc.struct()
        self.sp = calc.struct()

    def specgram(self, NFFT=2**10, ovr=0.5, window="hann", dT=5e-3,
                 cmap="viridis", magnify=True, fmin=None, fmax=None, pause=0.,
                 display=True, detrend="constant", sub_phi=False, save=True):

        plot.set("notebook", "ticks")

        self.spg.NFFT = NFFT
        self.spg.ovr = ovr
        self.spg.NOV = int(self.spg.NFFT * self.spg.ovr)
        self.spg.window = window
        self.spg.dT = dT
        self.spg.Fs = 1./self.spg.dT
        self.spg.NSamp = int(self.spg.dT / self.dT)
        # self.spg.NSamp = int(self.Fs / self.spg.Fs)
        self.spg.Nsp = self.size // self.spg.NSamp
        self.spg.NEns = calc.NEnsFromNSample(self.spg.NFFT, self.spg.NOV, self.spg.NSamp)
        self.spg.cmap = cmap

        self.spg.tarray = self.t[:self.spg.Nsp * self.spg.NSamp].reshape((self.spg.Nsp, self.spg.NSamp))
        self.spg.t = self.spg.tarray.mean(axis=-1)
        self.spg.IQarray = self.IQ[:self.spg.Nsp * self.spg.NSamp].reshape((self.spg.Nsp, self.spg.NSamp))
        self.spg.f, self.spg.psd = welch(x=self.spg.IQarray, fs=self.Fs, window="hann",
                                         nperseg=self.spg.NFFT, noverlap=self.spg.NOV,
                                         return_onesided=False,
                                         detrend=detrend, scaling="density",
                                         axis=-1, average="mean")

        self.spg.psd = fft.fftshift(self.spg.psd, axes=-1)
        self.spg.psdamp = np.sqrt(self.spg.psd)
        self.spg.psddB = 10 * np.log10(self.spg.psd)
        if sub_phi:
            self.phase_ma = calc.moving_average(self.phaseIQ, window_size=self.spg.NFFT + 1, mode="same")
            self.dphase = self.phaseIQ - self.phase_ma
            self.IQ_subphi = self.ampIQ * np.exp(1.j * self.dphase)
            self.I_subphi = np.real(self.IQ_subphi)
            self.Q_subphi = np.imag(self.IQ_subphi)
            self.spg.IQarray_subphi = self.IQ_subphi[:self.spg.Nsp * self.spg.NSamp].reshape((self.spg.Nsp, self.spg.NSamp))
            self.spg.f, self.spg.psd_subphi = welch(x=self.spg.IQarray_subphi, fs=self.Fs, window="hann",
                                                    nperseg=self.spg.NFFT, noverlap=self.spg.NOV,
                                                    return_onesided=False,
                                                    detrend=detrend, scaling="density",
                                                    axis=-1, average="mean")
            self.spg.psd_subphi = fft.fftshift(self.spg.psd_subphi, axes=-1)
            self.spg.psddB_subphi = 10 * np.log10(self.spg.psd_subphi)

        self.spg.f = fft.fftshift(self.spg.f)
        self.spg.dF = self.Fs / self.spg.NFFT

        if magnify:
            if fmin is not None:
                self.spg.fmin = fmin
            else:
                self.spg.fmin = self.spg.dF
            if fmax is not None:
                self.spg.fmax = fmax
            else:
                self.spg.fmax = self.Fs / 2

        _idx = np.where((np.abs(self.spg.f) > self.spg.fmin) & (np.abs(self.spg.f) < self.spg.fmax))[0]
        self.spg.vmax = self.spg.psd[:, _idx].max()
        self.spg.vmaxdB = 10 * np.log10(self.spg.vmax)
        self.spg.vmin = self.spg.psd[:, _idx].min()
        self.spg.vmindB = 10 * np.log10(self.spg.vmin)

        self.spg.psddB_diff = np.diff(self.spg.psddB, axis=-1)

        if save:
            if not display:
                original_backend = matplotlib.get_backend()
                matplotlib.use('Agg')

            if sub_phi:
                figdir = os.path.join(self.dirbase, "specgram_subphi")
            else:
                figdir = os.path.join(self.dirbase, "specgram")
            proc.ifNotMake(figdir)
            if sub_phi:
                fnm_base = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}_" \
                           f"{self.spg.NFFT}_{self.spg.ovr}_{self.spg.window}_{self.spg.dT}_subphi"
            else:
                fnm_base = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}_" \
                           f"{self.spg.NFFT}_{self.spg.ovr}_{self.spg.window}_{self.spg.dT}"
            if fmin is not None:
                fnm_base += f"_min{fmin*1e-3}kHz"
            if fmax is not None:
                fnm_base += f"_max{fmax*1e-3}kHz"
            path = os.path.join(figdir, f"{fnm_base}.png")
            self.fnm_spg = fnm_base
            if sub_phi:
                title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                        f"{self.diagname} ch:{self.chI},{self.chQ}\n" \
                        f"{self.spg.NFFT} {self.spg.ovr} {self.spg.window} {self.spg.dT}s\n" \
                        f"subtract phase line"
            else:
                title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                        f"{self.diagname} ch:{self.chI},{self.chQ}\n" \
                        f"{self.spg.NFFT} {self.spg.ovr} {self.spg.window} {self.spg.dT}s"

            fig, axs = plt.subplots(nrows=3, sharex=True,
                                    figsize=(5, 10), gridspec_kw={'height_ratios': [1, 1, 3]},
                                    num=fnm_base)

            if sub_phi:
                axs[0].plot(self.t, self.I_subphi, c="black", lw=0.1)
            else:
                axs[0].plot(self.t, self.I, c="black", lw=0.1)
            axs[0].set_ylabel("I [V]")
            if magnify:
                p2p = self.I.max() - self.I.min()
                axs[0].set_ylim(self.I.min() - p2p * 0.05, self.I.max() + p2p * 0.05)
            else:
                axs[0].set_ylim(float(self.Iprms["RangeLow"][0]), float(self.Iprms["RangeHigh"][0]))

            if sub_phi:
                axs[1].plot(self.t, self.Q_subphi, c="black", lw=0.1)
            else:
                axs[1].plot(self.t, self.Q, c="black", lw=0.1)
            if magnify:
                p2p = self.Q.max() - self.Q.min()
                axs[1].set_ylim(self.Q.min() - p2p * 0.05, self.Q.max() + p2p * 0.05)
            else:
                axs[1].set_ylim(float(self.Qprms["RangeLow"][0]), float(self.Qprms["RangeHigh"][0]))
            axs[1].set_ylabel("Q [V]")

            if sub_phi:
                axs[2].pcolormesh(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                                  np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                                  self.spg.psddB_subphi.T,
                                  cmap=self.spg.cmap, vmin=self.spg.vmindB, vmax=self.spg.vmaxdB)
            else:
                axs[2].pcolormesh(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                                  np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                                  self.spg.psddB.T,
                                  cmap=self.spg.cmap, vmin=self.spg.vmindB, vmax=self.spg.vmaxdB)
            if magnify:
                axs[2].set_ylim(- self.spg.fmax, self.spg.fmax)
            axs[2].set_ylabel("Frequency [Hz]")
            axs[2].set_xlabel("Time [s]")
            axs[2].set_xlim(self.tstart, self.tend)

            plot.caption(fig, title, hspace=0.1, wspace=0.1)
            plot.capsave(fig, title, fnm_base, path)

            if display:
                plot.check(pause)
            else:
                plot.close(fig)

            fig2dir = os.path.join(self.dirbase, "specgram_diff")
            proc.ifNotMake(fig2dir)
            fnm = f"{fnm_base}_diff"
            if fmin is not None:
                fnm += f"_min{fmin * 1e-3}kHz"
            if fmax is not None:
                fnm += f"_max{fmax * 1e-3}kHz"
            path = os.path.join(fig2dir, f"{fnm}.png")
            title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                    f"{self.diagname} ch:{self.chI},{self.chQ}\n" \
                    f"{self.spg.NFFT} {self.spg.ovr} {self.spg.window} {self.spg.dT}s"
            fig2, ax2 = plt.subplots(nrows=1, num=fnm, figsize=(5, 5))

            ax2.pcolormesh(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                           self.spg.f,
                           self.spg.psddB_diff.T,
                           cmap="coolwarm")
            if magnify:
                ax2.set_ylim(- self.spg.fmax, self.spg.fmax)
            ax2.set_ylabel("Frequency [Hz]")
            ax2.set_xlabel("Time [s]")
            ax2.set_xlim(self.tstart, self.tend)

            plot.caption(fig2, title, hspace=0.1, wspace=0.1)
            plot.capsave(fig2, title, fnm, path)

            if display:
                plot.check(pause)
            else:
                plot.close(fig2)

            if not display:
                matplotlib.use(original_backend)

        return self.spg

    def specgram_asymmetric_component(self, fmin=None, fmax=None,
                                      peak_detection=True, gaussfitting=True,
                                      polyorder=2, prominence=3, iniwidth=40e3, maxfev=2000,
                                      cmap="coolwarm",
                                      display=True, pause=0., show_smoothing=True):

        self.spg.asym = calc.struct()
        self.spg.asym.iniwidth = iniwidth
        self.spg.asym.cmap = cmap

        if fmin is None:
            self.spg.asym.fmin = self.spg.dF
        else:
            self.spg.asym.fmin = fmin
        if fmax is None:
            self.spg.asym.fmax = self.Fs / 2
        else:
            self.spg.asym.fmax = fmax
        _idx = np.where((np.abs(self.spg.f) > self.spg.asym.fmin)
                        & (np.abs(self.spg.f) < self.spg.asym.fmax))[0]
        self.spg.asym.psddB = self.spg.psddB - np.flip(self.spg.psddB, axis=-1)
        self.spg.asym.winlen = int(self.spg.asym.iniwidth / self.spg.dF / 2) * 2 + 1
        if peak_detection:
            self.spg.asym.psddB_smooth = np.zeros(self.spg.asym.psddB.shape)
            self.spg.asym.peak_idxs = [0] * len(self.spg.t)
            self.spg.asym.peak_freqs = [0] * len(self.spg.t)
            self.spg.asym.peak_psddB = [0] * len(self.spg.t)
            self.spg.asym.peak_nums = [0] * len(self.spg.t)
            for i in range(len(self.spg.t)):
                # peak detection
                tmp = interp1d(self.spg.f[_idx], self.spg.asym.psddB[i][_idx],
                               bounds_error=False, fill_value=0.)(self.spg.f)
                self.spg.asym.psddB_smooth[i] = savgol_filter(tmp,
                                                              window_length=self.spg.asym.winlen,
                                                              polyorder=polyorder)
                self.spg.asym.peak_idxs[i] = find_peaks(self.spg.asym.psddB_smooth[i], prominence=prominence)[0]
                self.spg.asym.peak_freqs[i] = self.spg.f[self.spg.asym.peak_idxs[i]]
                self.spg.asym.peak_psddB[i] = self.spg.asym.psddB_smooth[i][self.spg.asym.peak_idxs[i]]
                self.spg.asym.peak_nums[i] = np.array(self.spg.asym.peak_idxs[i]).size

        if gaussfitting:
            self.spg.asym.peak_fit = [0] * len(self.spg.t)
            self.spg.asym.peak_fit_err = [0] * len(self.spg.t)
            self.spg.asym.fD_fit = [0] * len(self.spg.t)
            self.spg.asym.fD_fit_err = [0] * len(self.spg.t)
            self.spg.asym.width_fit = [0] * len(self.spg.t)
            self.spg.asym.width_fit_err = [0] * len(self.spg.t)
            self.spg.asym.gaussfit_num = [0] * len(self.spg.t)
            for i in range(len(self.spg.t)):
                # gauss fitting
                if self.spg.asym.peak_nums[i] == 0:
                    self.spg.asym.gaussfit_num[i] = 0
                    self.spg.asym.peak_fit[i], self.spg.asym.width_fit[i], self.spg.asym.fD_fit[i] \
                        = [np.array([]), np.array([]), np.array([])]
                    self.spg.asym.peak_fit_err[i], self.spg.asym.width_fit_err[i], self.spg.asym.fD_fit_err[i] \
                        = [np.array([]), np.array([]), np.array([])]
                elif self.spg.asym.peak_nums[i] == 1:
                    self.spg.asym.gaussfit_num[i] = 1
                    inip = [self.spg.asym.peak_psddB[i][0], iniwidth, self.spg.asym.peak_freqs[i][0]]
                    try:
                        popt, pcov = curve_fit(oddgauss, self.spg.f[_idx], self.spg.asym.psddB[i][_idx],
                                               p0=inip, maxfev=maxfev)
                        self.spg.asym.peak_fit[i] = [popt[0]]
                        self.spg.asym.width_fit[i] = [popt[1]]
                        self.spg.asym.fD_fit[i] = [popt[2]]
                        self.spg.asym.peak_fit_err[i] = [np.sqrt(np.diag(pcov))[0]]
                        self.spg.asym.width_fit_err[i] = [np.sqrt(np.diag(pcov))[1]]
                        self.spg.asym.fD_fit_err[i] = [np.sqrt(np.diag(pcov))[2]]

                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.spg.asym.peak_fit[i], self.spg.asym.width_fit[i], self.spg.asym.fD_fit[i] \
                            = [np.array([]), np.array([]), np.array([])]
                        self.spg.asym.peak_fit_err[i], self.spg.asym.width_fit_err[i], self.spg.asym.fD_fit_err[i] \
                            = [np.array([]), np.array([]), np.array([])]
                elif self.spg.asym.peak_nums[i] == 2:
                    self.spg.asym.gaussfit_num[i] = 2
                    inip = [self.spg.asym.peak_psddB[i], [iniwidth, iniwidth], self.spg.asym.peak_freqs[i]]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss2, self.spg.f[_idx], self.spg.asym.psddB[i][_idx],
                                               p0=inip, maxfev=maxfev)
                        self.spg.asym.peak_fit[i], self.spg.asym.width_fit[i], self.spg.asym.fD_fit[i] \
                            = popt.reshape((self.spg.asym.peak_nums[i], 3)).T
                        self.spg.asym.peak_fit_err[i], self.spg.asym.width_fit_err[i], self.spg.asym.fD_fit_err[i] \
                            = np.sqrt(np.diag(pcov)).reshape((self.spg.asym.peak_nums[i], 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.spg.asym.peak_fit[i], self.spg.asym.width_fit[i], self.spg.asym.fD_fit[i] \
                            = [np.array([]), np.array([]), np.array([])]
                        self.spg.asym.peak_fit_err[i], self.spg.asym.width_fit_err[i], self.spg.asym.fD_fit_err[i] \
                            = [np.array([]), np.array([]), np.array([])]
                elif self.spg.asym.peak_nums[i] == 3:
                    self.spg.asym.gaussfit_num[i] = 3
                    inip = [self.spg.asym.peak_psddB[i], [iniwidth, iniwidth, iniwidth], self.spg.asym.peak_freqs[i]]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss3, self.spg.f[_idx], self.spg.asym.psddB[i][_idx],
                                               p0=inip, maxfev=maxfev)
                        self.spg.asym.peak_fit[i], self.spg.asym.width_fit[i], self.spg.asym.fD_fit[i] \
                            = popt.reshape((self.spg.asym.peak_nums[i], 3)).T
                        self.spg.asym.peak_fit_err[i], self.spg.asym.width_fit_err[i], self.spg.asym.fD_fit_err[i] \
                            = np.sqrt(np.diag(pcov)).reshape((self.spg.asym.peak_nums[i], 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.spg.asym.peak_fit[i], self.spg.asym.width_fit[i], self.spg.asym.fD_fit[i] \
                            = [np.array([]), np.array([]), np.array([])]
                        self.spg.asym.peak_fit_err[i], self.spg.asym.width_fit_err[i], self.spg.asym.fD_fit_err[i] \
                            = [np.array([]), np.array([]), np.array([])]
                elif self.spg.asym.peak_nums[i] == 4:
                    self.spg.asym.gaussfit_num[i] = 4
                    inip = [self.spg.asym.peak_psddB[i],
                            [iniwidth, iniwidth, iniwidth, iniwidth],
                            self.spg.asym.peak_freqs[i]]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss4, self.spg.f[_idx], self.spg.asym.psddB[i][_idx],
                                               p0=inip, maxfev=maxfev)
                        self.spg.asym.peak_fit[i], self.spg.asym.width_fit[i], self.spg.asym.fD_fit[i] \
                            = popt.reshape((self.spg.asym.gaussfit_num[i], 3)).T
                        self.spg.asym.peak_fit_err[i], self.spg.asym.width_fit_err[i], self.spg.asym.fD_fit_err[i] \
                            = np.sqrt(np.diag(pcov)).reshape((self.spg.asym.gaussfit_num[i], 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.spg.asym.peak_fit[i], self.spg.asym.width_fit[i], self.spg.asym.fD_fit[i] \
                            = [np.array([]), np.array([]), np.array([])]
                        self.spg.asym.peak_fit_err[i], self.spg.asym.width_fit_err[i], self.spg.asym.fD_fit_err[i] \
                            = [np.array([]), np.array([]), np.array([])]
                else:  # until 3.
                    self.spg.asym.gaussfit_num[i] = 4
                    idx_str = np.argsort(self.spg.asym.peak_psddB[i])[-self.spg.asym.gaussfit_num[i]:]
                    inip = [self.spg.asym.peak_psddB[i][idx_str],
                            [iniwidth, iniwidth, iniwidth, iniwidth],
                            self.spg.asym.peak_freqs[i][idx_str]]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss4, self.spg.f[_idx], self.spg.asym.psddB[i][_idx],
                                               p0=inip, maxfev=maxfev)
                        self.spg.asym.peak_fit[i], self.spg.asym.width_fit[i], self.spg.asym.fD_fit[i] \
                            = popt.reshape((self.spg.asym.gaussfit_num[i], 3)).T
                        self.spg.asym.peak_fit_err[i], self.spg.asym.width_fit_err[i], self.spg.asym.fD_fit_err[i] \
                            = np.sqrt(np.diag(pcov)).reshape((self.spg.asym.gaussfit_num[i], 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.spg.asym.peak_fit[i], self.spg.asym.width_fit[i], self.spg.asym.fD_fit[i] \
                            = [np.array([]), np.array([]), np.array([])]
                        self.spg.asym.peak_fit_err[i], self.spg.asym.width_fit_err[i], self.spg.asym.fD_fit_err[i] \
                            = [np.array([]), np.array([]), np.array([])]

        def pad_lists_to_ndarray(lists, len_list):
            _max = np.max(len_list)
            padded_array = np.array([np.pad(np.array(lists[i]),
                                             (0, _max - len_list[i]),
                                              constant_values=np.nan)
                                      for i in range(len(lists))])

            return padded_array

        if peak_detection:
            self.spg.asym.peak_freqs = pad_lists_to_ndarray(self.spg.asym.peak_freqs, self.spg.asym.peak_nums)
            self.spg.asym.peak_psddB = pad_lists_to_ndarray(self.spg.asym.peak_psddB, self.spg.asym.peak_nums)
        if gaussfitting:
            self.spg.asym.valnums_fit = [np.array(self.spg.asym.peak_fit[i]).size for i in range(self.spg.Nsp)]
            self.spg.asym.peak_fit = pad_lists_to_ndarray(self.spg.asym.peak_fit, self.spg.asym.valnums_fit)
            self.spg.asym.peak_fit_err = pad_lists_to_ndarray(self.spg.asym.peak_fit_err, self.spg.asym.valnums_fit)
            self.spg.asym.width_fit = pad_lists_to_ndarray(self.spg.asym.width_fit, self.spg.asym.valnums_fit)
            self.spg.asym.width_fit_err = pad_lists_to_ndarray(self.spg.asym.width_fit_err, self.spg.asym.valnums_fit)
            self.spg.asym.fD_fit = pad_lists_to_ndarray(self.spg.asym.fD_fit, self.spg.asym.valnums_fit)
            self.spg.asym.fD_fit_err = pad_lists_to_ndarray(self.spg.asym.fD_fit_err, self.spg.asym.valnums_fit)

        figdir = "Retrieve_MWRM_asym"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}"
        if fmin is not None:
            fnm += f"_min{self.spg.asym.fmin * 1e-3}kHz"
        if fmax is not None:
            fnm += f"_max{self.spg.asym.fmax * 1e-3}kHz"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"
        fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(7, 6), num=fnm)

        self.spg.asym.vmaxdB = self.spg.asym.psddB[:, _idx].max()
        self.spg.asym.vmindB = self.spg.asym.psddB[:, _idx].min()

        if show_smoothing and peak_detection:
            im = ax.pcolormesh(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                               np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                               self.spg.asym.psddB_smooth.T,
                               cmap=self.spg.asym.cmap, vmin=self.spg.asym.vmindB, vmax=self.spg.asym.vmaxdB)
            ax.fill_between(self.spg.t, -self.spg.asym.fmin, self.spg.asym.fmin, color="white", alpha=0.3)
        else:
            im = ax.pcolormesh(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                               np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                               self.spg.asym.psddB.T,
                               cmap=self.spg.asym.cmap, vmin=self.spg.asym.vmindB, vmax=self.spg.asym.vmaxdB)
            ax.fill_between(self.spg.t, -self.spg.asym.fmin, self.spg.asym.fmin, color="white", alpha=0.3)
        cbar = plt.colorbar(im)
        cbar.set_label("S(f) / S(-f) [dB]")
        if peak_detection:
            # for i in range(len(self.spg.t)):
            #     ax.scatter([self.spg.t[i]] * self.spg.asym.peak_nums[i], self.spg.asym.peak_freqs[i],
            #                s=1, c="red", alpha=0.5)
            ax.plot(self.spg.t, self.spg.asym.peak_freqs, ".", ms=3, c="green", alpha=0.5)
        if gaussfitting:
            # for i in range(len(self.spg.t)):
            #     if (~isinstance(self.spg.asym.fD_fit[i], np.ndarray)) and (self.spg.asym.fD_fit[i] is np.nan):
            #         continue
            #     ax.scatter([self.spg.t[i]]*self.spg.asym.gaussfit_num[i], self.spg.asym.fD_fit[i],
            #                 s=1, c="pink", alpha=0.5)
            ax.plot(self.spg.t, self.spg.asym.fD_fit, ".", ms=2, c="purple")
        ax.set_ylim(- self.spg.asym.fmax, self.spg.asym.fmax)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        ax.set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    def specgram_smooth(self, twin_size=10, fwin_size=1, mode="ma", sub_phi=False, display=True, pause=0.):
        # mode = "ma", "gauss", "med"

        if sub_phi:
            if mode == "ma":
                self.spg.psd_subphi_smooth = ndimage.uniform_filter(self.spg.psd_subphi, size=(twin_size, fwin_size))
            elif mode == "gauss":
                self.spg.psd_subphi_smooth = ndimage.gaussian_filter(self.spg.psd_subphi, sigma=(0.5 * twin_size, 0.5 * fwin_size))
            elif mode == "med":
                self.spg.psd_subphi_smooth = ndimage.median_filter(self.spg.psd_subphi, size=(twin_size, fwin_size))
            else:
                exit()

            self.spg.psd_subphi_smooth_dB = 10 * np.log10(self.spg.psd_subphi_smooth)

        else:
            if mode == "ma":
                self.spg.psd_smooth = ndimage.uniform_filter(self.spg.psd, size=(twin_size, fwin_size))
            elif mode == "gauss":
                self.spg.psd_smooth = ndimage.gaussian_filter(self.spg.psd, sigma=(0.5 * twin_size, 0.5 * fwin_size))
            elif mode == "med":
                self.spg.psd_smooth = ndimage.median_filter(self.spg.psd, size=(twin_size, fwin_size))
            else:
                exit()

            self.spg.psd_smooth_dB = 10 * np.log10(self.spg.psd_smooth)

        plot.set("notebook", "ticks")

        figdir = os.path.join(self.dirbase, "specgram_smooth")
        proc.ifNotMake(figdir)
        fnm = f"{self.fnm_spg}_{twin_size}_{fwin_size}_{mode}"
        if sub_phi:
            fnm += "_subphi"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}\n" \
                f"{self.spg.NFFT} {self.spg.ovr} {self.spg.window} {self.spg.dT}s\n" \
                f"{mode} {twin_size} {fwin_size}"

        fig, ax = plt.subplots(nrows=1, num=fnm)

        if sub_phi:
            ax.pcolormesh(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                          np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                          self.spg.psd_subphi_smooth_dB.T,
                          cmap=self.spg.cmap, vmin=self.spg.vmindB, vmax=self.spg.vmaxdB)
        else:
            ax.pcolormesh(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                          np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                          self.spg.psd_smooth_dB.T,
                          cmap=self.spg.cmap, vmin=self.spg.vmindB, vmax=self.spg.vmaxdB)
        ax.set_ylim(- self.spg.fmax, self.spg.fmax)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        ax.set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)


    def specgram_tder(self, twin_size=10, fwin_size=1, mode="med", sub_phi=False, display=True, pause=0.):

        self.specgram_smooth(twin_size, fwin_size, mode, sub_phi=sub_phi, display=False)

        _idx = np.where((np.abs(self.spg.f) > self.spg.fmin)
                        & (np.abs(self.spg.f) < self.spg.fmax))[0]

        if sub_phi:
            self.spg.dpsddT_subphi = np.gradient(self.spg.psd_subphi_smooth, self.spg.dT, axis=0)
            vlim = np.abs(self.spg.dpsddT_subphi)[:, _idx].max()
        else:
            self.spg.dpsddT = np.gradient(self.spg.psd_smooth, self.spg.dT, axis=0)
            vlim = np.abs(self.spg.dpsddT)[:, _idx].max()


        plot.set("notebook", "ticks")

        figdir = os.path.join(self.dirbase, "specgram_tder")
        proc.ifNotMake(figdir)
        fnm = f"{self.fnm_spg}_{twin_size}_{fwin_size}_{mode}"
        if sub_phi:
            fnm += "_subphi"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}\n" \
                f"{self.spg.NFFT} {self.spg.ovr} {self.spg.window} {self.spg.dT}s\n" \
                f"{mode} {twin_size} {fwin_size}"

        fig, ax = plt.subplots(nrows=1, num=fnm)

        if sub_phi:
            im = ax.pcolormesh(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                               np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                               self.spg.dpsddT_subphi.T, vmin=-vlim, vmax=vlim,
                               cmap="coolwarm")
        else:
            im = ax.pcolormesh(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                               np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                               self.spg.dpsddT.T, vmin=-vlim, vmax=vlim,
                               cmap="coolwarm")
        cbar = plt.colorbar(im)
        cbar.set_label("dS / dt")
        ax.set_ylim(- self.spg.fmax, self.spg.fmax)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        ax.set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)



    def specgram_amp(self, NFFT=2**10, ovr=0.5, window="hann", dT=5e-3,
                     cmap="viridis", magnify=True,
                     fmin=None, fmax=None, logfreq=False,
                     pause=0., display=True, detrend="constant"):

        self.spg.amp = calc.struct()

        self.spg.amp.NFFT = NFFT
        self.spg.amp.ovr = ovr
        self.spg.amp.NOV = int(self.spg.amp.NFFT * self.spg.amp.ovr)
        self.spg.amp.window = window
        self.spg.amp.dT = dT
        self.spg.amp.Fs = 1. / self.spg.amp.dT
        self.spg.amp.NSamp = int(self.spg.amp.dT / self.dT)
        # self.spg.amp.NSamp = int(self.Fs / self.spg.amp.Fs)
        self.spg.amp.Nsp = self.size // self.spg.amp.NSamp
        self.spg.amp.cmap = cmap

        self.spg.amp.t = self.t[:self.spg.amp.Nsp * self.spg.amp.NSamp].reshape((self.spg.amp.Nsp, self.spg.amp.NSamp)).mean(axis=-1)
        self.spg.amp.ampIQarray = self.ampIQ[:self.spg.amp.Nsp * self.spg.amp.NSamp].reshape((self.spg.amp.Nsp, self.spg.amp.NSamp))
        self.spg.amp.f, self.spg.amp.psd = welch(x=self.spg.amp.ampIQarray, fs=self.Fs, window="hann",
                                                 nperseg=self.spg.amp.NFFT, noverlap=self.spg.amp.NOV,
                                                 return_onesided=True,
                                                 detrend=detrend, scaling="density",
                                                 axis=-1, average="mean")
        self.spg.amp.f = self.spg.amp.f
        self.spg.amp.psd = self.spg.amp.psd
        self.spg.amp.psddB = 10 * np.log10(self.spg.amp.psd)
        self.spg.amp.dF = self.Fs / self.spg.amp.NFFT

        if magnify:
            if fmin is not None:
                self.spg.amp.fmin = fmin
            else:
                self.spg.amp.fmin = self.spg.amp.dF
            if fmax is not None:
                self.spg.amp.fmax = fmax
            else:
                self.spg.amp.fmax = self.Fs / 2

            _idx = np.where((np.abs(self.spg.amp.f) > self.spg.amp.fmin) & (np.abs(self.spg.amp.f) < self.spg.amp.fmax))[0]
            self.spg.amp.vmax = self.spg.amp.psd[:, _idx].max()
            self.spg.amp.vmaxdB = 10 * np.log10(self.spg.amp.vmax)
            self.spg.amp.vmin = self.spg.amp.psd[:, _idx].min()
            self.spg.amp.vmindB = 10 * np.log10(self.spg.amp.vmin)
        else:
            self.spg.amp.vmax = self.spg.amp.psd.max()
            self.spg.amp.vmaxdB = 10 * np.log10(self.spg.amp.vmax)
            self.spg.amp.vmin = self.spg.amp.psd.min()
            self.spg.amp.vmindB = 10 * np.log10(self.spg.amp.vmin)

        if not display:
            matplotlib.use('Agg')

        figdir = os.path.join(self.dirbase, "ampIQ")
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}" \
              f"_{self.spg.amp.NFFT}_{self.spg.amp.ovr}_{self.spg.amp.window}_{self.spg.amp.dT}"
        if fmin:
            fnm += f"_min{fmin*1e-3}kHz"
        if fmax:
            fnm += f"_max{fmax*1e-3}kHz"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}\n" \
                f"{self.spg.amp.NFFT} {self.spg.amp.ovr} {self.spg.amp.window} {self.spg.amp.dT}s"
        fig, axs = plt.subplots(nrows=2, sharex=True,
                                figsize=(5, 6), gridspec_kw={'height_ratios': [1, 3]},
                                num=fnm)

        axs[0].plot(self.t, self.ampIQ, c="black", lw=0.1)
        axs[0].set_ylabel("amp [a.u.]")
        axs[0].set_ylim(0, self.ampIQ.max() * 1.05)

        axs[1].pcolormesh(np.append(self.spg.amp.t - 0.5 * self.spg.amp.dT, self.spg.amp.t[-1] + 0.5 * self.spg.amp.dT),
                          np.append(self.spg.amp.f - 0.5 * self.spg.amp.dF, self.spg.amp.f[-1] + 0.5 * self.spg.amp.dF),
                          self.spg.amp.psddB.T,
                          cmap=self.spg.amp.cmap, vmin=self.spg.amp.vmindB, vmax=self.spg.amp.vmaxdB)
        if magnify:
            axs[1].set_ylim(self.spg.amp.fmin, self.spg.amp.fmax)
        if logfreq:
            axs[1].set_yscale("log")
        axs[1].set_ylabel("Frequency [Hz]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    def specgram_phase(self, NFFT=2**10, ovr=0.5, window="hann", dT=5e-3,
                       cmap="viridis", magnify=True,
                       fmin=False, fmax=False, logfreq=False,
                       pause=0., display=True, detrend="linear"):
        # choosing detrend="linear", the output spectrogram describes dphi fluctuation (subtracting constant velocity).

        self.spg.phase = calc.struct()

        self.spg.phase.NFFT = NFFT
        self.spg.phase.ovr = ovr
        self.spg.phase.NOV = int(self.spg.phase.NFFT * self.spg.phase.ovr)
        self.spg.phase.window = window
        self.spg.phase.dT = dT
        self.spg.phase.Fs = 1. / self.spg.phase.dT
        self.spg.phase.NSamp = int(self.spg.phase.dT / self.dT)
        # self.spg.phase.NSphase = int(self.Fs / self.spg.phase.Fs)
        self.spg.phase.Nsp = self.size // self.spg.phase.NSamp
        self.spg.phase.cmap = cmap

        self.spg.phase.t = self.t[:self.spg.phase.Nsp * self.spg.phase.NSamp].reshape((self.spg.phase.Nsp, self.spg.phase.NSamp)).mean(axis=-1)
        self.spg.phase.phaseIQarray = self.phaseIQ[:self.spg.phase.Nsp * self.spg.phase.NSamp].reshape((self.spg.phase.Nsp, self.spg.phase.NSamp))
        self.spg.phase.f, self.spg.phase.psd = welch(x=self.spg.phase.phaseIQarray, fs=self.Fs, window="hann",
                                                     nperseg=self.spg.phase.NFFT, noverlap=self.spg.phase.NOV,
                                                     return_onesided=True,
                                                     detrend=detrend, scaling="density",
                                                     axis=-1, average="mean")
        # self.spg.phase.f, self.spg.phase.lindetpsd = welch(x=self.spg.phase.phaseIQarray, fs=self.Fs, window="hann",
        #                                                    nperseg=self.spg.phase.NFFT, noverlap=self.spg.phase.NOV,
        #                                                    return_onesided=True,
        #                                                    detrend="linear", scaling="density",
        #                                                    axis=-1, average="mean")
        self.spg.phase.f = self.spg.phase.f
        self.spg.phase.psd = self.spg.phase.psd
        self.spg.phase.psddB = 10 * np.log10(self.spg.phase.psd)
        # self.spg.phase.lindetpsd = self.spg.phase.lindetpsd
        # self.spg.phase.lindetpsddB = 10 * np.log10(self.spg.phase.lindetpsd)
        self.spg.phase.dF = self.Fs / self.spg.phase.NFFT

        if magnify:
            if fmin:
                self.spg.phase.fmin = fmin
            else:
                self.spg.phase.fmin = self.spg.phase.dF
            if fmax:
                self.spg.phase.fmax = fmax
            else:
                self.spg.phase.fmax = self.Fs / 2

            _idx = \
                np.where((np.abs(self.spg.phase.f) > self.spg.phase.fmin) & (np.abs(self.spg.phase.f) < self.spg.phase.fmax))[0]
            self.spg.phase.vmax = self.spg.phase.psd[:, _idx].max()
            self.spg.phase.vmaxdB = 10 * np.log10(self.spg.phase.vmax)
            self.spg.phase.vmin = self.spg.phase.psd[:, _idx].min()
            self.spg.phase.vmindB = 10 * np.log10(self.spg.phase.vmin)
        else:
            self.spg.phase.vmax = self.spg.phase.psd.max()
            self.spg.phase.vmaxdB = 10 * np.log10(self.spg.phase.vmax)
            self.spg.phase.vmin = self.spg.phase.psd.min()
            self.spg.phase.vmindB = 10 * np.log10(self.spg.phase.vmin)

        if not display:
            matplotlib.use('Agg')

        figdir = os.path.join(self.dirbase, "phaseIQ")
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}" \
              f"_{self.spg.phase.NFFT}_{self.spg.phase.ovr}_{self.spg.phase.window}_{self.spg.phase.dT}"
        if fmin:
            fnm += f"_min{fmin*1e-3}kHz"
        if fmax:
            fnm += f"_max{fmax*1e-3}kHz"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"
        fig, axs = plt.subplots(nrows=2, sharex=True,
                                figsize=(5, 6), gridspec_kw={'height_ratios': [1, 3]},
                                num=fnm)

        axs[0].plot(self.t, self.phaseIQ, c="black", lw=0.1)
        axs[0].set_ylabel("phase [rad.]")
        p2p = self.phaseIQ.max() - self.phaseIQ.min()
        axs[0].set_ylim(self.phaseIQ.min() - p2p * 0.05, self.phaseIQ.max() + p2p * 0.05)

        axs[1].pcolormesh(np.append(self.spg.phase.t - 0.5 * self.spg.phase.dT,
                                    self.spg.phase.t[-1] + 0.5 * self.spg.phase.dT),
                          np.append(self.spg.phase.f - 0.5 * self.spg.phase.dF,
                                    self.spg.phase.f[-1] + 0.5 * self.spg.phase.dF),
                          self.spg.phase.psddB.T,
                          cmap=self.spg.phase.cmap, vmin=self.spg.phase.vmindB, vmax=self.spg.phase.vmaxdB)
        if magnify:
            axs[1].set_ylim(self.spg.phase.fmin, self.spg.phase.fmax)
        if logfreq:
            axs[1].set_yscale("log")
        axs[1].set_ylabel("Frequency [Hz]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    def centerofgravity(self, NFFT=2**6, ovr=0.5, window="hann", dT=1e-4,
                        cmap="viridis", magnify=True, fmin=1e3, fmax=500e3, pause=0.,
                        display=True, detrend="constant"):

        self.cog = calc.struct()

        self.cog.NFFT = NFFT
        self.cog.ovr = ovr
        self.cog.NOV = int(self.cog.NFFT * self.cog.ovr)
        self.cog.window = window
        self.cog.dT = dT
        self.cog.Fs = 1./self.cog.dT
        self.cog.NSamp = int(self.cog.dT / self.dT)
        # self.cog.NSamp = int(self.Fs / self.cog.Fs)
        self.cog.Nsp = self.size // self.cog.NSamp
        self.cog.cmap = cmap

        self.cog.t = self.t[:self.cog.Nsp * self.cog.NSamp].reshape((self.cog.Nsp, self.cog.NSamp)).mean(axis=-1)
        self.cog.IQarray = self.IQ[:self.cog.Nsp * self.cog.NSamp].reshape((self.cog.Nsp, self.cog.NSamp))
        self.cog.f, self.cog.psd = welch(x=self.cog.IQarray, fs=self.Fs, window="hann",
                                         nperseg=self.cog.NFFT, noverlap=self.cog.NOV,
                                         return_onesided=False,
                                         detrend=detrend, scaling="density",
                                         axis=-1, average="mean")
        # self.cog.f, self.cog.lindetpsd = welch(x=self.cog.IQarray, fs=self.Fs, window="hann",
        #                                        nperseg=self.cog.NFFT, noverlap=self.cog.NOV,
        #                                        return_onesided=False,
        #                                        detrend="linear", scaling="density",
        #                                        axis=-1, average="mean")
        self.cog.f = fft.fftshift(self.cog.f)
        self.cog.psd = fft.fftshift(self.cog.psd, axes=-1)
        self.cog.psddB = 10 * np.log10(self.cog.psd)
        # self.cog.lindetpsd = fft.fftshift(self.cog.lindetpsd, axes=-1)
        # self.cog.lindetpsddB = 10 * np.log10(self.cog.lindetpsd)
        self.cog.dF = self.dT * self.cog.NFFT

        if magnify:
            if fmin:
                self.cog.fmin = fmin
            else:
                self.cog.fmin = self.cog.dF
            if fmax:
                self.cog.fmax = fmax
            else:
                self.cog.fmax = self.Fs / 2

            _idx = np.where((np.abs(self.cog.f) > self.cog.fmin) & (np.abs(self.cog.f) < self.cog.fmax))[0]
            self.cog.vmax = self.cog.psd[:, _idx].max()
            self.cog.vmaxdB = 10*np.log10(self.cog.vmax)
            self.cog.vmin = self.cog.psd[:, _idx].min()
            self.cog.vmindB = 10*np.log10(self.cog.vmin)
        else:
            _idx = np.where((np.abs(self.cog.f) > self.cog.fmin) & (np.abs(self.cog.f) < self.cog.fmax))[0]
            self.cog.vmax = self.cog.psd.max()
            self.cog.vmaxdB = 10*np.log10(self.cog.vmax)
            self.cog.vmin = self.cog.psd.min()
            self.cog.vmindB = 10*np.log10(self.cog.vmin)

        self.cog.psd_use = self.cog.psd[:, _idx]
        self.cog.f_use = self.cog.f[_idx]
        self.cog.fd = np.average(np.tile(self.cog.f_use, (self.cog.psd_use.shape[0], 1)), weights=self.cog.psd_use,
                                  axis=-1)


        if not display:
            original_backend = matplotlib.get_backend()
            matplotlib.use('Agg')

        figdir = os.path.join(self.dirbase, "centerofgravity")
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}" \
              f"_{self.cog.NFFT}_{self.cog.ovr}_{self.cog.window}_{self.cog.dT}"
        if fmin:
            fnm += f"_min{fmin*1e-3}kHz"
        if fmax:
            fnm += f"_max{fmax*1e-3}kHz"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"
        fig, axs = plt.subplots(nrows=3, sharex=True,
                                figsize=(5, 10), gridspec_kw={'height_ratios': [1, 1, 3]},
                                num=fnm)

        axs[0].plot(self.t, self.I, c="black", lw=0.1)
        axs[0].set_ylabel("I [V]")
        if magnify:
            p2p = self.I.max() - self.I.min()
            axs[0].set_ylim(self.I.min() - p2p * 0.05, self.I.max() + p2p * 0.05)
        else:
            axs[0].set_ylim(float(self.Iprms["RangeLow"][0]), float(self.Iprms["RangeHigh"][0]))

        axs[1].plot(self.t, self.Q, c="black", lw=0.1)
        if magnify:
            p2p = self.Q.max() - self.Q.min()
            axs[1].set_ylim(self.Q.min() - p2p * 0.05, self.Q.max() + p2p * 0.05)
        else:
            axs[1].set_ylim(float(self.Qprms["RangeLow"][0]), float(self.Qprms["RangeHigh"][0]))
        axs[1].set_ylabel("Q [V]")

        axs[2].pcolormesh(np.append(self.cog.t - 0.5 * self.cog.dT, self.cog.t[-1] + 0.5 * self.cog.dT),
                          np.append(self.cog.f - 0.5 * self.cog.dF, self.cog.f[-1] + 0.5 * self.cog.dF),
                          self.cog.psddB.T,
                          cmap=self.cog.cmap, vmin=self.cog.vmindB, vmax=self.cog.vmaxdB)
        axs[2].fill_between(self.cog.t, -self.cog.fmin, self.cog.fmin, color = "white")
        axs[2].plot(self.cog.t, self.cog.fd, color="red", lw=0.1)
        if magnify:
            axs[2].set_ylim(- self.cog.fmax, self.cog.fmax)
        axs[2].set_ylabel("Frequency [Hz]")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)
            matplotlib.use(original_backend)

    def calc_vperp(self, diag="comb_R", ch=1):

        # diag = highK, comb_R, comb_U

        if diag == "comb_R":

            self.lay = eg_mwrm.comb_R(self.sn, self.subsn)
            self.lay.t_window_RAY_OUT(self.tstart, self.tend)
            self.lay_freq = self.lay.chs[ch - 1].freq
            self.tkp = self.lay.chs[ch - 1].RAY_OUT.twin.time
            self.kp = self.lay.chs[ch - 1].RAY_OUT.twin.kp * 1e2
            self.kp_err = self.lay.chs[ch - 1].RAY_OUT.twin.kp_error * 1e2
            self.kp_avg = self.lay.chs[ch - 1].RAY_OUT.twin.avg.kp * 1e2
            self.kp_std = self.lay.chs[ch - 1].RAY_OUT.twin.std.kp * 1e2
            self.kp_ste = self.lay.chs[ch - 1].RAY_OUT.twin.ste.kp * 1e2

            self.cog.fd_avg, self.cog.fd_std, self.cog.fd_ste \
                = calc.timeAverageDatByRefs_v2(self.cog.t, self.cog.fd, self.tkp)

            self.cog.vp, self.cog.vp_err \
                = calc.divide(2 * np.pi * self.cog.fd_avg, self.kp, 2 * np.pi * self.cog.fd_std, self.kp_err)
            self.cog.vp_fast, self.cog.vp_fast_err \
                = calc.divide(2 * np.pi * self.cog.fd, self.kp_avg, d_err=self.kp_std)

        elif diag == "highK":

            self.lay = eg_mwrm.highK(self.sn, self.subsn)
            self.lay.t_window_rho_k(self.tstart, self.tend)
            self.tkp = self.lay.chs[ch-1].twin.t
            self.kp_avg = self.lay.chs[ch - 1].twin.avg.kp * 1e2
            self.kp_std = self.lay.chs[ch - 1].twin.std.kp * 1e2
            self.kp_ste = self.lay.chs[ch - 1].twin.ste.kp * 1e2

            self.cog.vp_fast, self.cog.vp_fast_err \
                = calc.divide(2 * np.pi * self.cog.fd, self.kp_avg, d_err=self.kp_std)

        else:
            print("未実装!!")
            exit()


    def pulsepair(self, ovr=0.5, dT=1e-4):
        self.pp = calc.struct()
        self.pp.ovr = ovr
        self.pp.dT = dT
        self.pp.NSamp = int(self.pp.dT / ovr / self.dT)
        self.pp.Fs = 1./self.pp.dT
        o = calc.pulsepair(self.t, self.IQ, Nsample=self.pp.NSamp, ovr=self.pp.ovr)
        self.pp.t = o.t
        self.pp.fd = o.fd
        self.pp.fdstd = o.fdstd

    # def gaussfit(self, numgauss=1, fmin=1e3, fmax=500e3, p0_list=[(1, 1, 1)]):
    #     calc.shifted_gauss()

    def specgram_pp(self, NFFT=2 ** 7, ovr=0.5, window="hann", dT=2e-2,
                     cmap="viridis", magnify=True,
                     fmin=False, fmax=False, logfreq=False,
                     pause=0., display=True, detrend="constant"):
        # choosing detrend="linear", the output spectrogram describes dphi fluctuation (subtracting constant velocity).

        self.pp.spg = calc.struct()

        self.pp.spg.NFFT = NFFT
        self.pp.spg.ovr = ovr
        self.pp.spg.NOV = int(self.pp.spg.NFFT * self.pp.spg.ovr)
        self.pp.spg.window = window
        self.pp.spg.dT = dT
        self.pp.spg.Fs = 1. / self.pp.spg.dT
        self.pp.spg.NSamp = int(self.pp.spg.dT / self.pp.dT)
        # self.pp.spg.NSphase = int(self.Fs / self.pp.spg.Fs)
        self.pp.spg.Nsp = self.pp.fd.size // self.pp.spg.NSamp
        self.pp.spg.cmap = cmap

        self.pp.spg.t = self.pp.t[:self.pp.spg.Nsp * self.pp.spg.NSamp].reshape(
            (self.pp.spg.Nsp, self.pp.spg.NSamp)).mean(axis=-1)
        self.pp.spg.cogarray = self.pp.fd[:self.pp.spg.Nsp * self.pp.spg.NSamp].reshape(
            (self.pp.spg.Nsp, self.pp.spg.NSamp))
        self.pp.spg.f, self.pp.spg.psd = welch(x=self.pp.spg.cogarray, fs=self.pp.Fs, window=window,
                                                 nperseg=self.pp.spg.NFFT, noverlap=self.pp.spg.NOV,
                                                 return_onesided=True,
                                                 detrend=detrend, scaling="density",
                                                 axis=-1, average="mean")
        self.pp.spg.psddB = 10 * np.log10(self.pp.spg.psd)
        self.pp.spg.dF = self.dT * self.pp.spg.NFFT

        if magnify:
            if fmin:
                self.pp.spg.fmin = fmin
            else:
                self.pp.spg.fmin = self.pp.spg.dF
            if fmax:
                self.pp.spg.fmax = fmax
            else:
                self.pp.spg.fmax = self.pp.Fs / 2

            _idx = \
                np.where((np.abs(self.pp.spg.f) > self.pp.spg.fmin) & (
                        np.abs(self.pp.spg.f) < self.pp.spg.fmax))[0]
            self.pp.spg.vmax = self.pp.spg.psd[:, _idx].max()
            self.pp.spg.vmaxdB = 10 * np.log10(self.pp.spg.vmax)
            self.pp.spg.vmin = self.pp.spg.psd[:, _idx].min()
            self.pp.spg.vmindB = 10 * np.log10(self.pp.spg.vmin)
        else:
            self.pp.spg.vmax = self.pp.spg.psd.max()
            self.pp.spg.vmaxdB = 10 * np.log10(self.pp.spg.vmax)
            self.pp.spg.vmin = self.pp.spg.psd.min()
            self.pp.spg.vmindB = 10 * np.log10(self.pp.spg.vmin)

        if not display:
            matplotlib.use('Agg')

        figdir = os.path.join(self.dirbase, "specgram_pp")
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}" \
              f"_{self.pp.spg.NFFT}_{self.pp.spg.ovr}_{self.pp.spg.window}_{self.pp.spg.dT}"
        if fmin:
            fnm += f"_min{fmin * 1e-3}kHz"
        if fmax:
            fnm += f"_max{fmax * 1e-3}kHz"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}" \
                f"{self.pp.spg.NFFT} {self.pp.spg.ovr} {self.pp.spg.window} {self.pp.spg.dT}s"
        fig, axs = plt.subplots(nrows=2, sharex=True,
                                figsize=(5, 6), gridspec_kw={'height_ratios': [1, 3]},
                                num=fnm)

        axs[0].errorbar(self.pp.t, self.pp.fd, self.pp.fdstd, c="black", ecolor="grey", lw=0.1)
        axs[0].set_ylabel("center of gravity [Hz]")
        p2p = self.pp.fd.max() - self.pp.fd.min()
        axs[0].set_ylim(self.pp.fd.min() - p2p * 0.05, self.pp.fd.max() + p2p * 0.05)

        axs[1].pcolormesh(
            np.append(self.pp.spg.t - 0.5 * self.pp.spg.dT, self.pp.spg.t[-1] + 0.5 * self.pp.spg.dT),
            np.append(self.pp.spg.f - 0.5 * self.pp.spg.dF, self.pp.spg.f[-1] + 0.5 * self.pp.spg.dF),
            self.pp.spg.psddB.T,
            cmap=self.pp.spg.cmap, vmin=self.pp.spg.vmindB, vmax=self.pp.spg.vmaxdB)
        if magnify:
            axs[1].set_ylim(self.pp.spg.fmin, self.pp.spg.fmax)
        if logfreq:
            axs[1].set_yscale("log")
        axs[1].set_ylabel("Frequency [Hz]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    def specgram_cog(self, NFFT=2**7, ovr=0.5, window="hann", dT=2e-2,
                       cmap="viridis", magnify=True,
                       fmin=False, fmax=False, logfreq=False,
                       pause=0., display=True, detrend="constant"):
        # choosing detrend="linear", the output spectrogram describes dphi fluctuation (subtracting constant velocity).

        self.cog.spg = calc.struct()

        self.cog.spg.NFFT = NFFT
        self.cog.spg.ovr = ovr
        self.cog.spg.NOV = int(self.cog.spg.NFFT * self.cog.spg.ovr)
        self.cog.spg.window = window
        self.cog.spg.dT = dT
        self.cog.spg.Fs = 1. / self.cog.spg.dT
        self.cog.spg.NSamp = int(self.cog.spg.dT / self.cog.dT)
        # self.cog.spg.NSphase = int(self.Fs / self.cog.spg.Fs)
        self.cog.spg.Nsp = self.cog.fd.size // self.cog.spg.NSamp
        self.cog.spg.cmap = cmap

        self.cog.spg.t = self.cog.t[:self.cog.spg.Nsp * self.cog.spg.NSamp].reshape(
            (self.cog.spg.Nsp, self.cog.spg.NSamp)).mean(axis=-1)
        self.cog.spg.cogarray = self.cog.fd[:self.cog.spg.Nsp * self.cog.spg.NSamp].reshape(
            (self.cog.spg.Nsp, self.cog.spg.NSamp))
        self.cog.spg.f, self.cog.spg.psd = welch(x=self.cog.spg.cogarray, fs=self.cog.Fs, window="hann",
                                                 nperseg=self.cog.spg.NFFT, noverlap=self.cog.spg.NOV,
                                                 return_onesided=True,
                                                 detrend=detrend, scaling="density",
                                                 axis=-1, average="mean")
        # self.cog.spg.f, self.cog.spg.lindetpsd = welch(x=self.cog.spg.phaseIQarray, fs=self.Fs, window="hann",
        #                                                    nperseg=self.cog.spg.NFFT, noverlap=self.cog.spg.NOV,
        #                                                    return_onesided=True,
        #                                                    detrend="linear", scaling="density",
        #                                                    axis=-1, average="mean")
        self.cog.spg.psddB = 10 * np.log10(self.cog.spg.psd)
        # self.cog.spg.lindetpsd = self.cog.spg.lindetpsd
        # self.cog.spg.lindetpsddB = 10 * np.log10(self.cog.spg.lindetpsd)
        self.cog.spg.dF = self.dT * self.cog.spg.NFFT

        if magnify:
            if fmin:
                self.cog.spg.fmin = fmin
            else:
                self.cog.spg.fmin = self.cog.spg.dF
            if fmax:
                self.cog.spg.fmax = fmax
            else:
                self.cog.spg.fmax = self.cog.Fs / 2

            _idx = \
                np.where((np.abs(self.cog.spg.f) > self.cog.spg.fmin) & (
                            np.abs(self.cog.spg.f) < self.cog.spg.fmax))[0]
            self.cog.spg.vmax = self.cog.spg.psd[:, _idx].max()
            self.cog.spg.vmaxdB = 10 * np.log10(self.cog.spg.vmax)
            self.cog.spg.vmin = self.cog.spg.psd[:, _idx].min()
            self.cog.spg.vmindB = 10 * np.log10(self.cog.spg.vmin)
        else:
            self.cog.spg.vmax = self.cog.spg.psd.max()
            self.cog.spg.vmaxdB = 10 * np.log10(self.cog.spg.vmax)
            self.cog.spg.vmin = self.cog.spg.psd.min()
            self.cog.spg.vmindB = 10 * np.log10(self.cog.spg.vmin)

        if not display:
            matplotlib.use('Agg')

        figdir = os.path.join(self.dirbase, "specgram_cog")
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}" \
              f"_{self.cog.spg.NFFT}_{self.cog.spg.ovr}_{self.cog.spg.window}_{self.cog.spg.dT}"
        if fmin:
            fnm += f"_min{fmin * 1e-3}kHz"
        if fmax:
            fnm += f"_max{fmax * 1e-3}kHz"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}\n" \
                f"{self.cog.spg.NFFT} {self.cog.spg.ovr} {self.cog.spg.window} {self.cog.spg.dT}s"
        fig, axs = plt.subplots(nrows=2, sharex=True,
                                figsize=(5, 6), gridspec_kw={'height_ratios': [1, 3]},
                                num=fnm)

        axs[0].plot(self.cog.t, self.cog.fd, c="black", lw=0.1)
        axs[0].set_ylabel("center of gravity [Hz]")
        p2p = self.cog.fd.max() - self.cog.fd.min()
        axs[0].set_ylim(self.cog.fd.min() - p2p * 0.05, self.cog.fd.max() + p2p * 0.05)

        axs[1].pcolormesh(
            np.append(self.cog.spg.t - 0.5 * self.cog.spg.dT, self.cog.spg.t[-1] + 0.5 * self.cog.spg.dT),
            np.append(self.cog.spg.f - 0.5 * self.cog.spg.dF, self.cog.spg.f[-1] + 0.5 * self.cog.spg.dF),
            self.cog.spg.psddB.T,
            cmap=self.cog.spg.cmap, vmin=self.cog.spg.vmindB, vmax=self.cog.spg.vmaxdB)
        if magnify:
            axs[1].set_ylim(self.cog.spg.fmin, self.cog.spg.fmax)
        if logfreq:
            axs[1].set_yscale("log")
        axs[1].set_ylabel("Frequency [Hz]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    def create_frame(self, i, ax):
        line = ax.plot(self.spg.f, self.spg.psddB[i], color='black')[0]
        text = ax.text(0., self.spg.vmaxdB - 5, f'{self.spg.t[i]:.3f}s', ha='center', va='center', fontsize=12)
        return line, text

    def create_frame_bgon(self, i, ax):
        t1idx = np.searchsorted(self.bg.t, self.spg.t[i])
        t0idx = t1idx - 1
        line = ax.plot(self.spg.f, self.spg.psddB[i], color='black')[0]
        text = ax.text(0., self.spg.vmaxdB - 5, f'{self.spg.t[i]:.3f}s', ha='center', va='center', fontsize=12)
        bg0 = ax.plot(self.spg.f, self.bg.psd[t0idx], color="grey", ls="--")[0]
        bg1 = ax.plot(self.spg.f, self.bg.psd[t1idx], color="grey", ls="--")[0]
        return line, text, bg0, bg1

    def spec_anime(self, speedrate=1., bgon=False):
        fname = f"{self.sn}_{self.subsn}_{self.diagname}_{self.chI}_{self.chQ}"
        fig, ax = plt.subplots(num=fname)
        ax.set_xlim(- self.spg.fmax, self.spg.fmax)
        ax.set_ylim(self.spg.vmindB, self.spg.vmaxdB)

        if bgon:
            frames = [self.create_frame_bgon(i, ax) for i in range(self.spg.Nsp)]
        else:
            frames = [self.create_frame(i, ax) for i in range(self.spg.Nsp)]
        ani = ArtistAnimation(fig, frames, interval=int(self.spg.dT / speedrate * 1e3), blit=True)
        # ani = ArtistAnimation(fig, frames, interval=50, blit=True)
        # fig.legend()

        figdir = "Retrieve_MWRM_spec_anime"
        path = os.path.join(figdir, f"{fname}.gif")
        ani.save(path)
        plt.show(ani)

    def spectrum(self, tstart=4.9, tend=5.0, NFFT=2**10, ovr=0.5, window="hann", detrend="constant",
                 fmin=None, fmax=None,
                 pause=0., display=True, bgon=False):

        self.sp = calc.struct()
        self.sp.tstart = tstart
        self.sp.tend = tend

        self.sp.NFFT = NFFT
        self.sp.ovr = ovr
        self.sp.NOV = int(self.sp.NFFT * self.sp.ovr)
        self.sp.window = window

        _, datlist = proc.getTimeIdxsAndDats(self.t, self.sp.tstart, self.sp.tend,
                                             [self.t, self.IQ, self.I, self.Q])
        self.sp.traw, self.sp.IQraw, self.sp.Iraw, self.sp.Qraw = datlist
        self.sp.NSamp = self.sp.traw.size
        self.sp.dF = self.Fs / self.sp.NFFT

        self.sp.t = (self.sp.tstart + self.sp.tend) / 2
        self.sp.fIQ, self.sp.psdIQ = welch(x=self.sp.IQraw, fs=self.Fs, window="hann",
                                           nperseg=self.sp.NFFT, noverlap=self.sp.NOV,
                                           detrend=detrend, scaling="density",
                                           average="mean", return_onesided=False)
        self.sp.fIQ = fft.fftshift(self.sp.fIQ)
        self.sp.psdIQ = fft.fftshift(self.sp.psdIQ)
        self.sp.psdIQdB = 10 * np.log10(self.sp.psdIQ)

        if fmin is not None:
            self.sp.fmin = fmin
        else:
            self.sp.fmin = self.sp.dF
        if fmax is not None:
            self.sp.fmax = fmax
        else:
            self.sp.fmax = self.Fs / 2

        self.sp.vmindB = np.min(self.sp.psdIQdB)
        self.sp.vmaxdB = np.max(self.sp.psdIQdB)

        self.sp.fI, self.sp.psdI = welch(x=self.sp.Iraw, fs=self.Fs, window="hann",
                                         nperseg=self.sp.NFFT, noverlap=self.sp.NOV,
                                         detrend="constant", scaling="density",
                                         average="mean")
        self.sp.fQ, self.sp.psdQ = welch(x=self.sp.Qraw, fs=self.Fs, window="hann",
                                         nperseg=self.sp.NFFT, noverlap=self.sp.NOV,
                                         detrend="constant", scaling="density",
                                         average="mean")
        self.sp.psdIdB = 10 * np.log10(self.sp.psdI)
        self.sp.psdQdB = 10 * np.log10(self.sp.psdQ)

        figdir = "Retrieve_MWRM_spectrum"
        proc.ifNotMake(figdir)

        fnm_base = f"{self.sn}_{self.subsn}_{self.sp.tstart}_{self.sp.tend}_{self.diagname}_{self.chI}_{self.chQ}"
        if fmin is not None:
            fnm_base += f"_min{fmin * 1e-3}kHz"
        if fmax is not None:
            fnm_base += f"_max{fmax * 1e-3}kHz"

        fnm = f"{fnm_base}_IQsp"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.sp.t}s\n" \
                f"({self.sp.tstart}-{self.sp.tend}s)\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"

        if not display:
            original_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=(6, 6), num=fnm)
        ax.plot(self.sp.fIQ, self.sp.psdIQdB, c="black")
        if bgon:
            self.BSBackground(self.sp.NFFT, self.sp.ovr)
            tidxs, datlist = proc.getTimeIdxsAndDats(self.bg.t, self.sp.tstart, self.sp.tend, [self.bg.t, self.bg.psd])
            self.sp.tbg, self.sp.psdbg = datlist
            self.sp.bg, _, self.sp.bg_err = calc.average(self.sp.psdbg, axis=0)
            self.sp.bg_dB, self.sp.bg_err_dB = calc.dB(self.sp.bg, self.sp.bg_err)
            self.sp.snr = self.sp.psdIQdB - self.sp.bg_dB
            # for i in range(len(self.sp.tbg)):
            #     ax.scatter(self.bg.f, 10 * np.log10(self.sp.psdbg[i]), marker=".", c="grey")
            ax.errorbar(self.bg.f, self.sp.bg_dB, self.sp.bg_err_dB, color="grey", ecolor="lightgrey")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [dBV/$\sqrt{\\rm{Hz}}$]")
        ax.set_xlim(- self.sp.fmax, self.sp.fmax)
        # ax.set_ylim()

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

        fnm2 = f"{fnm_base}_eachsp"
        path2 = os.path.join(figdir, f"{fnm2}.png")
        title2 = f"#{self.sn}-{self.subsn} {self.sp.t}s\n" \
                 f"({self.sp.tstart}-{self.sp.tend}s)\n" \
                 f"{self.diagname} ch:{self.chI},{self.chQ}"
        fig2, ax2 = plt.subplots(figsize=(6, 6), num=fnm2)
        ax2.plot(self.sp.fI, self.sp.psdIdB, c="blue")
        ax2.plot(self.sp.fQ, self.sp.psdQdB, c="orange")
        ax2.legend(["I", "Q"])
        ax2.set_xscale("log")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("PSD [dBV/$\sqrt{\\rm{Hz}}$]")
        ax2.set_xlim(self.sp.fmin, self.sp.fmax)

        plot.caption(fig2, title2, hspace=0.1, wspace=0.1)
        plot.capsave(fig2, title2, fnm2, path2)

        if display:
            plot.check(pause)
        else:
            plot.close(fig2)

        if bgon:
            fnm3 = f"{fnm_base}_SNR"
            path3 = os.path.join(figdir, f"{fnm3}.png")
            title3 = f"#{self.sn}-{self.subsn} {self.sp.t}s\n" \
                     f"({self.sp.tstart}-{self.sp.tend}s)\n" \
                     f"{self.diagname} ch:{self.chI},{self.chQ}"
            fig3, ax3 = plt.subplots(1, num=fnm3)

            ax3.plot(self.sp.fIQ, self.sp.snr, c="black")
            ax3.set_xlabel("Frequency [Hz]")
            ax3.set_ylabel("S/N [dB]")
            ax3.set_ylim(bottom=0)
            ax3.set_xlim(- self.sp.fmax, self.sp.fmax)

            plot.caption(fig3, title3, hspace=0.1, wspace=0.1)
            plot.capsave(fig3, title3, fnm3, path3)

            if display:
                plot.check(pause)
            else:
                plot.close(fig3)

        if not display:
            matplotlib.use(original_backend)

        return self.sp

    def spectrum_asymmetric_component(self, fmin=None, fmax=None,
                                      peak_detection=True, gaussfitting=True,
                                      polyorder=2, prominence=3, iniwidth=40e3,
                                      maxfev=2000, bydB=True,
                                      display=True, pause=0.):

        self.sp.asym = calc.struct()
        self.sp.asym.iniwidth = iniwidth

        if fmin is None:
            self.sp.asym.fmin = self.sp.dF
        else:
            self.sp.asym.fmin = fmin
        if fmax is None:
            self.sp.asym.fmax = self.Fs / 2
        else:
            self.sp.asym.fmax = fmax
        _idx = np.where((np.abs(self.sp.fIQ) > self.sp.asym.fmin)
                        & (np.abs(self.sp.fIQ) < self.sp.asym.fmax))[0]

        self.sp.asym.srcpsddB = self.sp.psdIQdB[_idx]
        self.sp.asym.f = self.sp.fIQ[_idx]
        self.sp.asym.winlen = int(self.sp.asym.iniwidth / self.sp.dF / 2) * 2 + 1
        if bydB:
            self.sp.asym.psddB = self.sp.psdIQdB - np.flip(self.sp.psdIQdB, axis=-1)

            if peak_detection:
                # peak detection
                self.sp.asym.psddB_smooth = savgol_filter(self.sp.asym.psddB[_idx],
                                                          window_length=self.sp.asym.winlen,
                                                          polyorder=polyorder)
                self.sp.asym.peak_idxs = find_peaks(self.sp.asym.psddB_smooth, prominence=prominence,
                                                    height=0)[0]
                self.sp.asym.peak_freqs = self.sp.fIQ[_idx][self.sp.asym.peak_idxs]
                self.sp.asym.peak_psddB = self.sp.asym.psddB_smooth[self.sp.asym.peak_idxs]
                self.sp.asym.peak_nums = np.array(self.sp.asym.peak_idxs).size

            if gaussfitting:
                # gauss fitting
                if self.sp.asym.peak_nums == 0:
                    self.sp.asym.gaussfit_num = 0
                    self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                        = [np.nan, np.nan, np.nan]
                    self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                        = [np.nan, np.nan, np.nan]
                elif self.sp.asym.peak_nums == 1:
                    self.sp.asym.gaussfit_num = 1
                    inip = [self.sp.asym.peak_psddB[0], iniwidth, self.sp.asym.peak_freqs[0]]
                    try:
                        popt, pcov = curve_fit(oddgauss, self.sp.fIQ[_idx], self.sp.asym.psddB[_idx],
                                               p0=inip, maxfev=maxfev)
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit = popt
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = np.sqrt(np.diag(pcov))
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = [np.nan, np.nan, np.nan]
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = [np.nan, np.nan, np.nan]
                elif self.sp.asym.peak_nums == 2:
                    self.sp.asym.gaussfit_num = 2
                    inip = [self.sp.asym.peak_psddB, [iniwidth, iniwidth], self.sp.asym.peak_freqs]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss2, self.sp.fIQ[_idx], self.sp.asym.psddB[_idx],
                                               p0=inip, maxfev=maxfev)
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = popt.reshape((self.sp.asym.gaussfit_num, 3)).T
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = np.sqrt(np.diag(pcov)).reshape((self.sp.asym.gaussfit_num, 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = [np.nan, np.nan, np.nan]
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = [np.nan, np.nan, np.nan]
                elif self.sp.asym.peak_nums == 3:
                    self.sp.asym.gaussfit_num = 3
                    inip = [self.sp.asym.peak_psddB, [iniwidth, iniwidth, iniwidth], self.sp.asym.peak_freqs]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss3, self.sp.fIQ[_idx], self.sp.asym.psddB[_idx],
                                               p0=inip, maxfev=maxfev)
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = popt.reshape((self.sp.asym.peak_nums, 3)).T
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = np.sqrt(np.diag(pcov)).reshape((self.sp.asym.peak_nums, 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = [np.nan, np.nan, np.nan]
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = [np.nan, np.nan, np.nan]
                elif self.sp.asym.peak_nums == 4:
                    self.sp.asym.gaussfit_num = 4
                    inip = [self.sp.asym.peak_psddB, [iniwidth, iniwidth, iniwidth, iniwidth], self.sp.asym.peak_freqs]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss4, self.sp.fIQ[_idx], self.sp.asym.psddB[_idx],
                                               p0=inip, maxfev=maxfev)
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = popt.reshape((self.sp.asym.gaussfit_num, 3)).T
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = np.sqrt(np.diag(pcov)).reshape((self.sp.asym.gaussfit_num, 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = [np.nan, np.nan, np.nan]
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = [np.nan, np.nan, np.nan]
                else:  # until 4.
                    self.sp.asym.gaussfit_num = 4
                    idx_str = np.argsort(self.sp.asym.peak_psddB)[- self.sp.asym.gaussfit_num:]
                    inip = [self.sp.asym.peak_psddB[idx_str],
                            [iniwidth, iniwidth, iniwidth, iniwidth],
                            self.sp.asym.peak_freqs[idx_str]]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss4, self.sp.fIQ[_idx], self.sp.asym.psddB[_idx],
                                               p0=inip, maxfev=maxfev)
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = popt.reshape((self.sp.asym.gaussfit_num, 3)).T
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = np.sqrt(np.diag(pcov)).reshape((self.sp.asym.gaussfit_num, 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = [np.nan, np.nan, np.nan]
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = [np.nan, np.nan, np.nan]
        else:
            self.sp.asym.psd = self.sp.psdIQ - np.flip(self.sp.psdIQ)

            if peak_detection:
                # peak detection
                self.sp.asym.psd_smooth = savgol_filter(self.sp.asym.psd[_idx],
                                                          window_length=self.sp.asym.winlen,
                                                          polyorder=polyorder)
                self.sp.asym.peak_idxs = find_peaks(self.sp.asym.psd_smooth, prominence=prominence)[0]
                self.sp.asym.peak_freqs = self.sp.fIQ[_idx][self.sp.asym.peak_idxs]
                self.sp.asym.peak_psd = self.sp.asym.psd_smooth[self.sp.asym.peak_idxs]
                self.sp.asym.peak_nums = np.array(self.sp.asym.peak_idxs).size

            if gaussfitting:
                # gauss fitting
                if self.sp.asym.peak_nums == 0:
                    self.sp.asym.gaussfit_num = 0
                    self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                        = [np.nan, np.nan, np.nan]
                    self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                        = [np.nan, np.nan, np.nan]
                elif self.sp.asym.peak_nums == 1:
                    self.sp.asym.gaussfit_num = 1
                    inip = [self.sp.asym.peak_psd[0], iniwidth, self.sp.asym.peak_freqs[0]]
                    try:
                        popt, pcov = curve_fit(oddgauss, self.sp.fIQ[_idx], self.sp.asym.psd[_idx],
                                               p0=inip, maxfev=maxfev)
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit = popt
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = np.sqrt(np.diag(pcov))
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = [np.nan, np.nan, np.nan]
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = [np.nan, np.nan, np.nan]
                elif self.sp.asym.peak_nums == 2:
                    self.sp.asym.gaussfit_num = 2
                    inip = [self.sp.asym.peak_psd, [iniwidth, iniwidth], self.sp.asym.peak_freqs]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss2, self.sp.fIQ[_idx], self.sp.asym.psd[_idx],
                                               p0=inip, maxfev=maxfev)
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = popt.reshape((self.sp.asym.peak_nums, 3)).T
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = np.sqrt(np.diag(pcov)).reshape((self.sp.asym.peak_nums, 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = [np.nan, np.nan, np.nan]
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = [np.nan, np.nan, np.nan]
                elif self.sp.asym.peak_nums == 3:
                    self.sp.asym.gaussfit_num = 3
                    inip = [self.sp.asym.peak_psd, [iniwidth, iniwidth, iniwidth], self.sp.asym.peak_freqs]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss3, self.sp.fIQ[_idx], self.sp.asym.psd[_idx],
                                               p0=inip, maxfev=maxfev)
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = popt.reshape((self.sp.asym.peak_nums, 3)).T
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = np.sqrt(np.diag(pcov)).reshape((self.sp.asym.peak_nums, 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = [np.nan, np.nan, np.nan]
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = [np.nan, np.nan, np.nan]
                elif self.sp.asym.peak_nums == 4:
                    self.sp.asym.gaussfit_num = 4
                    inip = [self.sp.asym.peak_psd, [iniwidth, iniwidth, iniwidth, iniwidth], self.sp.asym.peak_freqs]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss4, self.sp.fIQ[_idx], self.sp.asym.psd[_idx],
                                               p0=inip, maxfev=maxfev)
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = popt.reshape((self.sp.asym.gaussfit_num, 3)).T
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = np.sqrt(np.diag(pcov)).reshape((self.sp.asym.gaussfit_num, 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = [np.nan, np.nan, np.nan]
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = [np.nan, np.nan, np.nan]
                else:  # until 4.
                    self.sp.asym.gaussfit_num = 4
                    idx_str = np.argsort(self.sp.asym.peak_psd)[- self.sp.asym.gaussfit_num:]
                    inip = [self.sp.asym.peak_psd[idx_str],
                            [iniwidth, iniwidth, iniwidth, iniwidth],
                            self.sp.asym.peak_freqs[idx_str]]
                    inip = np.array(inip).flatten(order="F").tolist()
                    try:
                        popt, pcov = curve_fit(oddgauss4, self.sp.fIQ[_idx], self.sp.asym.psd[_idx],
                                               p0=inip, maxfev=maxfev)
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = popt.reshape((self.sp.asym.gaussfit_num, 3)).T
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = np.sqrt(np.diag(pcov)).reshape((self.sp.asym.gaussfit_num, 3)).T
                    except RuntimeError as e:
                        print(f"Optimal parameters not found: {e}")
                        self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit \
                            = [np.nan, np.nan, np.nan]
                        self.sp.asym.peak_fit_err, self.sp.asym.width_fit_err, self.sp.asym.fD_fit_err \
                            = [np.nan, np.nan, np.nan]

        if bydB:
            figdir = "Retrieve_MWRM_spectrum_asym"
            proc.ifNotMake(figdir)
            fnm = f"{self.sn}_{self.subsn}_{self.sp.tstart}_{self.sp.tend}_{self.diagname}_{self.chI}_{self.chQ}_dB"
            if fmin is not None:
                fnm += f"_min{self.sp.asym.fmin * 1e-3}kHz"
            if fmax is not None:
                fnm += f"_max{self.sp.asym.fmax * 1e-3}kHz"
            path = os.path.join(figdir, f"{fnm}.png")
            title = f"#{self.sn}-{self.subsn} {self.sp.tstart}-{self.sp.tend}s\n" \
                    f"{self.diagname} ch:{self.chI},{self.chQ}"
            fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(5, 8), num=fnm)

            self.sp.asym.vmaxdB = self.sp.asym.psddB[_idx].max()
            self.sp.asym.vmindB = self.sp.asym.psddB[_idx].min()

            ax.plot(self.sp.fIQ, self.sp.asym.psddB, ".", c="grey", ms=1)
            ax2.plot(self.sp.fIQ, self.sp.psdIQdB, ".", c="grey", ms=1)
            ax.plot(self.sp.fIQ[_idx], self.sp.asym.psddB_smooth, c="blue", lw=2, alpha=0.5)
            ax.hlines(0, - self.sp.asym.fmax, self.sp.asym.fmax, colors="grey", ls="--", lw=1)

            # if peak_detection:
                # ax.vlines(self.sp.asym.peak_freqs, self.sp.asym.vmindB, self.sp.asym.vmaxdB,
                #           colors="red", alpha=0.5, lw=1, ls="--")
            if gaussfitting:
                if (~isinstance(self.sp.asym.fD_fit, np.ndarray)) and (self.sp.asym.fD_fit is np.nan):
                    print("\n")
                else:
                    ax.vlines(self.sp.asym.fD_fit, self.sp.asym.vmindB, self.sp.asym.vmaxdB,
                              colors="pink", lw=1, ls="--")
                    if self.sp.asym.gaussfit_num == 1:
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit,
                                                      self.sp.asym.width_fit, self.sp.asym.fD_fit), c="lightblue", lw=2)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit,
                                                    self.sp.asym.width_fit, self.sp.asym.fD_fit, self.sp.vmindB),
                                 c="lightblue", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit,
                                                      self.sp.asym.width_fit, self.sp.asym.fD_fit),
                                 c="green", lw=1, alpha=0.5, ls="--")
                        ax2.plot(self.sp.fIQ, self.sp.psdIQdB - gauss(self.sp.fIQ, self.sp.asym.peak_fit,
                                                                      self.sp.asym.width_fit, self.sp.asym.fD_fit,
                                                                      0),
                                 c="green", lw=1, alpha=0.5, ls="--")
                        ax2.plot(- self.sp.fIQ, self.sp.psdIQdB - gauss(self.sp.fIQ, self.sp.asym.peak_fit,
                                                                      self.sp.asym.width_fit, self.sp.asym.fD_fit,
                                                                      0),
                                 c="purple", lw=1, alpha=0.5, ls="--")
                    elif self.sp.asym.gaussfit_num == 2:
                        ax.plot(self.sp.fIQ, oddgauss2(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                       self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0],
                                                       self.sp.asym.peak_fit[1],
                                                       self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1],
                                                       ),
                                c="green", lw=1, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                      self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0]),
                                c="lightblue", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], self.sp.vmindB),
                                c="lightblue", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                      self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1]),
                                c="lightgreen", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], self.sp.vmindB),
                                c="lightgreen", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ,
                                 self.sp.psdIQdB
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0),
                                 c="green", lw=1, alpha=0.5, ls="--")
                        ax2.plot(- self.sp.fIQ,
                                 self.sp.psdIQdB
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0),
                                 c="purple", lw=1, alpha=0.5, ls="--")
                    elif self.sp.asym.gaussfit_num == 3:
                        ax.plot(self.sp.fIQ, oddgauss3(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                       self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0],
                                                       self.sp.asym.peak_fit[1],
                                                       self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1],
                                                       self.sp.asym.peak_fit[2],
                                                       self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2],
                                                       ),
                                c="green", lw=1, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                      self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0]),
                                c="lightblue", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], self.sp.vmindB),
                                c="lightblue", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                      self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1]),
                                c="lightgreen", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], self.sp.vmindB),
                                c="lightgreen", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                      self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2]),
                                c="plum", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                    self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], self.sp.vmindB),
                                c="plum", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ,
                                 self.sp.psdIQdB
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                    self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], 0),
                                 c="green", lw=1, alpha=0.5, ls="--")
                        ax2.plot(- self.sp.fIQ,
                                 self.sp.psdIQdB
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                    self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], 0),
                                 c="purple", lw=1, alpha=0.5, ls="--")
                    elif self.sp.asym.gaussfit_num == 4:
                        ax.plot(self.sp.fIQ, oddgauss4(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                       self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0],
                                                       self.sp.asym.peak_fit[1],
                                                       self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1],
                                                       self.sp.asym.peak_fit[2],
                                                       self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2],
                                                       self.sp.asym.peak_fit[3],
                                                       self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3]
                                                       ),
                                c="green", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                      self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0]),
                                c="lightblue", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], self.sp.vmindB),
                                c="lightblue", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                      self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1]),
                                c="lightgreen", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], self.sp.vmindB),
                                c="lightgreen", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                      self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2]),
                                c="plum", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                    self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], self.sp.vmindB),
                                c="plum", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[3],
                                                      self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3]),
                                c="lightsalmon", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[3],
                                                    self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3], self.sp.vmindB),
                                c="lightsalmon", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ,
                                 self.sp.psdIQdB
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                    self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[3],
                                         self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3], 0),
                                 c="green", lw=1, alpha=0.5, ls="--")
                        # ax2.plot(- self.sp.fIQ,
                        #          self.sp.psdIQdB
                        #          - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                        #                  self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                        #          - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                        #                  self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0)
                        #          - gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                        #                  self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], 0)
                        #          - gauss(self.sp.fIQ, self.sp.asym.peak_fit[3],
                        #                  self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3], 0),
                        #          c="purple", lw=1, alpha=0.5, ls="--")
                        ax2.plot(self.sp.fIQ, (self.sp.psdIQdB + np.flip(self.sp.psdIQdB))/2,
                                 c="purple", lw=1, alpha=0.5, ls="--")

            ax.set_ylim(self.sp.asym.vmindB, self.sp.asym.vmaxdB)
            ax2.set_ylim(self.sp.vmindB, self.sp.vmaxdB)
            ax.set_ylabel("S(f) - S(-f)\n"
                          "[dB]")
            ax2.set_ylabel("S(f) [dB]")
            ax2.set_xlabel("Frequency [Hz]")
            ax2.set_xlim(- self.sp.asym.fmax, self.sp.asym.fmax)

        else:
            figdir = "Retrieve_MWRM_spectrum_asym"
            proc.ifNotMake(figdir)
            fnm = f"{self.sn}_{self.subsn}_{self.sp.tstart}_{self.sp.tend}_{self.diagname}_{self.chI}_{self.chQ}"
            if fmin is not None:
                fnm += f"_min{self.sp.asym.fmin * 1e-3}kHz"
            if fmax is not None:
                fnm += f"_max{self.sp.asym.fmax * 1e-3}kHz"
            path = os.path.join(figdir, f"{fnm}.png")
            title = f"#{self.sn}-{self.subsn} {self.sp.tstart}-{self.sp.tend}s\n" \
                    f"{self.diagname} ch:{self.chI},{self.chQ}"
            fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(5, 8), num=fnm)

            self.sp.asym.vmax = self.sp.asym.psd[_idx].max()
            self.sp.asym.vmin = self.sp.asym.psd[_idx].min()

            ax.plot(self.sp.fIQ, self.sp.asym.psd, ".", c="grey", ms=1)
            ax2.plot(self.sp.fIQ, self.sp.psdIQ, ".", c="grey", ms=1)
            ax.plot(self.sp.fIQ[_idx], self.sp.asym.psd_smooth, c="blue", lw=2, alpha=0.5)
            ax.hlines(0, - self.sp.asym.fmax, self.sp.asym.fmax, colors="grey", ls="--", lw=1)

            if peak_detection:
                ax.vlines(self.sp.asym.peak_freqs, self.sp.asym.vmin, self.sp.asym.vmax,
                          colors="red", alpha=0.5, lw=1, ls="--")
            if gaussfitting:
                if (~isinstance(self.sp.asym.fD_fit, np.ndarray)) and (self.sp.asym.fD_fit is np.nan):
                    print("\n")
                else:
                    ax.vlines(self.sp.asym.fD_fit, self.sp.asym.vmin, self.sp.asym.vmax,
                              colors="pink", lw=1, ls="--")
                    if self.sp.asym.gaussfit_num == 1:
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit,
                                                      self.sp.asym.width_fit, self.sp.asym.fD_fit), c="lightblue", lw=2)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit,
                                                    self.sp.asym.width_fit, self.sp.asym.fD_fit, 0),
                                 c="lightblue", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit,
                                                      self.sp.asym.width_fit, self.sp.asym.fD_fit),
                                c="green", lw=1, alpha=0.5, ls="--")
                        ax2.plot(self.sp.fIQ, self.sp.psdIQ - gauss(self.sp.fIQ, self.sp.asym.peak_fit,
                                                                      self.sp.asym.width_fit, self.sp.asym.fD_fit,
                                                                      0),
                                 c="green", lw=1, alpha=0.5, ls="--")
                        ax2.plot(- self.sp.fIQ, self.sp.psdIQ - gauss(self.sp.fIQ, self.sp.asym.peak_fit,
                                                                        self.sp.asym.width_fit, self.sp.asym.fD_fit,
                                                                        0),
                                 c="purple", lw=1, alpha=0.5, ls="--")
                    elif self.sp.asym.gaussfit_num == 2:
                        ax.plot(self.sp.fIQ, oddgauss2(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                       self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0],
                                                       self.sp.asym.peak_fit[1],
                                                       self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1],
                                                       ),
                                c="green", lw=1, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                      self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0]),
                                c="lightblue", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0),
                                 c="lightblue", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                      self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1]),
                                c="lightgreen", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0),
                                 c="lightgreen", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ,
                                 self.sp.psdIQ
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                         self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                         self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0),
                                 c="green", lw=1, alpha=0.5, ls="--")
                        ax2.plot(- self.sp.fIQ,
                                 self.sp.psdIQ
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                         self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                         self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0),
                                 c="purple", lw=1, alpha=0.5, ls="--")
                    elif self.sp.asym.gaussfit_num == 3:
                        ax.plot(self.sp.fIQ, oddgauss3(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                       self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0],
                                                       self.sp.asym.peak_fit[1],
                                                       self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1],
                                                       self.sp.asym.peak_fit[2],
                                                       self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2],
                                                       ),
                                c="green", lw=1, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                      self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0]),
                                c="lightblue", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0),
                                 c="lightblue", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                      self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1]),
                                c="lightgreen", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0),
                                 c="lightgreen", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                      self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2]),
                                c="plum", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                    self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], 0),
                                 c="plum", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ,
                                 self.sp.psdIQ
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                         self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                         self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                         self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], 0),
                                 c="green", lw=1, alpha=0.5, ls="--")
                        ax2.plot(- self.sp.fIQ,
                                 self.sp.psdIQ
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                         self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                         self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                         self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], 0),
                                 c="purple", lw=1, alpha=0.5, ls="--")
                    elif self.sp.asym.gaussfit_num == 4:
                        ax.plot(self.sp.fIQ, oddgauss4(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                       self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0],
                                                       self.sp.asym.peak_fit[1],
                                                       self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1],
                                                       self.sp.asym.peak_fit[2],
                                                       self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2],
                                                       self.sp.asym.peak_fit[3],
                                                       self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3]
                                                       ),
                                c="green", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                      self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0]),
                                c="lightblue", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                                    self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0),
                                 c="lightblue", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                      self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1]),
                                c="lightgreen", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                                    self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0),
                                 c="lightgreen", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                      self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2]),
                                c="plum", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                                    self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], 0),
                                 c="plum", lw=2, alpha=0.5)
                        ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.asym.peak_fit[3],
                                                      self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3]),
                                c="lightsalmon", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.asym.peak_fit[3],
                                                    self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3], 0),
                                 c="lightsalmon", lw=2, alpha=0.5)
                        ax2.plot(self.sp.fIQ,
                                 self.sp.psdIQ
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                         self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                         self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                         self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[3],
                                         self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3], 0),
                                 c="green", lw=1, alpha=0.5, ls="--")
                        ax2.plot(- self.sp.fIQ,
                                 self.sp.psdIQ
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[0],
                                         self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[1],
                                         self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[2],
                                         self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2], 0)
                                 - gauss(self.sp.fIQ, self.sp.asym.peak_fit[3],
                                         self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3], 0),
                                 c="purple", lw=1, alpha=0.5, ls="--")

            ax.set_ylim(self.sp.asym.vmin, self.sp.asym.vmax)
            ax.set_ylabel("S(f) - S(-f)"
                          "[a.u.]")
            ax2.set_ylabel("S(f) [a.u.]")
            ax2.set_xlabel("Frequency [Hz]")
            ax2.set_xlim(- self.sp.asym.fmax, self.sp.asym.fmax)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    def spectrum_symmetric_component(self, fmin=None, fmax=None,
                                     polyorder=2, iniwidth=100e3,
                                     maxfev=2000,
                                     display=True, pause=0.):

        self.sp.sym = calc.struct()
        self.sp.sym.iniwidth = iniwidth

        if fmin is None:
            self.sp.sym.fmin = self.sp.dF
        else:
            self.sp.sym.fmin = fmin
        if fmax is None:
            self.sp.sym.fmax = self.Fs / 2
        else:
            self.sp.sym.fmax = fmax
        _idx = np.where((np.abs(self.sp.fIQ) > self.sp.sym.fmin)
                        & (np.abs(self.sp.fIQ) < self.sp.sym.fmax))[0]
        self.sp.sym.winlen = int(self.sp.sym.iniwidth / self.sp.dF / 2) * 2 + 1

        self.sp.sym.psddB = (self.sp.psdIQdB + np.flip(self.sp.psdIQdB, axis=-1)) / 2

        self.sp.sym.srcpsddB = self.sp.psdIQdB[_idx]
        self.sp.sym.f = self.sp.fIQ[_idx]

        # if peak_detection:
        #     # peak detection
        self.sp.sym.psddB_smooth = savgol_filter(self.sp.sym.psddB[_idx],
                                                 window_length=self.sp.sym.winlen,
                                                 polyorder=polyorder)
        #     self.sp.sym.peak_idxs = find_peaks(self.sp.sym.psddB_smooth, prominence=prominence)[0]
        #     self.sp.sym.peak_freqs = self.sp.fIQ[self.sp.sym.peak_idxs]
        #     self.sp.sym.peak_psddB = self.sp.sym.psddB_smooth[self.sp.sym.peak_idxs]
        #     self.sp.sym.peak_nums = np.array(self.sp.sym.peak_idxs).size

        self.sp.sym.vmaxdB = self.sp.sym.psddB[_idx].max()
        self.sp.sym.vmindB = self.sp.sym.psddB[_idx].min()

        # if gaussfitting:
        # gauss fitting
        # if self.sp.sym.peak_nums == 0:
        #     self.sp.sym.gaussfit_num = 0
        #     self.sp.sym.peak_c_fit, self.sp.sym.width_c_fit, self.sp.sym.floor_fit \
        #         = [np.nan, np.nan, np.nan]
        #     self.sp.sym.peak_c_fit_err, self.sp.sym.width_c_fit_err, self.sp.sym.floor_fit_err \
        #         = [np.nan, np.nan, np.nan]
        # elif self.sp.sym.peak_nums == 1:
        self.sp.sym.gaussfit_num = 1
        inip = [self.sp.sym.vmaxdB, iniwidth, self.sp.sym.vmindB]
        # try:
        popt, pcov = curve_fit(symgauss, self.sp.fIQ[_idx], self.sp.sym.psddB_smooth,
                               p0=inip, maxfev=maxfev)
        self.sp.sym.cpeak_fit, self.sp.sym.cwidth_fit, self.sp.sym.floor_fit = popt
        self.sp.sym.cpeak_fit_err, self.sp.sym.cwidth_fit_err, self.sp.sym.floor_fit_err \
            = np.sqrt(np.diag(pcov))
        # except RuntimeError as e:
        #     print(f"Optimal parameters not found: {e}")
        #     self.sp.sym.peak_fit, self.sp.sym.width_fit, self.sp.sym.fD_fit \
        #         = [np.nan, np.nan, np.nan]
        #     self.sp.sym.peak_fit_err, self.sp.sym.width_fit_err, self.sp.sym.fD_fit_err \
        #         = [np.nan, np.nan, np.nan]
        # elif self.sp.sym.peak_nums == 3:
        #     self.sp.sym.gaussfit_num = 2
        #     inip = [self.sp.sym.peak_psddB, [iniwidth, iniwidth], self.sp.sym.peak_freqs]
        #     inip = np.array(inip).flatten(order="F").tolist()
        #     try:
        #         popt, pcov = curve_fit(oddgauss2, self.sp.fIQ[_idx], self.sp.sym.psddB[_idx],
        #                                p0=inip, maxfev=maxfev)
        #         self.sp.sym.peak_fit, self.sp.sym.width_fit, self.sp.sym.fD_fit \
        #             = popt.reshape((self.sp.sym.peak_nums, 3)).T
        #         self.sp.sym.peak_fit_err, self.sp.sym.width_fit_err, self.sp.sym.fD_fit_err \
        #             = np.sqrt(np.diag(pcov)).reshape((self.sp.sym.peak_nums, 3)).T
        #     except RuntimeError as e:
        #         print(f"Optimal parameters not found: {e}")
        #         self.sp.sym.peak_fit, self.sp.sym.width_fit, self.sp.sym.fD_fit \
        #             = [np.nan, np.nan, np.nan]
        #         self.sp.sym.peak_fit_err, self.sp.sym.width_fit_err, self.sp.sym.fD_fit_err \
        #             = [np.nan, np.nan, np.nan]
        # elif self.sp.sym.peak_nums == 5:
        #     self.sp.sym.gaussfit_num = 3
        #     inip = [self.sp.sym.peak_psddB, [iniwidth, iniwidth, iniwidth], self.sp.sym.peak_freqs]
        #     inip = np.array(inip).flatten(order="F").tolist()
        #     try:
        #         popt, pcov = curve_fit(oddgauss3, self.sp.fIQ[_idx], self.sp.sym.psddB[_idx],
        #                                p0=inip, maxfev=maxfev)
        #         self.sp.sym.peak_fit, self.sp.sym.width_fit, self.sp.sym.fD_fit \
        #             = popt.reshape((self.sp.sym.peak_nums, 3)).T
        #         self.sp.sym.peak_fit_err, self.sp.sym.width_fit_err, self.sp.sym.fD_fit_err \
        #             = np.sqrt(np.diag(pcov)).reshape((self.sp.sym.peak_nums, 3)).T
        #     except RuntimeError as e:
        #         print(f"Optimal parameters not found: {e}")
        #         self.sp.sym.peak_fit, self.sp.sym.width_fit, self.sp.sym.fD_fit \
        #             = [np.nan, np.nan, np.nan]
        #         self.sp.sym.peak_fit_err, self.sp.sym.width_fit_err, self.sp.sym.fD_fit_err \
        #             = [np.nan, np.nan, np.nan]
        # elif self.sp.sym.peak_nums == 7:
        #     self.sp.sym.gaussfit_num = 4
        #     inip = [self.sp.sym.peak_psddB, [iniwidth, iniwidth, iniwidth, iniwidth], self.sp.sym.peak_freqs]
        #     inip = np.array(inip).flatten(order="F").tolist()
        #     try:
        #         popt, pcov = curve_fit(oddgauss4, self.sp.fIQ[_idx], self.sp.sym.psddB[_idx],
        #                                p0=inip, maxfev=maxfev)
        #         self.sp.sym.peak_fit, self.sp.sym.width_fit, self.sp.sym.fD_fit \
        #             = popt.reshape((self.sp.sym.gaussfit_num, 3)).T
        #         self.sp.sym.peak_fit_err, self.sp.sym.width_fit_err, self.sp.sym.fD_fit_err \
        #             = np.sqrt(np.diag(pcov)).reshape((self.sp.sym.gaussfit_num, 3)).T
        #     except RuntimeError as e:
        #         print(f"Optimal parameters not found: {e}")
        #         self.sp.sym.peak_fit, self.sp.sym.width_fit, self.sp.sym.fD_fit \
        #             = [np.nan, np.nan, np.nan]
        #         self.sp.sym.peak_fit_err, self.sp.sym.width_fit_err, self.sp.sym.fD_fit_err \
        #             = [np.nan, np.nan, np.nan]
        # else:  # until 4.
        #     self.sp.sym.gaussfit_num = 4
        #     idx_str = np.argsort(self.sp.sym.peak_psddB)[- self.sp.sym.gaussfit_num:]
        #     inip = [self.sp.sym.peak_psddB[idx_str],
        #             [iniwidth, iniwidth, iniwidth, iniwidth],
        #             self.sp.sym.peak_freqs[idx_str]]
        #     inip = np.array(inip).flatten(order="F").tolist()
        #     try:
        #         popt, pcov = curve_fit(oddgauss4, self.sp.fIQ[_idx], self.sp.sym.psddB[_idx],
        #                                p0=inip, maxfev=maxfev)
        #         self.sp.sym.peak_fit, self.sp.sym.width_fit, self.sp.sym.fD_fit \
        #             = popt.reshape((self.sp.sym.gaussfit_num, 3)).T
        #         self.sp.sym.peak_fit_err, self.sp.sym.width_fit_err, self.sp.sym.fD_fit_err \
        #             = np.sqrt(np.diag(pcov)).reshape((self.sp.sym.gaussfit_num, 3)).T
        #     except RuntimeError as e:
        #         print(f"Optimal parameters not found: {e}")
        #         self.sp.sym.peak_fit, self.sp.sym.width_fit, self.sp.sym.fD_fit \
        #             = [np.nan, np.nan, np.nan]
        #         self.sp.sym.peak_fit_err, self.sp.sym.width_fit_err, self.sp.sym.fD_fit_err \
        #             = [np.nan, np.nan, np.nan]
        figdir = "Retrieve_MWRM_spectrum_sym"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.sp.tstart}_{self.sp.tend}_{self.diagname}_{self.chI}_{self.chQ}_dB"
        if fmin is not None:
            fnm += f"_min{self.sp.sym.fmin * 1e-3}kHz"
        if fmax is not None:
            fnm += f"_max{self.sp.sym.fmax * 1e-3}kHz"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.sp.tstart}-{self.sp.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"
        fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(5, 5), num=fnm)

        ax.plot(self.sp.fIQ, self.sp.psdIQdB, ".", c="grey", ms=1)
        ax.plot(self.sp.fIQ, self.sp.sym.psddB, ".", c="grey", ms=1)
        ax.plot(self.sp.fIQ[_idx], self.sp.sym.psddB_smooth, c="blue", lw=2, alpha=0.5)

        # if gaussfitting:
        ax.plot(self.sp.fIQ, symgauss(self.sp.fIQ, self.sp.sym.cpeak_fit,
                                      self.sp.sym.cwidth_fit, self.sp.sym.floor_fit), c="pink", lw=2)

        # if peak_detection:
        # ax.vlines(self.sp.sym.peak_freqs, self.sp.sym.vmindB, self.sp.sym.vmaxdB,
        #           colors="red", alpha=0.5, lw=1, ls="--")
        # if gaussfitting:
        #     if (~isinstance(self.sp.sym.fD_fit, np.ndarray)) and (self.sp.sym.fD_fit is np.nan):
        #         print("\n")
        #     else:
        #         ax.vlines(self.sp.sym.fD_fit, self.sp.sym.vmindB, self.sp.sym.vmaxdB,
        #                   colors="pink", lw=1, ls="--")
        #         if self.sp.sym.gaussfit_num == 1:
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit,
        #                                           self.sp.sym.width_fit, self.sp.sym.fD_fit), c="lightblue", lw=2)
        #             ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.sym.peak_fit,
        #                                         self.sp.sym.width_fit, self.sp.sym.fD_fit, self.sp.vmindB),
        #                      c="lightblue", lw=2, alpha=0.5)
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit,
        #                                           self.sp.sym.width_fit, self.sp.sym.fD_fit),
        #                     c="green", lw=1, alpha=0.5, ls="--")
        #             ax2.plot(self.sp.fIQ, self.sp.psdIQdB - gauss(self.sp.fIQ, self.sp.sym.peak_fit,
        #                                                           self.sp.sym.width_fit, self.sp.sym.fD_fit,
        #                                                           0),
        #                      c="green", lw=1, alpha=0.5, ls="--")
        #             ax2.plot(- self.sp.fIQ, self.sp.psdIQdB - gauss(self.sp.fIQ, self.sp.sym.peak_fit,
        #                                                             self.sp.sym.width_fit, self.sp.sym.fD_fit,
        #                                                             0),
        #                      c="purple", lw=1, alpha=0.5, ls="--")
        #         elif self.sp.sym.gaussfit_num == 2:
        #             ax.plot(self.sp.fIQ, oddgauss2(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                                            self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0],
        #                                            self.sp.sym.peak_fit[1],
        #                                            self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1],
        #                                            ),
        #                     c="green", lw=1, alpha=0.5)
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                                           self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0]),
        #                     c="lightblue", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                                         self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0], self.sp.vmindB),
        #                      c="lightblue", lw=2, alpha=0.5)
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                                           self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1]),
        #                     c="lightgreen", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                                         self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1], self.sp.vmindB),
        #                      c="lightgreen", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ,
        #                      self.sp.psdIQdB
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                              self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0], 0)
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                              self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1], 0),
        #                      c="green", lw=1, alpha=0.5, ls="--")
        #             ax2.plot(- self.sp.fIQ,
        #                      self.sp.psdIQdB
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                              self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0], 0)
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                              self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1], 0),
        #                      c="purple", lw=1, alpha=0.5, ls="--")
        #         elif self.sp.sym.gaussfit_num == 3:
        #             ax.plot(self.sp.fIQ, oddgauss3(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                                            self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0],
        #                                            self.sp.sym.peak_fit[1],
        #                                            self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1],
        #                                            self.sp.sym.peak_fit[2],
        #                                            self.sp.sym.width_fit[2], self.sp.sym.fD_fit[2],
        #                                            ),
        #                     c="green", lw=1, alpha=0.5)
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                                           self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0]),
        #                     c="lightblue", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                                         self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0], self.sp.vmindB),
        #                      c="lightblue", lw=2, alpha=0.5)
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                                           self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1]),
        #                     c="lightgreen", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                                         self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1], self.sp.vmindB),
        #                      c="lightgreen", lw=2, alpha=0.5)
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit[2],
        #                                           self.sp.sym.width_fit[2], self.sp.sym.fD_fit[2]),
        #                     c="plum", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.sym.peak_fit[2],
        #                                         self.sp.sym.width_fit[2], self.sp.sym.fD_fit[2], self.sp.vmindB),
        #                      c="plum", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ,
        #                      self.sp.psdIQdB
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                              self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0], 0)
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                              self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1], 0)
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[2],
        #                              self.sp.sym.width_fit[2], self.sp.sym.fD_fit[2], 0),
        #                      c="green", lw=1, alpha=0.5, ls="--")
        #             ax2.plot(- self.sp.fIQ,
        #                      self.sp.psdIQdB
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                              self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0], 0)
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                              self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1], 0)
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[2],
        #                              self.sp.sym.width_fit[2], self.sp.sym.fD_fit[2], 0),
        #                      c="purple", lw=1, alpha=0.5, ls="--")
        #         elif self.sp.sym.gaussfit_num == 4:
        #             ax.plot(self.sp.fIQ, oddgauss4(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                                            self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0],
        #                                            self.sp.sym.peak_fit[1],
        #                                            self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1],
        #                                            self.sp.sym.peak_fit[2],
        #                                            self.sp.sym.width_fit[2], self.sp.sym.fD_fit[2],
        #                                            self.sp.sym.peak_fit[3],
        #                                            self.sp.sym.width_fit[3], self.sp.sym.fD_fit[3]
        #                                            ),
        #                     c="green", lw=2, alpha=0.5)
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                                           self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0]),
        #                     c="lightblue", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                                         self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0], self.sp.vmindB),
        #                      c="lightblue", lw=2, alpha=0.5)
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                                           self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1]),
        #                     c="lightgreen", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                                         self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1], self.sp.vmindB),
        #                      c="lightgreen", lw=2, alpha=0.5)
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit[2],
        #                                           self.sp.sym.width_fit[2], self.sp.sym.fD_fit[2]),
        #                     c="plum", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.sym.peak_fit[2],
        #                                         self.sp.sym.width_fit[2], self.sp.sym.fD_fit[2], self.sp.vmindB),
        #                      c="plum", lw=2, alpha=0.5)
        #             ax.plot(self.sp.fIQ, oddgauss(self.sp.fIQ, self.sp.sym.peak_fit[3],
        #                                           self.sp.sym.width_fit[3], self.sp.sym.fD_fit[3]),
        #                     c="lightsalmon", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.sym.peak_fit[3],
        #                                         self.sp.sym.width_fit[3], self.sp.sym.fD_fit[3], self.sp.vmindB),
        #                      c="lightsalmon", lw=2, alpha=0.5)
        #             ax2.plot(self.sp.fIQ,
        #                      self.sp.psdIQdB
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #                              self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0], 0)
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #                              self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1], 0)
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[2],
        #                              self.sp.sym.width_fit[2], self.sp.sym.fD_fit[2], 0)
        #                      - gauss(self.sp.fIQ, self.sp.sym.peak_fit[3],
        #                              self.sp.sym.width_fit[3], self.sp.sym.fD_fit[3], 0),
        #                      c="green", lw=1, alpha=0.5, ls="--")
        #             # ax2.plot(- self.sp.fIQ,
        #             #          self.sp.psdIQdB
        #             #          - gauss(self.sp.fIQ, self.sp.sym.peak_fit[0],
        #             #                  self.sp.sym.width_fit[0], self.sp.sym.fD_fit[0], 0)
        #             #          - gauss(self.sp.fIQ, self.sp.sym.peak_fit[1],
        #             #                  self.sp.sym.width_fit[1], self.sp.sym.fD_fit[1], 0)
        #             #          - gauss(self.sp.fIQ, self.sp.sym.peak_fit[2],
        #             #                  self.sp.sym.width_fit[2], self.sp.sym.fD_fit[2], 0)
        #             #          - gauss(self.sp.fIQ, self.sp.sym.peak_fit[3],
        #             #                  self.sp.sym.width_fit[3], self.sp.sym.fD_fit[3], 0),
        #             #          c="purple", lw=1, alpha=0.5, ls="--")
        #             ax2.plot(self.sp.fIQ, (self.sp.psdIQdB + np.flip(self.sp.psdIQdB)) / 2,
        #                      c="purple", lw=1, alpha=0.5, ls="--")

        ax.set_ylim(self.sp.sym.vmindB, self.sp.sym.vmaxdB)
        # ax2.set_ylim(self.sp.vmindB, self.sp.vmaxdB)
        ax.set_ylabel("(S(f) + S(-f))/2\n"
                      "[dB]")
        # ax2.set_ylabel("S(f) [dB]")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_xlim(- self.sp.sym.fmax, self.sp.sym.fmax)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    def spectrum_fit_by(self, function=gauss, fmin=None, fmax=None, usedB=True, p0=None, maxfev=2000):

        self.sp.fit = calc.struct()
        self.sp.fit.fmin = fmin
        self.sp.fit.fmax = fmax
        self.sp.fit.function = function

        _idx = np.where((np.abs(self.sp.fIQ) > self.sp.fit.fmin)
                        & (np.abs(self.sp.fIQ) < self.sp.fit.fmax))[0]
        self.sp.fit.f = self.sp.fIQ[_idx]
        if usedB:
            self.sp.fit.psddB = self.sp.psdIQdB[_idx]
            self.sp.fit.popt, self.sp.fit.pcov = curve_fit(function, self.sp.fit.f, self.sp.fit.psddB,
                                                           p0=p0, maxfev=maxfev)
        else:
            self.sp.fit.psd = self.sp.psdIQ[_idx]
            self.sp.fit.popt, self.sp.fit.pcov = curve_fit(function, self.sp.fit.f, self.sp.fit.psd,
                                                           p0=p0, maxfev=maxfev)


    def spectrum_gaussfit(self, fmin=None, fmax=None,
                          polyorder=2, prominence=3, iniwidth_asym=40e3,
                          iniwidth_sym=100e3, maxfev=2000,
                          display=True, pause=0.):

        self.sp.gauss = calc.struct()

        self.spectrum_asymmetric_component(fmin=fmin, fmax=fmax, peak_detection=True, gaussfitting=True,
                                           polyorder=polyorder, prominence=prominence,
                                           iniwidth=iniwidth_asym, maxfev=maxfev, bydB=True,
                                           display=True, pause=0)
        self.spectrum_symmetric_component(fmin=fmin, fmax=fmax, polyorder=polyorder, iniwidth=iniwidth_sym,
                                          maxfev=2000, display=True, pause=0)

        if self.sp.asym.gaussfit_num == 0:
            self.sp.gauss.cpeak, self.sp.gauss.cwidth, self.sp.gauss.floor, \
            self.sp.gauss.peak_fit, self.sp.gauss.width_fit, self.sp.gauss.fD_fit = [np.nan]*6
            self.sp.gauss.cpeak_err, self.sp.gauss.cwidth_err, self.sp.gauss.floor_err, \
            self.sp.gauss.peak_fit_err, self.sp.gauss.width_fit_err, self.sp.gauss.fD_fit_err = [np.nan]*6

        elif self.sp.asym.gaussfit_num == 1:
            inip = [self.sp.sym.cpeak_fit, self.sp.sym.cwidth_fit, self.sp.sym.floor_fit,
                    self.sp.asym.peak_fit, self.sp.asym.width_fit, self.sp.asym.fD_fit]
            popt, pcov = curve_fit(gauss1, self.sp.sym.f, self.sp.sym.srcpsddB, p0=inip, maxfev=maxfev)

            self.sp.gauss.cpeak, self.sp.gauss.cwidth, self.sp.gauss.floor, \
            self.sp.gauss.peak_fit, self.sp.gauss.width_fit, self.sp.gauss.fD_fit = popt
            self.sp.gauss.cpeak_err, self.sp.gauss.cwidth_err, self.sp.gauss.floor_err, \
            self.sp.gauss.peak_fit_err, self.sp.gauss.width_fit_err, self.sp.gauss.fD_fit_err = np.sqrt(np.diag(pcov))

        elif self.sp.asym.gaussfit_num == 2:
            inip = [self.sp.sym.cpeak_fit, self.sp.sym.cwidth_fit, self.sp.sym.floor_fit,
                    self.sp.asym.peak_fit[0], self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0],
                    self.sp.asym.peak_fit[1], self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1]]
            inip = np.array(inip).flatten(order="F").tolist()

            popt, pcov = curve_fit(gauss2, self.sp.sym.f, self.sp.sym.srcpsddB, p0=inip, maxfev=maxfev)
            self.sp.gauss.cpeak, self.sp.gauss.cwidth, self.sp.gauss.floor \
                = popt[:3]
            self.sp.gauss.peak_fit, self.sp.gauss.width_fit, self.sp.gauss.fD_fit \
                = popt[3:].reshape((self.sp.asym.gaussfit_num, 3)).T
            self.sp.gauss.cpeak_err, self.sp.gauss.cwidth_err, self.sp.gauss.floor_err \
                = np.sqrt(np.diag(pcov[:3]))
            self.sp.gauss.peak_fit_err, self.sp.gauss.width_fit_err, self.sp.gauss.fD_fit_err \
                = np.sqrt(np.diag(pcov[3:])).reshape((self.sp.asym.gaussfit_num, 3)).T

        elif self.sp.asym.gaussfit_num == 3:
            inip = [self.sp.sym.cpeak_fit, self.sp.sym.cwidth_fit, self.sp.sym.floor_fit,
                    self.sp.asym.peak_fit[0], self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0],
                    self.sp.asym.peak_fit[1], self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1],
                    self.sp.asym.peak_fit[2], self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2]]
            inip = np.array(inip).flatten(order="F").tolist()

            popt, pcov = curve_fit(gauss3, self.sp.sym.f, self.sp.sym.srcpsddB, p0=inip, maxfev=maxfev)
            self.sp.gauss.cpeak, self.sp.gauss.cwidth, self.sp.gauss.floor \
                = popt[:3]
            self.sp.gauss.peak_fit, self.sp.gauss.width_fit, self.sp.gauss.fD_fit \
                = popt[3:].reshape((self.sp.asym.gaussfit_num, 3)).T
            self.sp.gauss.cpeak_err, self.sp.gauss.cwidth_err, self.sp.gauss.floor_err \
                = np.sqrt(np.diag(pcov[:3]))
            self.sp.gauss.peak_fit_err, self.sp.gauss.width_fit_err, self.sp.gauss.fD_fit_err \
                = np.sqrt(np.diag(pcov[3:])).reshape((self.sp.asym.gaussfit_num, 3)).T

        elif self.sp.asym.gaussfit_num == 4:
            inip = [self.sp.sym.cpeak_fit, self.sp.sym.cwidth_fit, self.sp.sym.floor_fit,
                    self.sp.asym.peak_fit[0], self.sp.asym.width_fit[0], self.sp.asym.fD_fit[0],
                    self.sp.asym.peak_fit[1], self.sp.asym.width_fit[1], self.sp.asym.fD_fit[1],
                    self.sp.asym.peak_fit[2], self.sp.asym.width_fit[2], self.sp.asym.fD_fit[2],
                    self.sp.asym.peak_fit[3], self.sp.asym.width_fit[3], self.sp.asym.fD_fit[3]]
            inip = np.array(inip).flatten(order="F").tolist()

            popt, pcov = curve_fit(gauss4, self.sp.sym.f, self.sp.sym.srcpsddB, p0=inip, maxfev=maxfev)
            self.sp.gauss.cpeak, self.sp.gauss.cwidth, self.sp.gauss.floor \
                = popt[:3]
            self.sp.gauss.peak_fit, self.sp.gauss.width_fit, self.sp.gauss.fD_fit \
                = popt[3:].reshape((self.sp.asym.gaussfit_num, 3)).T
            self.sp.gauss.cpeak_err, self.sp.gauss.cwidth_err, self.sp.gauss.floor_err \
                = np.sqrt(np.diag(pcov[:3]))
            self.sp.gauss.peak_fit_err, self.sp.gauss.width_fit_err, self.sp.gauss.fD_fit_err \
                = np.sqrt(np.diag(pcov[3:])).reshape((self.sp.asym.gaussfit_num, 3)).T

        else:
            exit()

        figdir = "Retrieve_MWRM_spectrum_gaussfit"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.sp.tstart}_{self.sp.tend}_{self.diagname}_{self.chI}_{self.chQ}"
        if fmin is not None:
            fnm += f"_min{self.sp.asym.fmin * 1e-3}kHz"
        if fmax is not None:
            fnm += f"_max{self.sp.asym.fmax * 1e-3}kHz"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.sp.tstart}-{self.sp.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"
        fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(5, 5), num=fnm)

        ax.plot(self.sp.fIQ, self.sp.psdIQdB, ".", c="grey", ms=1)
        ax.plot(self.sp.fIQ, symgauss(self.sp.fIQ, self.sp.gauss.cpeak,
                                      self.sp.gauss.cwidth, self.sp.gauss.floor), c="black", alpha=0.5, lw=2)

        if self.sp.asym.gaussfit_num == 1:
            ax.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.gauss.peak_fit,
                                       self.sp.gauss.width_fit, self.sp.gauss.fD_fit, self.sp.gauss.floor),
                     c="lightblue", lw=2, alpha=0.5)
            ax.plot(self.sp.fIQ, gauss1(self.sp.fIQ, self.sp.gauss.cpeak, self.sp.gauss.cwidth, self.sp.gauss.floor,
                                        self.sp.gauss.peak_fit, self.sp.gauss.width_fit, self.sp.gauss.fD_fit),
                     c="green", lw=2, alpha=0.5, ls="--")
        elif self.sp.asym.gaussfit_num == 2:
            ax.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.gauss.peak_fit[0],
                                       self.sp.gauss.width_fit[0], self.sp.gauss.fD_fit[0], self.sp.gauss.floor),
                     c="lightblue", lw=2, alpha=0.5)
            ax.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.gauss.peak_fit[1],
                                       self.sp.gauss.width_fit[1], self.sp.gauss.fD_fit[1], self.sp.gauss.floor),
                     c="lightgreen", lw=2, alpha=0.5)
            ax.plot(self.sp.fIQ, gauss2(self.sp.fIQ, self.sp.gauss.cpeak, self.sp.gauss.cwidth, self.sp.gauss.floor,
                                        self.sp.gauss.peak_fit[0], self.sp.gauss.width_fit[0], self.sp.gauss.fD_fit[0],
                                        self.sp.gauss.peak_fit[1], self.sp.gauss.width_fit[1], self.sp.gauss.fD_fit[1]),
                     c="green", lw=2, alpha=0.5, ls="--")
        elif self.sp.asym.gaussfit_num == 3:
            ax.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.gauss.peak_fit[0],
                                       self.sp.gauss.width_fit[0], self.sp.gauss.fD_fit[0], self.sp.gauss.floor),
                    c="lightblue", lw=2, alpha=0.5)
            ax.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.gauss.peak_fit[1],
                                       self.sp.gauss.width_fit[1], self.sp.gauss.fD_fit[1], self.sp.gauss.floor),
                    c="lightgreen", lw=2, alpha=0.5)
            ax.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.gauss.peak_fit[2],
                                       self.sp.gauss.width_fit[2], self.sp.gauss.fD_fit[2], self.sp.gauss.floor),
                    c="plum", lw=2, alpha=0.5)
            ax.plot(self.sp.fIQ, gauss3(self.sp.fIQ, self.sp.gauss.cpeak, self.sp.gauss.cwidth, self.sp.gauss.floor,
                                        self.sp.gauss.peak_fit[0], self.sp.gauss.width_fit[0], self.sp.gauss.fD_fit[0],
                                        self.sp.gauss.peak_fit[1], self.sp.gauss.width_fit[1], self.sp.gauss.fD_fit[1],
                                        self.sp.gauss.peak_fit[2], self.sp.gauss.width_fit[2], self.sp.gauss.fD_fit[2]),
                    c="green", lw=2, alpha=0.5, ls="--")
        elif self.sp.asym.gaussfit_num == 4:
            ax.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.gauss.peak_fit[0],
                                       self.sp.gauss.width_fit[0], self.sp.gauss.fD_fit[0], self.sp.gauss.floor),
                    c="lightblue", lw=2, alpha=0.5)
            ax.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.gauss.peak_fit[1],
                                       self.sp.gauss.width_fit[1], self.sp.gauss.fD_fit[1], self.sp.gauss.floor),
                    c="lightgreen", lw=2, alpha=0.5)
            ax.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.gauss.peak_fit[2],
                                       self.sp.gauss.width_fit[2], self.sp.gauss.fD_fit[2], self.sp.gauss.floor),
                    c="plum", lw=2, alpha=0.5)
            ax.plot(self.sp.fIQ, gauss(self.sp.fIQ, self.sp.gauss.peak_fit[3],
                                       self.sp.gauss.width_fit[3], self.sp.gauss.fD_fit[3], self.sp.gauss.floor),
                    c="lightsalmon", lw=2, alpha=0.5)
            ax.plot(self.sp.fIQ, gauss4(self.sp.fIQ, self.sp.gauss.cpeak, self.sp.gauss.cwidth, self.sp.gauss.floor,
                                        self.sp.gauss.peak_fit[0], self.sp.gauss.width_fit[0], self.sp.gauss.fD_fit[0],
                                        self.sp.gauss.peak_fit[1], self.sp.gauss.width_fit[1], self.sp.gauss.fD_fit[1],
                                        self.sp.gauss.peak_fit[2], self.sp.gauss.width_fit[2], self.sp.gauss.fD_fit[2],
                                        self.sp.gauss.peak_fit[3], self.sp.gauss.width_fit[3], self.sp.gauss.fD_fit[3]),
                    c="green", lw=2, alpha=0.5, ls="--")

        ax.set_ylim(self.sp.sym.vmindB, self.sp.sym.vmaxdB)
        ax.set_ylabel("S(f) [dB]")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_xlim(- self.sp.sym.fmax, self.sp.sym.fmax)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    # def spectrum_cauchyfit(self):

    def intensity(self, fmin=150e3, fmax=490e3, bgon=True, sub_phi=False):

        self.spg.int = calc.struct()
        self.spg.int.subphi = sub_phi
        self.spg.int.fmin = fmin
        self.spg.int.fmax = fmax
        idx_use = np.where((np.abs(self.spg.f) >= self.spg.int.fmin) & (np.abs(self.spg.f) <= self.spg.int.fmax))[0]
        if sub_phi:
            spec_Sk = self.spg.psd_subphi[:, idx_use]
        else:
            spec_Sk = self.spg.psd[:, idx_use]

        self.spg.int.Sk = np.sum(spec_Sk, axis=-1) * self.spg.dF
        self.spg.int.Ia = np.sqrt(self.spg.int.Sk)

        self.spg.int.Sknorm = self.spg.int.Sk / np.max(self.spg.int.Sk)
        self.spg.int.Ianorm = self.spg.int.Ia / np.max(self.spg.int.Ia)

        if bgon:
            idx_use = np.where((np.abs(self.bg.f) >= self.spg.int.fmin) & (np.abs(self.bg.f) <= self.spg.int.fmax))[0]
            spec_Sk = self.bg.psd[:, idx_use]

            self.spg.int.Sk_bg = np.sum(spec_Sk, axis=-1) * self.bg.dF
            self.spg.int.Ia_bg = np.sqrt(self.spg.int.Sk_bg)

            self.spg.int.Sknorm_bg = self.spg.int.Sk_bg / self.spg.int.Sk.max()
            self.spg.int.Ianorm_bg = self.spg.int.Ia_bg / self.spg.int.Ia.max()

            for et in self.bg.ets:
                ts, te = et
                tssp = self.spg.t - self.spg.dT / 2
                tesp = self.spg.t + self.spg.dT / 2
                idx_nan = np.where((ts < tesp) & (te > tssp))[0]
                self.spg.int.Sk[idx_nan] = np.nan
                self.spg.int.Ia[idx_nan] = np.nan
                self.spg.int.Sknorm[idx_nan] = np.nan
                self.spg.int.Ianorm[idx_nan] = np.nan

        return self.spg.int

    def ref_to_tsmap(self, Rat=4.38, rho_cut=1.0, include_grad=True, use_nefit=True, skipnan=False, bgon=False):

        self.tsmap = get_eg.tsmap(self.sn, self.subsn, self.tstart, self.tend, rho_cut=rho_cut)
        if include_grad:
            self.tsmap.calcgrad()
        self.tsmap.R_window(Rat=Rat, include_grad=include_grad)
        datlist = [self.spg.int.Sk, self.spg.int.Ia, self.spg.int.Sknorm, self.spg.int.Ianorm]
        avglist, stdlist, errlist = calc.timeAverageDatListByRefs(self.spg.t, datlist, self.tsmap.t, skipnan=skipnan)
        self.tsmap.Sk, self.tsmap.Ia, self.tsmap.Sknorm, self.tsmap.Ianorm = avglist
        self.tsmap.Sk_err, self.tsmap.Ia_err, self.tsmap.Sknorm_err, self.tsmap.Ianorm_err = stdlist

        if use_nefit:
            self.tsmap.Sknesq = self.tsmap.Sk / ((self.tsmap.Rwin.avg.ne_fit)**2)
            self.tsmap.Sknesq_err = self.tsmap.Sknesq \
                                    * np.sqrt((self.tsmap.Sk_err/self.tsmap.Sk)**2
                                              + (2 * self.tsmap.Rwin.std.ne_fit / self.tsmap.Rwin.avg.ne_fit)**2)
            self.tsmap.Iane = self.tsmap.Ia / self.tsmap.Rwin.avg.ne_fit
            self.tsmap.Iane_err = self.tsmap.Iane \
                                    * np.sqrt((self.tsmap.Ia_err / self.tsmap.Ia) ** 2
                                              + (self.tsmap.Rwin.std.ne_fit / self.tsmap.Rwin.avg.ne_fit) ** 2)

            if bgon:
                self.tsmap.ne_bg = interp1d(self.tsmap.t, self.tsmap.Rwin.avg.ne_fit)(self.bg.t)
                self.tsmap.ne_bg_err = interp1d(self.tsmap.t, self.tsmap.Rwin.std.ne_fit)(self.bg.t)

                self.tsmap.Sknesq_bg = self.spg.int.Sk_bg / ((self.tsmap.ne_bg) ** 2)
                self.tsmap.Sknesq_bg_err = self.tsmap.Sknesq_bg \
                                        * np.sqrt((2 * self.tsmap.ne_bg_err / self.tsmap.ne_bg) ** 2)
                self.tsmap.Iane_bg = self.spg.int.Ia_bg / self.tsmap.ne_bg
                self.tsmap.Iane_bg_err = self.tsmap.Iane_bg \
                                      * np.sqrt((self.tsmap.ne_bg_err / self.tsmap.ne_bg) ** 2)

        else:
            self.tsmap.Sknesq = self.tsmap.Sk / ((self.tsmap.Rwin.avg.ne_polyfit)**2)
            self.tsmap.Sknesq_err = self.tsmap.Sknesq \
                                    * np.sqrt((self.tsmap.Sk_err/self.tsmap.Sk)**2
                                              + (2 * self.tsmap.Rwin.std.ne_polyfit / self.tsmap.Rwin.avg.ne_polyfit)**2)
            self.tsmap.Iane = self.tsmap.Ia / self.tsmap.Rwin.avg.ne_polyfit
            self.tsmap.Iane_err = self.tsmap.Iane \
                                    * np.sqrt((self.tsmap.Ia_err / self.tsmap.Ia) ** 2
                                              + (self.tsmap.Rwin.std.ne_polyfit / self.tsmap.Rwin.avg.ne_polyfit) ** 2)

            if bgon:
                self.tsmap.ne_bg = interp1d(self.tsmap.t, self.tsmap.Rwin.avg.ne_polyfit)(self.bg.t)
                self.tsmap.ne_bg_err = interp1d(self.tsmap.t, self.tsmap.Rwin.std.ne_polyfit)(self.bg.t)

                self.tsmap.Sknesq_bg = self.spg.int.Sk_bg / ((self.tsmap.ne_bg) ** 2)
                self.tsmap.Sknesq_bg_err = self.tsmap.Sknesq_bg \
                                        * np.sqrt((2 * self.tsmap.ne_bg_err / self.tsmap.ne_bg) ** 2)
                self.tsmap.Iane_bg = self.spg.int.Ia_bg / self.tsmap.ne_bg
                self.tsmap.Iane_bg_err = self.tsmap.Iane_bg \
                                      * np.sqrt((self.tsmap.ne_bg_err / self.tsmap.ne_bg) ** 2)

        return self.tsmap

    def ref_to_fir_nel(self, Rfir=4.1, bgon=False):
        Rfirs = np.array([3.309, 3.399, 3.489, 3.579,
                          3.669, 3.759, 3.849, 3.939,
                          4.029, 4.119, 4.209, 4.299, 4.389])

        self.fir = get_eg.fir_nel(self.sn, self.subsn, self.tstart, self.tend)
        self.fir.ref_to(self.spg.t)
        idx_dat = np.argmin(np.abs(Rfirs - Rfir))
        datlist = [self.fir.ref.avg.nl3309, self.fir.ref.avg.nl3399, self.fir.ref.avg.nl3489, self.fir.ref.avg.nl3579,
                   self.fir.ref.avg.nl3669, self.fir.ref.avg.nl3759, self.fir.ref.avg.nl3849, self.fir.ref.avg.nl3939,
                   self.fir.ref.avg.nl4029, self.fir.ref.avg.nl4119, self.fir.ref.avg.nl4209, self.fir.ref.avg.nl4299,
                   self.fir.ref.avg.nl4389]
        self.spg.nel = datlist[idx_dat]
        self.spg.int.Ianel = self.spg.int.Ia / self.spg.nel
        self.spg.int.Sknelsq = self.spg.int.Sk / (self.spg.nel**2)

        if bgon:
            self.spg.nel_bg = interp1d(self.spg.t, self.spg.nel, bounds_error=False, fill_value="extrapolate")(self.bg.t)

            self.spg.int.Sknelsq_bg = self.spg.int.Sk_bg / ((self.spg.nel_bg) ** 2)
            self.spg.int.Ianel_bg = self.spg.int.Ia_bg / self.spg.nel_bg

    def dopplershift(self, fmin=3e3, fmax=1250e3):

        self.spg.DS = calc.struct()
        self.spg.DS.fmin = fmin
        self.spg.DS.fmax = fmax
        idx_use = np.where((np.abs(self.spg.f) >= self.spg.DS.fmin) & (np.abs(self.spg.f) <= self.spg.DS.fmax))[0]
        freq_DS = self.spg.f[idx_use]
        spec_DS = self.spg.psd[:, idx_use]

    def phasespec(self, NFFT=2**10, ovr=0.5, window="hann", dT=2e-3,
                  cmap="viridis", pause=0., display=True):

        self.phase = calc.struct()
        self.phase.spg = calc.struct()
        self.phase.spg.NFFT = NFFT
        self.phase.spg.ovr = ovr
        self.phase.spg.NOV = int(self.phase.spg.NFFT * self.phase.spg.ovr)
        self.phase.spg.window = window
        self.phase.spg.dT = dT
        self.phase.spg.Fs = 1. / self.phase.spg.dT
        self.phase.spg.NSamp = int(self.Fs / self.phase.spg.Fs)
        self.phase.spg.Nsp = self.size // self.phase.spg.NSamp
        self.phase.spg.cmap = cmap

        self.phase.spg.t = self.t[:self.phase.spg.Nsp * self.phase.spg.NSamp].reshape((self.phase.spg.Nsp, self.phase.spg.NSamp)).mean(axis=-1)
        darray = self.phaseIQ[:self.phase.spg.Nsp * self.phase.spg.NSamp].reshape((self.phase.spg.Nsp, self.phase.spg.NSamp))
        self.phase.spg.f, self.phase.spg.psd = welch(x=darray, fs=self.Fs, window="hann",
                                                     nperseg=self.phase.spg.NFFT, noverlap=self.phase.spg.NOV,
                                                     detrend="linear", scaling="density",
                                                     axis=-1, average="mean")
        self.phase.spg.dF = self.dT * self.phase.spg.NFFT

        self.phase.spg.fmin = self.phase.spg.dF
        self.phase.spg.fmax = self.Fs / 2

        figdir = "Retrieve_MWRM\\phase"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI} {self.chQ}"
        fig, axs = plt.subplots(nrows=2, sharex=True,
                                figsize=(5, 8), gridspec_kw={'height_ratios': [1, 3]},
                                num=fnm)

        axs[0].plot(self.t, self.phaseIQ, c="black", lw=0.1)
        axs[0].set_ylabel("[rad]")
        p2p = self.phaseIQ.max() - self.phaseIQ.min()
        axs[0].set_ylim(self.phaseIQ.min() - p2p * 0.05, self.phaseIQ.max() + p2p * 0.05)

        axs[1].pcolorfast(np.append(self.phase.spg.t - 0.5 * self.phase.spg.dT, self.phase.spg.t[-1] + 0.5 * self.phase.spg.dT),
                          np.append(self.phase.spg.f - 0.5 * self.phase.spg.dF, self.phase.spg.f[-1] + 0.5 * self.phase.spg.dF),
                          self.phase.spg.psd.T, cmap=self.phase.spg.cmap)
        axs[1].set_ylabel("Frequency [Hz]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_yscale("log")
        axs[1].set_ylim(self.phase.spg.fmin, self.phase.spg.fmax)
        axs[1].set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    def BSmod(self, threshold=-0.1, diagmod="MWRM-PXI", chmod=4):

        self.mod = single(sn=self.sn, subsn=self.subsn, tstart=self.tstart,
                          tend=self.tend, diagname=diagmod, ch=chmod)
        self.mod.threshold = threshold

        idxs_modDatoverthresh = np.where(self.mod.d > self.mod.threshold)[0]
        idxDiffs_modDatoverthresh = np.diff(idxs_modDatoverthresh)
        idxs_DiffIsNotEqualTo1 = np.where(idxDiffs_modDatoverthresh > 100)[0]
        idxs_ets_modDatoverthresh = np.concatenate((idxs_DiffIsNotEqualTo1, idxs_DiffIsNotEqualTo1 + 1), axis=0)
        idxs_ets_modDatoverthresh = np.sort(idxs_ets_modDatoverthresh)
        idxs_ets = idxs_modDatoverthresh[idxs_ets_modDatoverthresh]
        idxs_ets = np.insert(idxs_ets, 0, idxs_modDatoverthresh[0])
        idxs_ets = np.append(idxs_ets, idxs_modDatoverthresh[-1])

        self.mod.ets = self.mod.t[idxs_ets]
        self.mod.ets = np.reshape(self.mod.ets, (len(idxs_DiffIsNotEqualTo1) + 1, 2))

    def BSBackground(self, NFFT=2**10, OVR=0.5):

        self.bg = calc.struct()
        self.bg.NFFT = NFFT
        self.bg.OVR = OVR
        self.bg.NOV = int(self.bg.NFFT * self.bg.OVR)
        self.bg.dF = self.Fs / self.bg.NFFT

        psd_list = []
        for offTs, offTe in self.mod.ets:
            _idxs, datList = proc.getTimeIdxsAndDats(self.t, offTs, offTe, [self.IQ])
            offIQ = datList[0]

            f, psd = welch(x=offIQ, fs=self.Fs, window="hann",
                           nperseg=self.bg.NFFT, noverlap=self.bg.NOV,
                           detrend="constant", scaling="density",
                           axis=-1, average="mean", return_onesided=False)
            self.bg.f = fft.fftshift(f)
            psd = fft.fftshift(psd)
            psd_list.append(psd)

        self.bg.ets = self.mod.ets
        self.bg.t = self.mod.ets.mean(axis=-1)
        self.bg.psd = np.array(psd_list)

    def plot_intensity(self, bgon=False, pause=0.):

        plot.set("paper", "ticks")

        self.ech = get_eg.ech_v2(sn=self.sn, tstart=self.tstart, tend=self.tend)
        self.nb = get_eg.nb_alldev(sn=self.sn, tstart=self.tstart, tend=self.tend)
        self.nel = get_eg.nel(sn=self.sn, sub=self.subsn, tstart=self.tstart, tend=self.tend)
        self.wp = get_eg.wp(sn=self.sn, sub=self.subsn, tstart=self.tstart, tend=self.tend)

        figdir = os.path.join(self.dirbase, "intensity")
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.diagname} {self.chI} {self.chQ}"

        fig, ax = plt.subplots(9, sharex=True, num=title, figsize=(6, 8))
        fig.suptitle(title)
        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        if bgon:
            self.BSmod()
            self.BSBackground(NFFT=self.spg.NFFT, OVR=self.spg.ovr)
        self.intensity(fmin=3e3, fmax=30e3, bgon=bgon)
        ax[4].plot(self.spg.t, self.spg.int.Ia, c="blue",
                   label=f"{self.spg.int.fmin*1e-3}-{self.spg.int.fmax*1e-3} kHz")
        if bgon:
            ax[4].plot(self.bg.t, self.spg.int.Ia_bg, ".", c="grey", label="bg")
        ax[4].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        self.intensity(fmin=30e3, fmax=150e3, bgon=bgon)
        ax[5].plot(self.spg.t, self.spg.int.Ia, c="green",
                   label=f"{self.spg.int.fmin * 1e-3}-{self.spg.int.fmax * 1e-3} kHz")
        if bgon:
            ax[5].plot(self.bg.t, self.spg.int.Ia_bg, ".", c="grey", label="bg")
        ax[5].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        self.intensity(fmin=100e3, fmax=490e3, bgon=bgon)
        ax[6].plot(self.spg.t, self.spg.int.Ia, c="red",
                   label=f"{self.spg.int.fmin * 1e-3}-{self.spg.int.fmax * 1e-3} kHz")
        if bgon:
            ax[6].plot(self.bg.t, self.spg.int.Ia_bg, ".", c="grey", label="bg")
        ax[6].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        self.intensity(fmin=20e3, fmax=200e3, bgon=bgon)
        ax[7].plot(self.spg.t, self.spg.int.Ia, c="green",
                   label=f"{self.spg.int.fmin * 1e-3}-{self.spg.int.fmax * 1e-3} kHz")
        if bgon:
            ax[7].plot(self.bg.t, self.spg.int.Ia_bg, ".", c="grey", label="bg")
        ax[7].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        self.intensity(fmin=200e3, fmax=500e3, bgon=bgon)
        ax[8].plot(self.spg.t, self.spg.int.Ia, c="red",
                   label=f"{self.spg.int.fmin * 1e-3}-{self.spg.int.fmax * 1e-3} kHz")
        if bgon:
            ax[8].plot(self.bg.t, self.spg.int.Ia_bg, ".", c="grey", label="bg")
        ax[8].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        ax[8].set_xlabel("Time [s]")
        ax[8].set_xlim(self.tstart, self.tend)

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")
        ax[4].set_ylabel("Amplitude\n"
                         "[a.u.]")
        ax[5].set_ylabel("Amplitude\n"
                         "[a.u.]")
        ax[6].set_ylabel("Amplitude\n"
                         "[a.u.]")
        ax[7].set_ylabel("Amplitude\n"
                         "[a.u.]")
        ax[8].set_ylabel("Amplitude\n"
                         "[a.u.]")

        plot.caption(fig, title)
        plot.capsave(fig, title, fnm, path)
        plot.check(pause)

    def read_param(self):

        self.ech = get_eg.ech_v2(sn=self.sn, tstart=self.tstart, tend=self.tend)
        self.nb = get_eg.nb_alldev(sn=self.sn, tstart=self.tstart, tend=self.tend)

        self.gp = get_eg.gas_puf(sn=self.sn, sub=self.subsn, tstart=self.tstart, tend=self.tend)

    def bispectrum(self, tstart=4, tend=5, NFFT=2**8, OVR=0.5,
                   window="hann", mode="IQ",
                   fmax1=None, fmax2=None, display=True, pause=0.):

        if mode == "IQ":
            self.spectrum(tstart, tend, NFFT, OVR, window, display=False, bgon=False)
            self.cbsp = calc.cross_bispectral_analysis(self.sp.IQraw, self.sp.IQraw, self.sp.IQraw,
                                                       self.dT, self.dT, self.dT,
                                                       NFFT, NFFT, NFFT,
                                                       flimx = fmax1, flimy = fmax2,
                                                       OVR=OVR, window=window)
            self.cbsp.freqz, self.cbsp.biCohSq_fz, self.cbsp.biCohSqErr_fz \
                = calc.average_bicoherence_at_f3_withErr(self.cbsp.freqx, self.cbsp.freqy,
                                                         self.cbsp.biCohSq, self.cbsp.biCohSqErr)
        else:
            print("未実装!")
            exit()

        plot.set("notebook", "ticks")

        if not display:
            original_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        if fmax2 is not None:
            fname += f"_{fmax2}"
        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT} {OVR} {window}"

        figdir = os.path.join(self.dirbase, "bicoherence")
        proc.ifNotMake(figdir)
        path = os.path.join(figdir, f"{fname}.png")

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biCohSq, cmap="plasma",
                           vmin=0, vmax=5./self.cbsp.NEns)
        cbar = plt.colorbar(im)
        cbar.set_label("bicoherence^2")

        ax.set_ylabel("Frequency 1 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)


        # figdir = os.path.join(self.dirbase, "bicoherence_err")
        # proc.ifNotMake(figdir)
        # path = os.path.join(figdir, f"{fname}.png")
        #
        # fig, ax = plt.subplots(1)
        # im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biCohSqErr, cmap="Greys",
        #                    vmin=0, vmax=5./self.cbsp.NEns)
        # cbar = plt.colorbar(im)
        # cbar.set_label("bicoherence^2 err")
        #
        # ax.set_ylabel("Frequency 1 [Hz]")
        # ax.set_xlabel("Frequency 1 [Hz]")
        # ax.set_title(axtitle)
        # if fmax1 is not None:
        #     ax.set_xlim(-fmax1, fmax1)
        # if fmax2 is not None:
        #     ax.set_ylim(-fmax2, fmax2)
        #
        # plot.caption(fig, self.figtitle)
        # plot.capsave(fig, self.figtitle, fname, path)
        # if display:
        #     plot.check(pause)
        # else:
        #     plot.close(fig)



        figdir = os.path.join(self.dirbase, "bicoherence_rer")
        proc.ifNotMake(figdir)
        path = os.path.join(figdir, f"{fname}.png")

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biCohSqRer, cmap="Greys", vmax=1, vmin=0)
        cbar = plt.colorbar(im)
        cbar.set_label("bicoherence^2 relative err")

        ax.set_ylabel("Frequency 1 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)


        figdir = os.path.join(self.dirbase, "phase")
        proc.ifNotMake(figdir)
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biPhs, cmap="plasma")
        cbar = plt.colorbar(im)
        cbar.set_label("phase [rad]")

        ax.set_ylabel("Frequency 1 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)

        figdir = os.path.join(self.dirbase, "phase_err")
        proc.ifNotMake(figdir)
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biPhsErr, cmap="Greys",
                           vmin=0, vmax=np.pi)
        cbar = plt.colorbar(im)
        cbar.set_label("phase err [rad]")

        ax.set_ylabel("Frequency 1 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)


        figdir = os.path.join(self.dirbase, "bicoherence_fz")
        proc.ifNotMake(figdir)
        path = os.path.join(figdir, f"{fname}.png")

        fig, ax = plt.subplots(1)
        l = ax.errorbar(self.cbsp.freqz, self.cbsp.biCohSq_fz, self.cbsp.biCohSqErr_fz,
                        fmt=".-", ecolor="grey")

        ax.set_ylabel("bicoherence^2 average")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        ax.set_ylim(0, 3. / np.sqrt(self.cbsp.NEns))

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)


        if not display:
            matplotlib.use(original_backend)

        return self.cbsp


class twinIQ:

    def __init__(self, sn=192781, sub=1, tstart=3, tend=6,
                 diag1="MWRM-COMB2", chI1=19, chQ1=20,
                 diag2="MWRM-COMB", chI2=1, chQ2=2):

        self.sn = sn
        self.sub = sub
        self.ts = tstart
        self.te = tend
        self.diag1 = diag1
        self.chI1 = chI1
        self.chQ1 = chQ1
        self.diag2 = diag2
        self.chI2 = chI2
        self.chQ2 = chQ2

        self.iq1 = IQ(sn, sub, tstart, tend, diag1, chI1, chQ1)
        self.iq2 = IQ(sn, sub, tstart, tend, diag2, chI2, chQ2)

        self.dirbase = "Retrieve_MWRM_twinIQ"
        proc.ifNotMake(self.dirbase)
        self.fnm_init = f"{sn}_{sub}_{tstart}_{tend}_{diag1}_{chI1}_{chQ1}_{diag2}_{chI2}_{chQ2}"
        self.figtitle = f"#{self.sn}-{self.sub}\n" \
                        f"1: {self.diag1} {self.chI1} {self.chQ1}\n" \
                        f"2: {self.diag2} {self.chI2} {self.chQ2}"


    def cross_spectrogram(self, NFFT=2**10, window="hann", NEns=20, OVR=0.5, mode="ampIQ",
                          fmin=None, fmax=None,
                          freqlog=True, pause=0.):
        # mode: IQ, ampIQ, phaseIQ, ampIQ_vs_phaseIQ, cog

        plot.set("notebook", "ticks")

        if mode == "ampIQ":
            self.csg = calc.cross_spectrogram_2s(self.iq1.t, self.iq1.ampIQ, self.iq2.ampIQ,
                                                 NFFT=NFFT, window=window, NEns=NEns, OVR=OVR)
        elif mode == "IQ":
            self.csg = calc.cross_spectrogram_2s(self.iq1.t, self.iq1.IQ, self.iq2.IQ,
                                                 NFFT=NFFT, window=window, NEns=NEns, OVR=OVR)
        elif mode == "phaseIQ":
            self.csg = calc.cross_spectrogram_2s(self.iq1.t, self.iq1.phaseIQ, self.iq2.phaseIQ,
                                                 NFFT=NFFT, window=window, NEns=NEns, OVR=OVR)
        else:
            print("未実装!!")
            exit()

        self.csg.mode = mode
        self.csg.fmin = fmin
        self.csg.fmax = fmax

        figdir_base = os.path.join(self.dirbase, "cross_spectrogram")
        proc.ifNotMake(figdir_base)
        figdir = os.path.join(figdir_base, mode)
        proc.ifNotMake(figdir)

        fnm = f"{self.sn}_{self.sub}_{self.ts}_{self.te}_" \
              f"{self.diag1}_{self.chI1}_{self.chQ1}_" \
              f"{self.diag2}_{self.chI2}_{self.chQ2}_" \
              f"{self.csg.mode}_{self.csg.NFFT}_{self.csg.window}_{self.csg.NEns}_{self.csg.fmin}_{self.csg.fmax}"
        path = os.path.join(figdir, f"{fnm}.png")

        figtitle = f"#{self.sn}-{self.sub}\n" \
                   f"{self.diag1} {self.chI1} {self.chQ1}\n" \
                   f"{self.diag2} {self.chI2} {self.chQ2}"
        axtitle = f"{self.csg.mode} {self.csg.NFFT} {self.csg.window} {self.csg.NEns}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.csg.tsp,
                           self.csg.freq, self.csg.coh_sq.T,
                           cmap="plasma",
                           vmin=0, vmax=4./np.sqrt(NEns))
        cbar = plt.colorbar(im)
        cbar.set_label("Coherence^2")

        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        ax.set_title(axtitle)
        if freqlog:
            ax.set_yscale("log")
        if fmin is not None:
            ax.set_ylim(bottom=fmin)
        if fmax is not None:
            ax.set_ylim(top=fmax)

        plot.caption(fig, figtitle)
        plot.capsave(fig, figtitle, fnm, path)
        plot.check(pause)

        return self.csg

    def bispectrum(self, tstart=4, tend=5, NFFT1=2**10, OVR=0.5,
                   window="hann", mode="IQ",
                   fmax1=None, fmax2=None, display=True, pause=0.):

        if mode == "IQ":
            self.iq1.spectrum(tstart, tend, NFFT1, OVR, window, display=False, bgon=False)
            NFFT2 = int(self.iq1.dT * NFFT1 / self.iq2.dT + 0.5)
            self.iq2.spectrum(tstart, tend, NFFT2, OVR, window, display=False, bgon=False)

            self.cbsp = calc.cross_bispectral_analysis(self.iq1.sp.IQraw, self.iq1.sp.IQraw, self.iq2.sp.IQraw,
                                                       self.iq1.dT, self.iq1.dT, self.iq2.dT,
                                                       NFFT1, NFFT1, NFFT2,
                                                       flimx = fmax1, flimy = fmax2,
                                                       OVR=OVR, window=window)
            self.cbsp.freqz, self.biCohSq_fz, self.biCohSqErr_fz \
                = calc.average_bicoherence_at_f3_withErr(self.cbsp.freqx, self.cbsp.freqy,
                                                         self.cbsp.biCohSq, self.cbsp.biCohSqErr)

        else:
            print("未実装!")
            exit()

        plot.set("notebook", "ticks")

        if not display:
            original_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

        figdir = os.path.join(self.dirbase, "bicoherence")
        proc.ifNotMake(figdir)
        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT1}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        if fmax2 is not None:
            fname += f"_{fmax2}"
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT1} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biCohSq, cmap="plasma",
                           vmin=0, vmax=5./self.cbsp.NEns)
        cbar = plt.colorbar(im)
        cbar.set_label("bicoherence^2")

        ax.set_ylabel("Frequency 1 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)


        figdir = os.path.join(self.dirbase, "bicoherence_err")
        proc.ifNotMake(figdir)
        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT1}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT1} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biCohSqErr, cmap="plasma")
        cbar = plt.colorbar(im)
        cbar.set_label("bicoherence^2 err")

        ax.set_ylabel("Frequency 1 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)



        figdir = os.path.join(self.dirbase, "bicoherence_rer")
        proc.ifNotMake(figdir)
        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT1}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT1} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biCohSqRer, cmap="Greys", vmax=1, vmin=0)
        cbar = plt.colorbar(im)
        cbar.set_label("bicoherence^2 relative err")

        ax.set_ylabel("Frequency 1 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)


        figdir = os.path.join(self.dirbase, "phase")
        proc.ifNotMake(figdir)
        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT1}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT1} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biPhs, cmap="plasma")
        cbar = plt.colorbar(im)
        cbar.set_label("phase [rad]")

        ax.set_ylabel("Frequency 1 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)

        figdir = os.path.join(self.dirbase, "phase_err")
        proc.ifNotMake(figdir)
        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT1}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT1} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biPhsErr, cmap="plasma")
        cbar = plt.colorbar(im)
        cbar.set_label("phase err [rad]")

        ax.set_ylabel("Frequency 1 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)



        # figdir = os.path.join(self.dirbase, "bicoherence_fz")
        # proc.ifNotMake(figdir)
        # path = os.path.join(figdir, f"{fname}.png")
        #
        # fig, ax = plt.subplots(1)
        # l = ax.errorbar(self.cbsp.freqz, self.cbsp.biCohSq_fz, self.cbsp.biCohSqErr_fz,
        #                 fmt=".-", ecolor="grey")
        #
        # ax.set_ylabel("bicoherence^2 average")
        # ax.set_xlabel("Frequency 2 [Hz]")
        # ax.set_title(axtitle)
        # ax.set_ylim(0, 3. / np.sqrt(self.cbsp.NEns))
        #
        # plot.caption(fig, self.figtitle)
        # plot.capsave(fig, self.figtitle, fname, path)
        # if display:
        #     plot.check(pause)
        # else:
        #     plot.close(fig)



        if not display:
            matplotlib.use(original_backend)

        return self.cbsp

    # def total_bicoherence(self, freq):
    #
    #     self.cbsp.freq3, self.cbsp.bicohsq_f3, self.cbsp.N_components \
    #         = calc.total_bicoherence(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biCohSq)
    #     self.cbsp.bicohsq_f3_std = self.cbsp.N_components / self.cbsp.NEns

    def bispectrum_at_f3(self, f3_at=100e3, tstart=4, tend=5, NFFT1=2**10, OVR=0.5,
                         window="hann", mode="IQ", fmax1=None, fmax2=None):

        if mode == "IQ":
            self.iq1.spectrum(tstart, tend, NFFT1, OVR, window, display=False, bgon=False)
            NFFT2 = int(self.iq1.dT / self.iq2.dT * NFFT1 + 0.5)
            self.iq2.spectrum(tstart, tend, NFFT2, OVR, window, display=False, bgon=False)

            _, datlist = proc.getTimeIdxsAndDats(self.iq1.t, tstart, tend,
                                                 [self.iq1.IQ])
            IQraw1 = datlist[0]
            _, datlist = proc.getTimeIdxsAndDats(self.iq2.t, tstart, tend,
                                                 [self.iq2.IQ])
            IQraw2 = datlist[0]

            self.cbsp = calc.cross_bispectral_analysis_at_f3(f3_at, IQraw1, IQraw1, IQraw2,
                                                             self.iq1.dT, self.iq1.dT, self.iq2.dT,
                                                             NFFT1, NFFT1, NFFT2,
                                                             flimx=fmax1, flimy=fmax2,
                                                             OVR=OVR, window=window)

        else:
            print("未実装!")
            exit()

        return self.cbsp

    def bispectrum_in_f_range(self, fmin3=200e3, fmax3=500e3, tstart=4, tend=5, NFFT1=2**10, OVR=0.5,
                              window="hann", mode="IQ", flim1=None, flim2=None, coef_OV=1.03):

        if mode == "IQ":
            NFFT2 = int(self.iq1.dT * NFFT1 / self.iq2.dT + 0.5)

            _, datlist = proc.getTimeIdxsAndDats(self.iq1.t, tstart, tend,
                                                 [self.iq1.IQ])
            IQraw1 = datlist[0]
            _, datlist = proc.getTimeIdxsAndDats(self.iq2.t, tstart, tend,
                                                 [self.iq2.IQ])
            IQraw2 = datlist[0]

            self.cbsp = calc.cross_bispectral_analysis_in_f_range(fmin3, fmax3, IQraw1, IQraw1, IQraw2,
                                                                  self.iq1.dT, self.iq1.dT, self.iq2.dT,
                                                                  NFFT1, NFFT1, NFFT2,
                                                                  flimx=flim1, flimy=flim2,
                                                                  OVR=OVR, window=window, coef_OV=coef_OV)

        else:
            print("未実装!")
            exit()

        return self.cbsp

    def compare_intensities(self, NFFT1=2**10, NFFT2=2**10,
                            fmin1=200e3, fmax1=500e3, fmin2=100e3, fmax2=500e3,
                            dT=5e-3, bgon1=True, bgon2=False, Rat=None, Rfir=None,
                            window_len=5, sub_phi=True):

        self.iq1.specgram(NFFT=NFFT1, dT=dT, pause=0.1, display=False, sub_phi=sub_phi)
        self.iq2.specgram(NFFT=NFFT2, dT=dT, pause=0.1, display=False, sub_phi=sub_phi)
        if bgon1:
            self.iq1.BSmod()
            self.iq1.BSBackground()
        self.iq1.intensity(fmin1, fmax1, bgon=bgon1, sub_phi=sub_phi)
        if bgon2:
            self.iq2.BSmod()
            self.iq2.BSBackground()
        self.iq2.intensity(fmin2, fmax2, bgon=bgon2, sub_phi=sub_phi)

        if Rat is not None:
            self.iq1.ref_to_tsmap(Rat=Rat, skipnan=bgon1, bgon=bgon1)
            self.iq2.ref_to_tsmap(Rat=Rat, skipnan=bgon2, bgon=bgon2)
            self.cc = calc.cross_correlation_analysis(calc.interpolate_nan(self.iq1.tsmap.Iane),
                                                      calc.interpolate_nan(self.iq2.tsmap.Iane), dT,
                                                      window_len=window_len)
        elif Rfir is not None:
            self.iq1.ref_to_fir_nel(Rfir=Rfir, bgon=bgon1)
            self.iq2.ref_to_fir_nel(Rfir=Rfir, bgon=bgon2)
            self.cc = calc.cross_correlation_analysis(calc.interpolate_nan(self.iq1.spg.int.Ianel),
                                                      calc.interpolate_nan(self.iq2.spg.int.Ianel), dT,
                                                      window_len=window_len)
        # cc.delay or cc.lags: iq1's delay or lag time to iq2
        else:
            exit()

    def Sf_corrcoef(self, NFFT1=2**10, NFFT2=2**10, dT=2e-3, smoothing=True):

        self.iq1.specgram(NFFT=NFFT1, dT=dT, pause=0.1, display=False, sub_phi=False)
        self.iq2.specgram(NFFT=NFFT2, dT=dT, pause=0.1, display=False, sub_phi=False)

        if smoothing:
            self.iq1.specgram_smooth(twin_size=1, fwin_size=10, mode="gauss", sub_phi=False, display=False)
            self.iq2.specgram_smooth(twin_size=1, fwin_size=10, mode="gauss", sub_phi=False, display=False)

        stacked_psdamp = np.hstack((self.iq1.spg.psdamp, self.iq2.spg.psdamp))
        if smoothing:
            stacked_psdamp = np.hstack((np.sqrt(self.iq1.spg.psd_smooth), np.sqrt(self.iq2.spg.psd_smooth)))
        corr_mat = np.corrcoef(stacked_psdamp, rowvar=False)
        self.corr_iq1psdamp = corr_mat[:NFFT1, :NFFT1]
        self.corr_iq2psdamp = corr_mat[NFFT2:, NFFT2:]
        self.corr_iq12psdamp = corr_mat[:NFFT1, NFFT2:]


def cross_spectrogram(iq1, iq2, NFFT=2**10, window="hann", NEns=20, OVR=0.5, mode="ampIQ",
                      fmin=None, fmax=None,
                      freqlog=True, display=True, pause=0.):
    # mode: IQ, ampIQ, phaseIQ, ampIQ_vs_phaseIQ, cog

    plot.set("notebook", "ticks")

    if mode == "ampIQ":
        csg = calc.cross_spectrogram_2s(iq1.t, iq1.ampIQ, iq2.ampIQ,
                                        NFFT=NFFT, window=window, NEns=NEns, OVR=OVR)
    elif mode == "IQ":
        csg = calc.cross_spectrogram_2s(iq1.t, iq1.IQ, iq2.IQ,
                                        NFFT=NFFT, window=window, NEns=NEns, OVR=OVR)
    elif mode == "phaseIQ":
        csg = calc.cross_spectrogram_2s(iq1.t, iq1.phaseIQ, iq2.phaseIQ,
                                        NFFT=NFFT, window=window, NEns=NEns, OVR=OVR)
    else:
        print("未実装!!")
        exit()

    csg.mode = mode
    csg.fmin = fmin
    csg.fmax = fmax

    if not display:
        original_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

    figdir_base = os.path.join("Retrieve_MWRM_twinIQ", "cross_spectrogram")
    proc.ifNotMake(figdir_base)
    figdir = os.path.join(figdir_base, mode)
    proc.ifNotMake(figdir)

    fnm = f"{iq1.sn}_{iq1.subsn}_{iq1.tstart}_{iq1.tend}_" \
          f"{iq1.diagname}_{iq1.chI}_{iq1.chQ}_" \
          f"{iq2.diagname}_{iq2.chI}_{iq2.chQ}_" \
          f"{csg.mode}_{csg.NFFT}_{csg.window}_{csg.NEns}_{csg.fmin}_{csg.fmax}"
    path = os.path.join(figdir, f"{fnm}.png")

    figtitle = f"#{iq1.sn}-{iq1.subsn}\n" \
               f"{iq1.diagname} {iq1.chI} {iq1.chQ}\n" \
               f"{iq2.diagname} {iq2.chI} {iq2.chQ}"
    axtitle = f"{csg.mode} {csg.NFFT} {csg.window} {csg.NEns}"

    fig, ax = plt.subplots(1)
    im = ax.pcolormesh(csg.tsp,
                       csg.freq, csg.coh_sq.T,
                       cmap="plasma",
                       vmin=0, vmax=5./np.sqrt(NEns))
    cbar = plt.colorbar(im)
    cbar.set_label("Coherence^2")

    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [s]")
    ax.set_title(axtitle)
    if freqlog:
        ax.set_yscale("log")
    if fmin is not None:
        ax.set_ylim(bottom=fmin)
    if fmax is not None:
        ax.set_ylim(top=fmax)

    plot.caption(fig, figtitle)
    plot.capsave(fig, figtitle, fnm, path)
    if display:
        plot.check(pause)
    else:
        plot.close(fig)

    if not display:
        matplotlib.use(original_backend)

    return csg

def bispectrum_in_f_range(iq1, iq2, fmin3=200e3, fmax3=500e3,
                          tstart=4, tend=5, NFFT1=2**10, OVR=0.5,
                          window="hann", mode="IQ", flim1=None, flim2=None):

    if mode == "IQ":
        NFFT2 = int(iq1.dT * NFFT1 / iq2.dT + 0.5)

        _, datlist = proc.getTimeIdxsAndDats(iq1.t, tstart, tend,
                                             [iq1.IQ])
        IQraw1 = datlist[0]
        _, datlist = proc.getTimeIdxsAndDats(iq2.t, tstart, tend,
                                             [iq2.IQ])
        IQraw2 = datlist[0]

        cbsp = calc.cross_bispectral_analysis_in_f_range(fmin3, fmax3, IQraw1, IQraw1, IQraw2,
                                                         iq1.dT, iq1.dT, iq2.dT,
                                                         NFFT1, NFFT1, NFFT2,
                                                         flimx=flim1, flimy=flim2,
                                                         OVR=OVR, window=window)

    else:
        print("未実装!")
        exit()

    return cbsp


class tripletIQ:

    def __init__(self, sn=192781, sub=1, tstart=3, tend=6,
                 diag1="MWRM-COMB2", chI1=19, chQ1=20,
                 diag2="MWRM-COMB2", chI2=19, chQ2=20,
                 diag3="MWRM-COMB", chI3=1, chQ3=2):

        self.sn = sn
        self.sub = sub
        self.ts = tstart
        self.te = tend
        self.diag1 = diag1
        self.chI1 = chI1
        self.chQ1 = chQ1
        self.diag2 = diag2
        self.chI2 = chI2
        self.chQ2 = chQ2
        self.diag3 = diag3
        self.chI3 = chI3
        self.chQ3 = chQ3

        self.iq1 = IQ(sn, sub, tstart, tend, diag1, chI1, chQ1)
        self.iq2 = IQ(sn, sub, tstart, tend, diag2, chI2, chQ2)
        self.iq3 = IQ(sn, sub, tstart, tend, diag3, chI3, chQ3)

        self.dirbase = "Retrieve_MWRM_tripletIQ"
        proc.ifNotMake(self.dirbase)
        self.fnm_init = f"{sn}_{sub}_{tstart}_{tend}" \
                        f"_{diag1}_{chI1}_{chQ1}_{diag2}_{chI2}_{chQ2}_{diag3}_{chI3}_{chQ3}"
        self.figtitle = f"#{self.sn}-{self.sub}\n" \
                        f"1: {self.diag1} {self.chI1} {self.chQ1}\n" \
                        f"2: {self.diag2} {self.chI2} {self.chQ2}\n" \
                        f"3: {self.diag3} {self.chI3} {self.chQ3}"

    def bispectrum(self, tstart=4, tend=5, NFFT1=2**10, OVR=0.5,
                   window="hann", mode="IQ",
                   fmax1=None, fmax2=None, display=True, pause=0.):

        if mode == "IQ":
            self.iq1.spectrum(tstart, tend, NFFT1, OVR, window, display=False, bgon=False)
            NFFT2 = int(self.iq1.dT / self.iq2.dT * NFFT1 + 0.5)
            NFFT3 = int(self.iq1.dT / self.iq3.dT * NFFT1 + 0.5)
            self.iq2.spectrum(tstart, tend, NFFT2, OVR, window, display=False, bgon=False)
            self.iq3.spectrum(tstart, tend, NFFT3, OVR, window, display=False, bgon=False)

            NEns = calc.Nens_from_dtout(tend - tstart, self.iq1.dT, NFFT1, OVR)
            self.cbsp = calc.cross_bispectral_analysis(self.iq1.sp.IQraw, self.iq2.sp.IQraw, self.iq3.sp.IQraw,
                                                       self.iq1.dT, self.iq2.dT, self.iq3.dT,
                                                       NFFT1, NFFT2, NFFT3,
                                                       flimx = fmax1, flimy = fmax2,
                                                       OVR=OVR, window=window)
            self.cbsp.freqz, self.biCohSq_fz, self.biCohSqErr_fz \
                = calc.average_bicoherence_at_f3_withErr(self.cbsp.freqx, self.cbsp.freqy,
                                                         self.cbsp.biCohSq, self.cbsp.biCohSqErr)
        else:
            print("未実装!")
            exit()

        plot.set("notebook", "ticks")

        if not display:
            original_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

        figdir = os.path.join(self.dirbase, "bicoherence")
        proc.ifNotMake(figdir)
        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT1}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        if fmax2 is not None:
            fname += f"_{fmax2}"
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT1} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biCohSq, cmap="plasma",
                           vmin=0, vmax=5./self.cbsp.NEns)
        cbar = plt.colorbar(im)
        cbar.set_label("bicoherence^2")

        ax.set_ylabel("Frequency 2 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)


        figdir = os.path.join(self.dirbase, "bicoherence_err")
        proc.ifNotMake(figdir)
        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT1}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        if fmax2 is not None:
            fname += f"_{fmax2}"
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT1} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biCohSqErr, cmap="plasma")
        cbar = plt.colorbar(im)
        cbar.set_label("bicoherence^2 err")

        ax.set_ylabel("Frequency 2 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)

        figdir = os.path.join(self.dirbase, "bicoherence_rer")
        proc.ifNotMake(figdir)
        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT1}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        if fmax2 is not None:
            fname += f"_{fmax2}"
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT1} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biCohSqRer, cmap="Greys", vmax=1, vmin=0)
        cbar = plt.colorbar(im)
        cbar.set_label("bicoherence^2 relative err")

        ax.set_ylabel("Frequency 2 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)


        figdir = os.path.join(self.dirbase, "phase")
        proc.ifNotMake(figdir)
        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT1}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        if fmax2 is not None:
            fname += f"_{fmax2}"
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT1} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biPhs, cmap="plasma")
        cbar = plt.colorbar(im)
        cbar.set_label("phase [rad]")

        ax.set_ylabel("Frequency 2 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)

        figdir = os.path.join(self.dirbase, "phase_err")
        proc.ifNotMake(figdir)
        fname = f"{self.fnm_init}_{tstart}_{tend}_{NFFT1}_{OVR}_{window}_{mode}"
        if fmax1 is not None:
            fname += f"_{fmax1}"
        if fmax2 is not None:
            fname += f"_{fmax2}"
        path = os.path.join(figdir, f"{fname}.png")

        axtitle = f"{tstart}-{tend}s\n" \
                  f"{mode} {NFFT1} {OVR} {window}"

        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(self.cbsp.freqx, self.cbsp.freqy, self.cbsp.biPhsErr, cmap="plasma")
        cbar = plt.colorbar(im)
        cbar.set_label("phase err [rad]")

        ax.set_ylabel("Frequency 2 [Hz]")
        ax.set_xlabel("Frequency 1 [Hz]")
        ax.set_title(axtitle)
        if fmax1 is not None:
            ax.set_xlim(-fmax1, fmax1)
        if fmax2 is not None:
            ax.set_ylim(-fmax2, fmax2)

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)




        figdir = os.path.join(self.dirbase, "bicoherence_fz")
        proc.ifNotMake(figdir)
        path = os.path.join(figdir, f"{fname}.png")

        fig, ax = plt.subplots(1)
        l = ax.errorbar(self.cbsp.freqz, self.cbsp.biCohSq_fz, self.cbsp.biCohSqErr_fz,
                        fmt=".-", ecolor="grey")

        ax.set_ylabel("bicoherence^2 average")
        ax.set_xlabel("Frequency 3 [Hz]")
        ax.set_title(axtitle)
        ax.set_ylim(0, 3. / np.sqrt(self.cbsp.NEns))

        plot.caption(fig, self.figtitle)
        plot.capsave(fig, self.figtitle, fname, path)
        if display:
            plot.check(pause)
        else:
            plot.close(fig)



        if not display:
            matplotlib.use(original_backend)

        return self.cbsp
