from nasu import read, plot, proc, calc, get
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib
import os
from scipy.signal import welch
from scipy import fft
import numpy as np

plot.set("talk", "ticks")
pathCalib = os.path.join("C:\\pythonProject\\nasu\\calib_table.csv")


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

    def specgram(self, NFFT=2**10, ovr=0.5, window="hann", dT=5e-3, cmap="viridis", magnify=True, pause=0.1):

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

        axs[1].pcolorfast(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                          np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                          10*np.log10(self.spg.psd.T), cmap=self.spg.cmap)
        axs[1].set_ylabel("Frequency [Hz]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        plot.check(pause)


class IQ:

    def __init__(self, sn=187570, subsn=1, tstart=3., tend=6., diagname="MWRM-PXI", chI=11, chQ=12):

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
        self.IQ = self.I + 1.j * self.Q

        if diagname == "MWRM-COMB2" and chI in np.arange(1, 16 + 1, 2):

            dictIF = {1: 40, 3: 80, 5: 120, 7: 160,
                      9: 200, 11: 240, 13: 300, 15: 340}
            self.frLO = dictIF[chI]
            calibPrms_df = read.calibPrms_df_v2(pathCalib)
            self.ampratioIQ, self.offsetI, self.offsetQ, self.phasediffIQ = calibPrms_df.loc[self.frLO]

            self.I, self.Q = calc.calibIQComp2(self.I, self.Q,
                                               self.ampratioIQ, self.offsetI, self.offsetQ, self.phasediffIQ)

        self.ampIQ = np.abs(self.IQ)
        self.phaseIQ = np.unwrap(np.angle(self.IQ), 1.5 * np.pi)

    def specgram(self, NFFT=2**10, ovr=0.5, window="hann", dT=5e-3,
                 cmap="viridis", magnify=True, fmin=False, fmax=False, pause=0,
                 display=True):

        self.spg = calc.struct()

        self.spg.NFFT = NFFT
        self.spg.ovr = ovr
        self.spg.NOV = int(self.spg.NFFT * self.spg.ovr)
        self.spg.window = window
        self.spg.dT = dT
        self.spg.Fs = 1./self.spg.dT
        self.spg.NSamp = int(self.spg.dT / self.dT)
        # self.spg.NSamp = int(self.Fs / self.spg.Fs)
        self.spg.Nsp = self.size // self.spg.NSamp
        self.spg.cmap = cmap

        self.spg.t = self.t[:self.spg.Nsp * self.spg.NSamp].reshape((self.spg.Nsp, self.spg.NSamp)).mean(axis=-1)
        self.spg.IQarray = self.IQ[:self.spg.Nsp * self.spg.NSamp].reshape((self.spg.Nsp, self.spg.NSamp))
        self.spg.f, self.spg.psd = welch(x=self.spg.IQarray, fs=self.Fs, window="hann",
                                         nperseg=self.spg.NFFT, noverlap=self.spg.NOV,
                                         detrend="constant", scaling="density",
                                         axis=-1, average="mean")
        self.spg.f, self.spg.lindetpsd = welch(x=self.spg.IQarray, fs=self.Fs, window="hann",
                                               nperseg=self.spg.NFFT, noverlap=self.spg.NOV,
                                               detrend="linear", scaling="density",
                                               axis=-1, average="mean")
        self.spg.f = fft.fftshift(self.spg.f)
        self.spg.psd = fft.fftshift(self.spg.psd, axes=-1)
        self.spg.lindetpsd = fft.fftshift(self.spg.lindetpsd, axes=-1)
        self.spg.dF = self.dT * self.spg.NFFT

        if magnify:
            if fmin:
                self.spg.fmin = fmin
            else:
                self.spg.fmin = self.spg.dF
            if fmax:
                self.spg.fmax = fmax
            else:
                self.spg.fmax = self.Fs / 2

            _idx = np.where((np.abs(self.spg.f) > self.spg.fmin) & (np.abs(self.spg.f) < self.spg.fmax))[0]
            self.spg.vmax = self.spg.psd[:, _idx].max()
            self.spg.vmin = self.spg.psd[:, _idx].min()
        else:
            self.spg.vmax = self.spg.psd.max()
            self.spg.vmin = self.spg.psd.min()

        if not display:
            matplotlib.use('Agg')

        figdir = "Retrieve_MWRM"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}"
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

        axs[2].pcolorfast(np.append(self.spg.t - 0.5 * self.spg.dT, self.spg.t[-1] + 0.5 * self.spg.dT),
                          np.append(self.spg.f - 0.5 * self.spg.dF, self.spg.f[-1] + 0.5 * self.spg.dF),
                          10*np.log10(self.spg.psd.T),
                          cmap=self.spg.cmap, vmin=10*np.log10(self.spg.vmin), vmax=10*np.log10(self.spg.vmax))
        if magnify:
            axs[2].set_ylim(- self.spg.fmax, self.spg.fmax)
        axs[2].set_ylabel("Frequency [Hz]")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_xlim(self.tstart, self.tend)

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    def spectrum(self, tstart=4.9, tend=5.0, NFFT=2**10, ovr=0.5, window="hann", pause=0, display=True):

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
        self.sp.dF = self.dT * self.sp.NFFT

        self.sp.t = (self.sp.tstart + self.sp.tend) / 2
        self.sp.fIQ, self.sp.psdIQ = welch(x=self.sp.IQraw, fs=self.Fs, window="hann",
                                                       nperseg=self.sp.NFFT, noverlap=self.sp.NOV,
                                                       detrend="constant", scaling="density",
                                                       average="mean")
        self.sp.fIQ = fft.fftshift(self.sp.fIQ)
        self.sp.psdIQ = fft.fftshift(self.sp.psdIQ)

        self.sp.fI, self.sp.psdI = welch(x=self.sp.Iraw, fs=self.Fs, window="hann",
                                                     nperseg=self.sp.NFFT, noverlap=self.sp.NOV,
                                                     detrend="constant", scaling="density",
                                                     average="mean")
        self.sp.fQ, self.sp.psdQ = welch(x=self.sp.Qraw, fs=self.Fs, window="hann",
                                                     nperseg=self.sp.NFFT, noverlap=self.sp.NOV,
                                                     detrend="constant", scaling="density",
                                                     average="mean")

        if not display:
            matplotlib.use('Agg')

        figdir = "Retrieve_MWRM"
        proc.ifNotMake(figdir)

        fnm = f"{self.sn}_{self.subsn}_{self.sp.t}_{self.diagname}_{self.chI}_{self.chQ}_IQsp"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.sp.t}s\n" \
                f"({self.sp.tstart}-{self.sp.tend}s)\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"
        fig, ax = plt.subplots(figsize=(6, 6), num=fnm)
        ax.plot(self.sp.fIQ, 10 * np.log10(self.sp.psdIQ), c="black")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [dBV/$\sqrt{\\rm{Hz}}$]")

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

        fnm2 = f"{self.sn}_{self.subsn}_{self.sp.t}_{self.diagname}_{self.chI}_{self.chQ}_sp"
        path2 = os.path.join(figdir, f"{fnm2}.png")
        title2 = f"#{self.sn}-{self.subsn} {self.sp.t}s\n" \
                 f"({self.sp.tstart}-{self.sp.tend}s)\n" \
                 f"{self.diagname} ch:{self.chI},{self.chQ}"
        fig2, ax2 = plt.subplots(figsize=(6, 6), num=fnm2)
        ax2.plot(self.sp.fI, 10 * np.log10(self.sp.psdI), c="blue")
        ax2.plot(self.sp.fQ, 10 * np.log10(self.sp.psdQ), c="orange")
        ax2.legend(["I", "Q"])
        ax2.set_xscale("log")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("PSD [dBV/$\sqrt{\\rm{Hz}}$]")

        plot.caption(fig2, title2, hspace=0.1, wspace=0.1)
        plot.capsave(fig2, title2, fnm2, path2)

        if display:
            plot.check(pause)
        else:
            plot.close(fig2)

    def intensity(self, fmin=150e3, fmax=490e3, linear_detrend=True):

        self.spg.int = calc.struct()
        self.spg.int.fmin = fmin
        self.spg.int.fmax = fmax
        idx_use = np.where((np.abs(self.spg.f) >= self.spg.int.fmin) & (np.abs(self.spg.f) <= self.spg.int.fmax))[0]
        if linear_detrend:
            spec_Sk = self.spg.lindetpsd[:, idx_use]
        else:
            spec_Sk = self.spg.psd[:, idx_use]

        self.spg.int.Sk = np.sum(spec_Sk, axis=-1) * self.spg.dF
        self.spg.int.Ia = np.sqrt(self.spg.int.Sk)

        self.spg.int.Sknorm = self.spg.int.Sk / np.max(self.spg.int.Sk)
        self.spg.int.Ianorm = self.spg.int.Ia / np.max(self.spg.int.Ia)

    def phasespec(self, NFFT=2**10, ovr=0.5, window="hann", dT=2e-3,
                  cmap="viridis", pause=0, display=True):

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

    def BSmod(self, threshold=-0.8, diagmod="MWRM-PXI", chmod=4):

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

    def BSBackground(self):

        psd_list = []
        for offTs, offTe in self.mod.ets:
            _idxs, datList = proc.getTimeIdxsAndDats(self.t, offTs, offTe, [self.IQ])
            offIQ = datList[0]

            f, psd = welch(x=offIQ, fs=self.Fs, window="hann",
                           nperseg=self.spg.NFFT, noverlap=self.spg.NOV,
                           detrend="constant", scaling="density",
                           axis=-1, average="mean")
            psd = fft.fftshift(psd)
            psd_list.append(psd)

        self.bg = calc.struct()
        self.bg.ets = self.mod.ets
        self.bg.t = self.mod.ets.mean(axis=-1)
        self.bg.psd = np.array(psd_list)

        idx_use = np.where((np.abs(self.spg.f) >= self.spg.int.fmin) & (np.abs(self.spg.f) <= self.spg.int.fmax))[0]
        spec_Sk = self.bg.psd[:, idx_use]

        self.bg.Sk = np.sum(spec_Sk, axis=-1) * self.spg.dF
        self.bg.Ia = np.sqrt(self.bg.Sk)

        self.bg.Sknorm = self.bg.Sk / self.spg.int.Sk.max()
        self.bg.Ianorm = self.bg.Ia / self.spg.int.Ia.max()

    def plot_intensity(self, bgon=True):

        plot.set("paper", "ticks")

        self.ech = get.ech_v2(sn=self.sn, tstart=self.tstart, tend=self.tend)
        self.nb = get.nb_alldev(sn=self.sn, tstart=self.tstart, tend=self.tend)
        self.nel = get.nel(sn=self.sn, sub=self.subsn, tstart=self.tstart, tend=self.tend)
        self.wp = get.wp(sn=self.sn, sub=self.subsn, tstart=self.tstart, tend=self.tend)

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
        self.intensity(fmin=3e3, fmax=30e3)
        ax[4].plot(self.spg.t, self.spg.int.Ia, c="blue",
                   label=f"{self.spg.int.fmin*1e-3}-{self.spg.int.fmax*1e-3} kHz")
        if bgon:
            self.BSBackground()
            ax[4].plot(self.bg.t, self.bg.Ia, ".", c="grey", label="bg")
        ax[4].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        self.intensity(fmin=30e3, fmax=150e3)
        ax[5].plot(self.spg.t, self.spg.int.Ia, c="green",
                   label=f"{self.spg.int.fmin * 1e-3}-{self.spg.int.fmax * 1e-3} kHz")
        if bgon:
            self.BSBackground()
            ax[5].plot(self.bg.t, self.bg.Ia, ".", c="grey", label="bg")
        ax[5].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        self.intensity(fmin=100e3, fmax=490e3)
        ax[6].plot(self.spg.t, self.spg.int.Ia, c="red",
                   label=f"{self.spg.int.fmin * 1e-3}-{self.spg.int.fmax * 1e-3} kHz")
        if bgon:
            self.BSBackground()
            ax[6].plot(self.bg.t, self.bg.Ia, ".", c="grey", label="bg")
        ax[6].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        self.intensity(fmin=20e3, fmax=200e3)
        ax[7].plot(self.spg.t, self.spg.int.Ia, c="green",
                   label=f"{self.spg.int.fmin * 1e-3}-{self.spg.int.fmax * 1e-3} kHz")
        if bgon:
            self.BSBackground()
            ax[7].plot(self.bg.t, self.bg.Ia, ".", c="grey", label="bg")
        ax[7].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        self.intensity(fmin=200e3, fmax=500e3)
        ax[8].plot(self.spg.t, self.spg.int.Ia, c="red",
                   label=f"{self.spg.int.fmin * 1e-3}-{self.spg.int.fmax * 1e-3} kHz")
        if bgon:
            self.BSBackground()
            ax[8].plot(self.bg.t, self.bg.Ia, ".", c="grey", label="bg")
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

        plt.show()

