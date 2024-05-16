from nasu import read, plot, proc, calc, get_eg
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
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

    def __init__(self, sn=187570, subsn=1, tstart=3., tend=6., diagname="MWRM-PXI", chI=11, chQ=12, phase_unwrap=True):

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
            self.Q *= -1

        elif diagname == "MWRM-COMB" and chI in [1, 3, 5, 7]:
            self.Q *= -1

        self.IQ = self.I + 1.j * self.Q
        self.ampIQ = np.abs(self.IQ)
        if phase_unwrap:
            self.phaseIQ = np.unwrap(np.angle(self.IQ), 1.5 * np.pi)
        else:
            self.phaseIQ = np.angle(self.IQ)

    def specgram(self, NFFT=2**10, ovr=0.5, window="hann", dT=5e-3,
                 cmap="viridis", magnify=True, fmin=False, fmax=False, pause=0,
                 display=True, detrend="constant"):

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
                                         return_onesided=False,
                                         detrend=detrend, scaling="density",
                                         axis=-1, average="mean")
        # self.spg.f, self.spg.lindetpsd = welch(x=self.spg.IQarray, fs=self.Fs, window="hann",
        #                                        nperseg=self.spg.NFFT, noverlap=self.spg.NOV,
        #                                        return_onesided=False,
        #                                        detrend="linear", scaling="density",
        #                                        axis=-1, average="mean")
        self.spg.f = fft.fftshift(self.spg.f)
        self.spg.psd = fft.fftshift(self.spg.psd, axes=-1)
        self.spg.psddB = 10 * np.log10(self.spg.psd)
        # self.spg.lindetpsd = fft.fftshift(self.spg.lindetpsd, axes=-1)
        # self.spg.lindetpsddB = 10 * np.log10(self.spg.lindetpsd)
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
            self.spg.vmaxdB = 10*np.log10(self.spg.vmax)
            self.spg.vmin = self.spg.psd[:, _idx].min()
            self.spg.vmindB = 10*np.log10(self.spg.vmin)
        else:
            self.spg.vmax = self.spg.psd.max()
            self.spg.vmaxdB = 10*np.log10(self.spg.vmax)
            self.spg.vmin = self.spg.psd.min()
            self.spg.vmindB = 10*np.log10(self.spg.vmin)

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
        plot.capsave(fig, title, fnm, path)

        if display:
            plot.check(pause)
        else:
            plot.close(fig)

    def specgram_amp(self, NFFT=2**10, ovr=0.5, window="hann", dT=5e-3,
                     cmap="viridis", magnify=True,
                     fmin=False, fmax=False, logfreq=False,
                     pause=0, display=True, detrend="constant"):

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
        # self.spg.amp.f, self.spg.amp.lindetpsd = welch(x=self.spg.amp.ampIQarray, fs=self.Fs, window="hann",
        #                                                nperseg=self.spg.amp.NFFT, noverlap=self.spg.amp.NOV,
        #                                                return_onesided=True,
        #                                                detrend="linear", scaling="density",
        #                                                axis=-1, average="mean")
        self.spg.amp.f = self.spg.amp.f
        self.spg.amp.psd = self.spg.amp.psd
        self.spg.amp.psddB = 10 * np.log10(self.spg.amp.psd)
        # self.spg.amp.lindetpsd = self.spg.amp.lindetpsd
        # self.spg.amp.lindetpsddB = 10 * np.log10(self.spg.amp.lindetpsd)
        self.spg.amp.dF = self.dT * self.spg.amp.NFFT

        if magnify:
            if fmin:
                self.spg.amp.fmin = fmin
            else:
                self.spg.amp.fmin = self.spg.amp.dF
            if fmax:
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

        figdir = "Retrieve_MWRM_amp"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}"
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
                       pause=0, display=True, detrend="linear"):
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
        self.spg.phase.dF = self.dT * self.spg.phase.NFFT

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

        figdir = "Retrieve_MWRM_phase"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}"
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

        axs[1].pcolormesh(np.append(self.spg.phase.t - 0.5 * self.spg.phase.dT, self.spg.phase.t[-1] + 0.5 * self.spg.phase.dT),
                          np.append(self.spg.phase.f - 0.5 * self.spg.phase.dF, self.spg.phase.f[-1] + 0.5 * self.spg.phase.dF),
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
                        cmap="viridis", magnify=True, fmin=1e3, fmax=500e3, pause=0,
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

        axs[2].pcolormesh(np.append(self.cog.t - 0.5 * self.cog.dT, self.cog.t[-1] + 0.5 * self.cog.dT),
                          np.append(self.cog.f - 0.5 * self.cog.dF, self.cog.f[-1] + 0.5 * self.cog.dF),
                          self.cog.psddB.T,
                          cmap=self.cog.cmap, vmin=self.cog.vmindB, vmax=self.cog.vmaxdB)
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

    def specgram_pp(self, NFFT=2 ** 7, ovr=0.5, window="hann", dT=2e-2,
                     cmap="viridis", magnify=True,
                     fmin=False, fmax=False, logfreq=False,
                     pause=0, display=True, detrend="constant"):
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
        self.pp.spg.f, self.pp.spg.psd = welch(x=self.pp.spg.cogarray, fs=self.pp.Fs, window="hann",
                                                 nperseg=self.pp.spg.NFFT, noverlap=self.pp.spg.NOV,
                                                 return_onesided=True,
                                                 detrend=detrend, scaling="density",
                                                 axis=-1, average="mean")
        # self.pp.spg.f, self.pp.spg.lindetpsd = welch(x=self.pp.spg.phaseIQarray, fs=self.Fs, window="hann",
        #                                                    nperseg=self.pp.spg.NFFT, noverlap=self.pp.spg.NOV,
        #                                                    return_onesided=True,
        #                                                    detrend="linear", scaling="density",
        #                                                    axis=-1, average="mean")
        self.pp.spg.psddB = 10 * np.log10(self.pp.spg.psd)
        # self.pp.spg.lindetpsd = self.pp.spg.lindetpsd
        # self.pp.spg.lindetpsddB = 10 * np.log10(self.pp.spg.lindetpsd)
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

        figdir = "Retrieve_MWRM_pp"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}"
        if fmin:
            fnm += f"_min{fmin * 1e-3}kHz"
        if fmax:
            fnm += f"_max{fmax * 1e-3}kHz"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"
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
                       pause=0, display=True, detrend="constant"):
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

        figdir = "Retrieve_MWRM_cog"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}"
        if fmin:
            fnm += f"_min{fmin * 1e-3}kHz"
        if fmax:
            fnm += f"_max{fmax * 1e-3}kHz"
        path = os.path.join(figdir, f"{fnm}.png")
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"
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

        ani.save(f"{fname}.gif")
        plt.show(ani)

    def spectrum(self, tstart=4.9, tend=5.0, NFFT=2**10, ovr=0.5, window="hann", pause=0, display=True, bgon=False):

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
        if bgon:
            tidx_s = np.argmin(np.abs(self.bg.t - self.sp.tstart))
            tidx_e = np.argmin(np.abs(self.bg.t - self.sp.tend))
            for i in range(tidx_s, tidx_e + 1):
                ax.scatter(self.spg.f, 10 * np.log10(self.bg.psd[i]), marker=".", c="grey")
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

        return self.sp

    def intensity(self, fmin=150e3, fmax=490e3):

        self.spg.int = calc.struct()
        self.spg.int.fmin = fmin
        self.spg.int.fmax = fmax
        idx_use = np.where((np.abs(self.spg.f) >= self.spg.int.fmin) & (np.abs(self.spg.f) <= self.spg.int.fmax))[0]
        spec_Sk = self.spg.psd[:, idx_use]

        self.spg.int.Sk = np.sum(spec_Sk, axis=-1) * self.spg.dF
        self.spg.int.Ia = np.sqrt(self.spg.int.Sk)

        self.spg.int.Sknorm = self.spg.int.Sk / np.max(self.spg.int.Sk)
        self.spg.int.Ianorm = self.spg.int.Ia / np.max(self.spg.int.Ia)

    def dopplershift(self, fmin=3e3, fmax=1250e3):

        self.spg.DS = calc.struct()
        self.spg.DS.fmin = fmin
        self.spg.DS.fmax = fmax
        idx_use = np.where((np.abs(self.spg.f) >= self.spg.DS.fmin) & (np.abs(self.spg.f) <= self.spg.DS.fmax))[0]
        freq_DS = self.spg.f[idx_use]
        spec_DS = self.spg.psd[:, idx_use]

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

    def plot_intensity(self, bgon=False, pause=0):

        plot.set("paper", "ticks")

        self.ech = get_eg.ech_v2(sn=self.sn, tstart=self.tstart, tend=self.tend)
        self.nb = get_eg.nb_alldev(sn=self.sn, tstart=self.tstart, tend=self.tend)
        self.nel = get_eg.nel(sn=self.sn, sub=self.subsn, tstart=self.tstart, tend=self.tend)
        self.wp = get_eg.wp(sn=self.sn, sub=self.subsn, tstart=self.tstart, tend=self.tend)

        figdir = "Retrieve_MWRM_intensity"
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

        plot.caption(fig, title)
        plot.capsave(fig, title, fnm, path)
        plot.check(pause)
