from nasu import read, plot, proc, calc
import matplotlib.pyplot as plt
from matplotlib import mlab
import os
from scipy.signal import welch
from scipy import fft
import numpy as np

plot.set("talk", "ticks")


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

    def specgram(self, NFFT=2**10, ovr=0.5, window="hann", dT=5e-3, cmap="jet", magnify=True):

        self.specgram = calc.struct()
        self.specgram.NFFT = NFFT
        self.specgram.ovr = ovr
        self.specgram.NOV = int(self.specgram.NFFT * self.specgram.ovr)
        self.specgram.window = window
        self.specgram.dT = dT
        self.specgram.Fssp = 1./self.specgram.dT
        self.specgram.NSamp = int(self.Fs / self.specgram.Fssp)
        self.specgram.Nsp = self.size // self.specgram.NSamp
        self.specgram.cmap = cmap

        self.specgram.t = self.t[:self.specgram.Nsp * self.specgram.NSamp].reshape((self.specgram.Nsp, self.specgram.NSamp)).mean(axis=-1)
        self.specgram.darray = self.d[:self.specgram.Nsp * self.specgram.NSamp].reshape((self.specgram.Nsp, self.specgram.NSamp))
        self.specgram.f, self.specgram.psd = welch(x=self.specgram.darray, fs=self.Fs, window="hann",
                                                   nperseg=self.specgram.NFFT, noverlap=self.specgram.NOV,
                                                   detrend="constant", scaling="density",
                                                   axis=-1, average="mean")
        self.specgram.dF = self.dT * self.specgram.NFFT

        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 3]})

        axs[0].plot(self.t, self.d, c="black", lw=0.1)
        axs[0].set_ylabel("[V]")
        if magnify:
            p2p = self.d.max() - self.d.min()
            axs[0].set_ylim(self.d.min() - p2p * 0.05, self.d.max() + p2p * 0.05)
        else:
            axs[0].set_ylim( - float(self.dprms["Range"][0]), float(self.dprms["Range"][0]))

        axs[1].pcolorfast(np.append(self.specgram.t - 0.5 * self.specgram.dT, self.specgram.t[-1] + 0.5 * self.specgram.dT),
                          np.append(self.specgram.f - 0.5 * self.specgram.dF, self.specgram.f[-1] + 0.5 * self.specgram.dF),
                          10*np.log10(self.specgram.psd.T), cmap=self.specgram.cmap)
        axs[1].set_ylabel("Frequency [Hz]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_xlim(self.tstart, self.tend)

        figdir = "Retrieve_MWRM"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.ch}.png"
        path = os.path.join(figdir, fnm)
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.ch}"

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        plot.check(0.1)


class IQ:

    def __init__(self, sn=187570, subsn=1, tstart=3., tend=6., diagname="MWRM-PXI", chI=11, chQ=12):

        self.sn = sn
        self.subsn = subsn
        self.tstart = tstart
        self.tend = tend
        self.diagname = diagname
        self.chI = chI
        self.chQ = chQ

        self.t, self.I, self.Q, self.dT, self.Fs, self.size, self.tprms, self.Iprms, self.Qprms = \
            read.LHD_IQ_et_v2(sn=sn, subsn=subsn, diagname=diagname, chs=[chI, chQ], et=(tstart, tend))
        self.IQ = self.I + 1.j * self.Q


    def specgram(self, NFFT=2**10, ovr=0.5, window="hann", dT=5e-3, cmap="jet", magnify=True):

        self.specgram = calc.struct()

        self.specgram.NFFT = NFFT
        self.specgram.ovr = ovr
        self.specgram.NOV = int(self.specgram.NFFT * self.specgram.ovr)
        self.specgram.window = window
        self.specgram.dT = dT
        self.specgram.Fs = 1./self.specgram.dT
        self.specgram.NSamp = int(self.Fs / self.specgram.Fs)
        self.specgram.Nsp = self.size // self.specgram.NSamp
        self.specgram.cmap = cmap
        self.specgram.noverlap = int(self.specgram.NFFT * self.specgram.ovr)

        self.specgram.t = self.t[:self.specgram.Nsp * self.specgram.NSamp].reshape((self.specgram.Nsp, self.specgram.NSamp)).mean(axis=-1)
        self.specgram.IQarray = self.IQ[:self.specgram.Nsp * self.specgram.NSamp].reshape((self.specgram.Nsp, self.specgram.NSamp))
        self.specgram.f, self.specgram.psd = welch(x=self.specgram.IQarray, fs=self.Fs, window="hann",
                                    nperseg=self.specgram.NFFT, noverlap=self.specgram.NOV,
                                    detrend="constant", scaling="density",
                                    axis=-1, average="mean")
        self.specgram.f = fft.fftshift(self.specgram.f)
        self.specgram.psd = fft.fftshift(self.specgram.psd, axes=-1)
        self.specgram.dF = self.dT * self.specgram.NFFT

        fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(5, 10), gridspec_kw={'height_ratios': [1, 1, 3]})

        axs[0].plot(self.t, self.I, c="black", lw=0.1)
        axs[0].set_ylabel("I [V]")
        if magnify:
            p2p = self.I.max() - self.I.min()
            axs[0].set_ylim(self.I.min() - p2p * 0.05, self.I.max() + p2p * 0.05)
        else:
            axs[0].set_ylim( - float(self.Iprms["Range"][0]), float(self.Iprms["Range"][0]))

        axs[1].plot(self.t, self.Q, c="black", lw=0.1)
        if magnify:
            p2p = self.Q.max() - self.Q.min()
            axs[1].set_ylim(self.Q.min() - p2p * 0.05, self.Q.max() + p2p * 0.05)
        else:
            axs[1].set_ylim(- float(self.Qprms["Range"][0]), float(self.Qprms["Range"][0]))
        axs[1].set_ylabel("Q [V]")

        axs[2].pcolorfast(np.append(self.specgram.t - 0.5 * self.specgram.dT, self.specgram.t[-1] + 0.5 * self.specgram.dT),
                          np.append(self.specgram.f - 0.5 * self.specgram.dF, self.specgram.f[-1] + 0.5 * self.specgram.dF),
                          10*np.log10(self.specgram.psd.T), cmap="jet")
        axs[2].set_ylabel("Frequency [Hz]")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_xlim(self.tstart, self.tend)

        figdir = "Retrieve_MWRM"
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.subsn}_{self.tstart}_{self.tend}_{self.diagname}_{self.chI}_{self.chQ}.png"
        path = os.path.join(figdir, fnm)
        title = f"#{self.sn}-{self.subsn} {self.tstart}-{self.tend}s\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)

        plot.check(0.1)


    def spectrum(self, tstart=4.0, tend=5.0, NFFT=2**10, ovr=0.5, window="hann"):

        self.spectrum = calc.struct()
        self.spectrum.tstart = tstart
        self.spectrum.tend = tend

        self.spectrum.NFFT = NFFT
        self.spectrum.ovr = ovr
        self.spectrum.NOV = int(self.spectrum.NFFT * self.spectrum.ovr)
        self.spectrum.window = window

        _, datlist = proc.getTimeIdxsAndDats(self.t, self.spectrum.tstart, self.spectrum.tend,
                                             [self.t, self.IQ, self.I, self.Q])
        self.spectrum.traw, self.spectrum.IQraw, self.spectrum.Iraw, self.spectrum.Qraw = datlist
        self.spectrum.NSamp = self.spectrum.traw.size
        self.spectrum.dF = self.dT * self.spectrum.NFFT

        self.spectrum.t = (self.spectrum.tstart + self.spectrum.tend) / 2
        self.spectrum.fIQ, self.spectrum.psdIQ = welch(x=self.spectrum.IQraw, fs=self.Fs, window="hann",
                                                       nperseg=self.spectrum.NFFT, noverlap=self.spectrum.NOV,
                                                       detrend="constant", scaling="density",
                                                       average="mean")
        self.spectrum.fIQ = fft.fftshift(self.spectrum.fIQ)
        self.spectrum.psdIQ = fft.fftshift(self.spectrum.psdIQ)

        self.spectrum.fI, self.spectrum.psdI = welch(x=self.spectrum.Iraw, fs=self.Fs, window="hann",
                                                     nperseg=self.spectrum.NFFT, noverlap=self.spectrum.NOV,
                                                     detrend="constant", scaling="density",
                                                     average="mean")
        self.spectrum.fQ, self.spectrum.psdQ = welch(x=self.spectrum.Qraw, fs=self.Fs, window="hann",
                                                     nperseg=self.spectrum.NFFT, noverlap=self.spectrum.NOV,
                                                     detrend="constant", scaling="density",
                                                     average="mean")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.spectrum.fIQ, 10 * np.log10(self.spectrum.psdIQ), c="black")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [dBV/$\sqrt{\\rm{Hz}}$]")

        figdir = "Retrieve_MWRM"
        proc.ifNotMake(figdir)

        fnm = f"{self.sn}_{self.subsn}_{self.spectrum.t}_{self.diagname}_{self.chI}_{self.chQ}_IQspectrum.png"
        path = os.path.join(figdir, fnm)
        title = f"#{self.sn}-{self.subsn} {self.spectrum.t}s\n" \
                f"({self.spectrum.tstart}-{self.spectrum.tend}s)\n" \
                f"{self.diagname} ch:{self.chI},{self.chQ}"

        plot.caption(fig, title, hspace=0.1, wspace=0.1)
        plot.capsave(fig, title, fnm, path)
        plot.check(0.1)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.plot(self.spectrum.fI, 10 * np.log10(self.spectrum.psdI), c="blue")
        ax2.plot(self.spectrum.fQ, 10 * np.log10(self.spectrum.psdQ), c="orange")
        ax2.legend(["I", "Q"])
        ax2.set_xscale("log")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("PSD [dBV/$\sqrt{\\rm{Hz}}$]")

        fnm2 = f"{self.sn}_{self.subsn}_{self.spectrum.t}_{self.diagname}_{self.chI}_{self.chQ}_spectrum.png"
        path2 = os.path.join(figdir, fnm2)
        title2 = f"#{self.sn}-{self.subsn} {self.spectrum.t}s\n" \
                 f"({self.spectrum.tstart}-{self.spectrum.tend}s)\n" \
                 f"{self.diagname} ch:{self.chI},{self.chQ}"

        plot.caption(fig2, title2, hspace=0.1, wspace=0.1)
        plot.capsave(fig2, title2, fnm2, path2)

        plot.check(0.1)
