"""
2024.2.20 19:10
by Tatsuhiro Nasu
"""

from nasu import read, myEgdb
import matplotlib.pyplot as plt
import numpy as np


class struct:
    pass


class comb_R:

    def __init__(self, sn=174070, sub=1):

        self.sn = sn
        self.sub = sub

        self.egnames = ["mwrm_comb_R_Iamp", "mwrm_comb_R_Vp", "mwrm_comb_R_RAY_OUT"]

        self.eg_Iamp = myEgdb.LoadEG(self.egnames[0], sn=sn, sub=sub, flg_remove=True)
        self.eg_Vp = myEgdb.LoadEG(self.egnames[1], sn=sn, sub=sub, flg_remove=True)
        self.eg_RAY_OUT = myEgdb.LoadEG(self.egnames[2], sn=sn, sub=sub, flg_remove=True)

        self.ch1 = struct()
        self.ch2 = struct()
        self.ch3 = struct()
        self.ch4 = struct()
        self.ch5 = struct()
        self.ch6 = struct()
        self.ch7 = struct()
        self.ch8 = struct()
        self.freqs = [27.7, 29.0, 30.5, 32.0, 33.3, 34.8, 37.0, 38.3]
        self.chs = [self.ch1, self.ch2, self.ch3, self.ch4, self.ch5, self.ch6, self.ch7, self.ch8]

        for i in range(len(self.chs)):
            self.chs[i].freq = self.freqs[i]
            self.chs[i].Iamp = struct()
            self.chs[i].Vp = struct()
            self.chs[i].RAY_OUT = struct()

            self.chs[i].Iamp.time = self.eg_Iamp.dims(dim=0)
            self.chs[i].Iamp.lowf = self.eg_Iamp.trace_of(name=f"Amplitude (3-30kHz)  {self.chs[i].freq:0.3f} [GHz]", dim=0, other_idxs=[0])
            self.chs[i].Iamp.midf = self.eg_Iamp.trace_of(name=f"Amplitude (30-150kHz)  {self.chs[i].freq:0.3f} [GHz]", dim=0, other_idxs=[0])
            self.chs[i].Iamp.highf = self.eg_Iamp.trace_of(name=f"Amplitude (150-490kHz)  {self.chs[i].freq:0.3f} [GHz]", dim=0, other_idxs=[0])
            self.chs[i].Iamp.allf = self.eg_Iamp.trace_of(name=f"Amplitude (20-490kHz)  {self.chs[i].freq:0.3f} [GHz]", dim=0, other_idxs=[0])

            self.chs[i].Vp.time = self.eg_Vp.dims(dim=0)
            self.chs[i].Vp.raw_meankp = self.eg_Vp.trace_of(name=f'V perp  {self.chs[i].freq:0.3f} [GHz] (raw) mean k_perp', dim=0, other_idxs=[0])
            self.chs[i].Vp.smooth_meankp = self.eg_Vp.trace_of(name=f'V perp (smoothing)  {self.chs[i].freq:0.3f} [GHz](smoothing) mean k_perp', dim=0, other_idxs=[0])
            self.chs[i].Vp.raw_eachkp = self.eg_Vp.trace_of(name=f'V perp  {self.chs[i].freq:0.3f} [GHz] (raw) each k_perp', dim=0, other_idxs=[0])
            self.chs[i].Vp.smooth_eachkp = self.eg_Vp.trace_of(name=f'V perp (smoothing)  {self.chs[i].freq:0.3f} [GHz](smoothing) each k_perp', dim=0, other_idxs=[0])
            self.chs[i].Vp.error = self.eg_Vp.trace_of(name=f'error casused by variations of ray {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].Vp.Er = self.eg_Vp.trace_of(name=f'Er related  {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])

            self.chs[i].RAY_OUT.time = self.eg_RAY_OUT.dims(dim=0)
            self.chs[i].RAY_OUT.meankp = self.eg_RAY_OUT.trace_of(name=f'mean k_perp  {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])[0]
            self.chs[i].RAY_OUT.kp = self.eg_RAY_OUT.trace_of(name=f'k_perp each time  {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].RAY_OUT.kp_error = self.eg_RAY_OUT.trace_of(name=f'delta k_perp error estimation  {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].RAY_OUT.reff = self.eg_RAY_OUT.trace_of(name=f'reff {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].RAY_OUT.a99 = self.eg_RAY_OUT.trace_of(name=f'a99 {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].RAY_OUT.rho = self.eg_RAY_OUT.trace_of(name=f'reff/a99 {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].RAY_OUT.B = self.eg_RAY_OUT.trace_of(name=f'B  {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])

    def plot_ch(self, freq=27.7):

        i = np.argmin(np.abs(np.asarray(self.freqs) - freq))
        title = f"mwrm_comb_R_#{self.sn}-{self.sub}_{freq:.3f}GHz"

        fig, ax = plt.subplots(8, sharex=True, num=title, figsize=(6, 8))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[3].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.lowf, c="blue", label="3-30 kHz")
        ax[3].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.midf, c="green", label="30-150 kHz")
        ax[3].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.highf, c="red", label="150-490 kHz")
        ax[3].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.allf, c="black", ls="--", label="20-490 kHz")
        ax[3].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[4].plot(self.chs[i].Vp.time, self.chs[i].Vp.raw_meankp, c="black", label="raw")
        ax[4].plot(self.chs[i].Vp.time, self.chs[i].Vp.smooth_meankp, c="red", label="smooth")
        ax[4].legend(loc="upper right", bbox_to_anchor=(1.5, 1))
        ax[4].fill_between(self.chs[i].Vp.time, self.chs[i].Vp.smooth_meankp - self.chs[i].Vp.error,
                         self.chs[i].Vp.smooth_meankp + self.chs[i].Vp.error, fc="lightgrey")
        ax[5].plot(self.chs[i].Vp.time, self.chs[i].Vp.Er, c="red")
        ax[6].errorbar(self.chs[i].RAY_OUT.time, self.chs[i].RAY_OUT.kp, self.chs[i].RAY_OUT.kp_error, c="black", ecolor="lightgrey")
        ax[6].axhline(self.chs[i].RAY_OUT.meankp, c="red", ls="--")
        ax[7].plot(self.chs[i].RAY_OUT.time, self.chs[i].RAY_OUT.rho, c="black")

        ax[7].set_xlabel("Time [s]")

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Amplitude\n"
                         "[a.u.]")
        ax[4].set_ylabel("V perp\n"
                         "[km/s]")
        ax[4].set_ylim(-5, 5)
        ax[5].set_ylabel("Er\n"
                         "[kV/m]")
        ax[6].set_ylabel("k perp\n"
                         "[cm-1]")
        ax[7].set_ylabel("reff/a99")
        ax[7].set_ylim(0.3, 1.2)

        plt.show()

    def plot_Iamp(self):

    def plot_Vp(self):

    def plot_Er(self):

    def plot_kp_rho(self):

