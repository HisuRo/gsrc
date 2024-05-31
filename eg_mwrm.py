"""
2024.2.20 19:10
by Tatsuhiro Nasu
"""

from nasu import get_eg, myEgdb
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

    def read_param(self):

        self.ech = get_eg.ech_v2(sn=self.sn, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])
        self.nb = get_eg.nb_alldev(sn=self.sn, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])
        self.wp = get_eg.wp(sn=self.sn, sub=self.sub, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])
        self.nel = get_eg.nel(sn=self.sn, sub=self.sub, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])

    def plot_ch(self, freq=27.7, tstart=False, tend=False):

        i = np.argmin(np.abs(np.asarray(self.freqs) - freq))
        title = f"mwrm_comb_R_#{self.sn}-{self.sub}_{freq:.3f}GHz"

        fig, ax = plt.subplots(11, sharex=True, num=title, figsize=(6, 10))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")
        ax[4].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.lowf, c="blue", label="3-30 kHz")
        ax[4].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[5].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.midf, c="green", label="30-150 kHz")
        ax[5].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[6].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.highf, c="red", label="150-490 kHz")
        ax[6].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.allf, c="black", label="20-490 kHz")
        ax[6].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[7].plot(self.chs[i].Vp.time, self.chs[i].Vp.raw_meankp, c="black", label="raw")
        ax[7].plot(self.chs[i].Vp.time, self.chs[i].Vp.smooth_meankp, c="red", label="smooth")
        ax[7].axhline(0, c="lightgrey", ls="--")
        ax[7].legend(loc="upper right", bbox_to_anchor=(1.5, 1))
        ax[7].fill_between(self.chs[i].Vp.time, self.chs[i].Vp.smooth_meankp - self.chs[i].Vp.error,
                         self.chs[i].Vp.smooth_meankp + self.chs[i].Vp.error, fc="lightgrey")
        ax[8].plot(self.chs[i].Vp.time, self.chs[i].Vp.Er, c="red")
        ax[8].axhline(0, c="lightgrey", ls="--")
        ax[9].errorbar(self.chs[i].RAY_OUT.time, self.chs[i].RAY_OUT.kp, self.chs[i].RAY_OUT.kp_error, c="black", ecolor="lightgrey")
        ax[9].axhline(self.chs[i].RAY_OUT.meankp, c="red", ls="--")
        ax[10].plot(self.chs[i].RAY_OUT.time, self.chs[i].RAY_OUT.rho, c="black")

        ax[10].set_xlabel("Time [s]")
        if tstart:
            ax[10].set_xlim(tstart, tend)
        else:
            ax[10].set_xlim(self.ch1.Iamp.time[0], self.ch1.Iamp.time[-1])

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
        ax[7].set_ylabel("V perp\n"
                         "[km/s]")
        ax[7].set_ylim(-5, 5)
        ax[8].set_ylabel("Er\n"
                         "[kV/m]")
        ax[9].set_ylabel("k perp\n"
                         "[cm-1]")
        ax[10].set_ylabel("reff/a99")
        ax[10].set_ylim(0.3, 1.2)

        plt.show()

    def plot_Iamp(self, tstart=False, tend=False, freq_range="midf", cmapname="tab10"):

        title = f"mwrm_comb_R_Iamp_#{self.sn}-{self.sub}_{freq_range}"

        fig, ax = plt.subplots(4+len(self.freqs), sharex=True, num=title, figsize=(6, 12))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        cmap = plt.get_cmap(name=cmapname)
        for i in range(len(self.freqs)):
            if freq_range == "lowf":
                ax[4+i].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.lowf, c=cmap(i), label=f"{self.freqs[i]:.3f} GHz")
            elif freq_range == "midf":
                ax[4+i].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.midf, c=cmap(i), label=f"{self.freqs[i]:.3f} GHz")
            elif freq_range == "highf":
                ax[4+i].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.highf, c=cmap(i), label=f"{self.freqs[i]:.3f} GHz")
            elif freq_range == "allf":
                ax[4 + i].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.allf, c=cmap(i), label=f"{self.freqs[i]:.3f} GHz")
            else:
                print("Wrong command for freq_range\n")
                continue

            ax[4+i].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
            ax[4].set_ylabel("Amplitude\n"
                             "[a.u.]")

        ax[4+len(self.freqs)-1].set_xlabel("Time [s]")
        if tstart:
            ax[4+len(self.freqs)-1].set_xlim(tstart, tend)
        else:
            ax[4+len(self.freqs)-1].set_xlim(self.ch1.Iamp.time[0], self.ch1.Iamp.time[-1])

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")

        plt.show()

    def plot_Vp(self, tstart=False, tend=False, cmapname="tab10"):

        title = f"mwrm_comb_R_Vp_#{self.sn}-{self.sub}"

        fig, ax = plt.subplots(4+len(self.freqs), sharex=True, num=title, figsize=(6, 12))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        cmap = plt.get_cmap(name=cmapname)
        for i in range(len(self.freqs)):
            ax[4+i].plot(self.chs[i].Vp.time, self.chs[i].Vp.raw_meankp, c="black", label="raw")
            ax[4+i].plot(self.chs[i].Vp.time, self.chs[i].Vp.smooth_meankp, c=cmap(i), label="smooth")
            ax[4+i].fill_between(self.chs[i].Vp.time, self.chs[i].Vp.smooth_meankp - self.chs[i].Vp.error,
                               self.chs[i].Vp.smooth_meankp + self.chs[i].Vp.error, fc="lightgrey")
            ax[4+i].axhline(0, c="lightgrey", ls="--")

            ax[4 + i].legend(loc="upper right", bbox_to_anchor=(1.5, 1))
            ax[4+i].set_ylabel(f"V perp\n"
                             f"{self.freqs[i]} GHz\n"
                             f"[km/s]")
            ax[4+i].set_ylim(-10, 10)

        ax[4 + len(self.freqs) - 1].set_xlabel("Time [s]")
        if tstart:
            ax[4 + len(self.freqs) - 1].set_xlim(tstart, tend)
        else:
            ax[4 + len(self.freqs) - 1].set_xlim(self.ch1.Vp.time[0], self.ch1.Vp.time[-1])

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")

        plt.show()

    def plot_Er(self, tstart=False, tend=False, cmapname="tab10"):

        title = f"mwrm_comb_R_Er_#{self.sn}-{self.sub}"

        fig, ax = plt.subplots(4 + len(self.freqs), sharex=True, num=title, figsize=(6, 12))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        cmap = plt.get_cmap(name=cmapname)
        for i in range(len(self.freqs)):
            ax[4 + i].plot(self.chs[i].Vp.time, self.chs[i].Vp.Er, c=cmap(i))
            ax[4 + i].axhline(0, c="lightgrey", ls="--")

            ax[4 + i].legend(loc="upper right", bbox_to_anchor=(1.5, 1))
            ax[4 + i].set_ylabel(f"Er\n"
                                 f"{self.freqs[i]} GHz\n"
                                 f"[kV/m]")

        ax[4 + len(self.freqs) - 1].set_xlabel("Time [s]")
        if tstart:
            ax[4 + len(self.freqs) - 1].set_xlim(tstart, tend)
        else:
            ax[4 + len(self.freqs) - 1].set_xlim(self.ch1.Vp.time[0], self.ch1.Vp.time[-1])

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")

        plt.show()

    def plot_kp_rho(self, tstart=False, tend=False, cmapname="tab10"):

        title = f"mwrm_comb_R_kp_rho_#{self.sn}-{self.sub}"

        fig, ax = plt.subplots(4 + 2, sharex=True, num=title, figsize=(6, 6))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        cmap = plt.get_cmap(name=cmapname)

        for i in range(len(self.freqs)):
            ax[4].errorbar(self.chs[i].RAY_OUT.time, self.chs[i].RAY_OUT.kp, self.chs[i].RAY_OUT.kp_error,
                           c=cmap(i), ecolor="lightgrey", label=f"{self.freqs[i]:.3f} GHz")
            ax[4].axhline(self.chs[i].RAY_OUT.meankp, c=cmap(i), ls="--")
            ax[5].plot(self.chs[i].RAY_OUT.time, self.chs[i].RAY_OUT.rho, c=cmap(i), label=f"{self.freqs[i]:.3f} GHz")
        ax[4].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        ax[5].set_xlabel("Time [s]")
        if tstart:
            ax[5].set_xlim(tstart, tend)
        else:
            ax[5].set_xlim(self.ch1.RAY_OUT.time[0], self.ch1.RAY_OUT.time[-1])

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")
        ax[4].set_ylabel("k perp\n"
                         "[cm-1]")
        ax[5].set_ylabel("reff/a99")
        ax[5].set_ylim(0.3, 1.2)


class comb_U:

    def __init__(self, sn=174070, sub=1):

        self.sn = sn
        self.sub = sub

        self.egnames = ["mwrm_comb_U_Iamp", "mwrm_comb_U_Vp", "mwrm_comb_U_RAY_OUT"]

        self.eg_Iamp = myEgdb.LoadEG(self.egnames[0], sn=sn, sub=sub, flg_remove=True)
        self.eg_Vp = myEgdb.LoadEG(self.egnames[1], sn=sn, sub=sub, flg_remove=True)
        self.eg_RAY_OUT = myEgdb.LoadEG(self.egnames[2], sn=sn, sub=sub, flg_remove=True)

        self.ch1 = struct()
        self.ch2 = struct()
        self.ch3 = struct()
        self.ch4 = struct()
        self.freqs = [45.0, 47.0, 51.0, 57.0]
        self.chs = [self.ch1, self.ch2, self.ch3, self.ch4]

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
            # self.chs[i].Vp.Er = self.eg_Vp.trace_of(name=f'Er related  {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])

            self.chs[i].RAY_OUT.time = self.eg_RAY_OUT.dims(dim=0)
            self.chs[i].RAY_OUT.meankp = self.eg_RAY_OUT.trace_of(name=f'mean k_perp  {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])[0]
            self.chs[i].RAY_OUT.kp = self.eg_RAY_OUT.trace_of(name=f'k_perp each time  {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].RAY_OUT.kp_error = self.eg_RAY_OUT.trace_of(name=f'delta k_perp error estimation  {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].RAY_OUT.reff = self.eg_RAY_OUT.trace_of(name=f'reff {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].RAY_OUT.a99 = self.eg_RAY_OUT.trace_of(name=f'a99 {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].RAY_OUT.rho = self.eg_RAY_OUT.trace_of(name=f'reff/a99 {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])
            self.chs[i].RAY_OUT.B = self.eg_RAY_OUT.trace_of(name=f'B  {self.chs[i].freq:0.3f} [GHz]', dim=0, other_idxs=[0])

    def read_heat(self):

        self.ech = get_eg.ech_v2(sn=self.sn, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])
        self.nb = get_eg.nb_alldev(sn=self.sn, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])
        self.wp = get_eg.wp(sn=self.sn, sub=self.sub, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])
        self.nel = get_eg.nel(sn=self.sn, sub=self.sub, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])

    def plot_ch(self, freq=45, tstart=False, tend=False):

        i = np.argmin(np.abs(np.asarray(self.freqs) - freq))
        title = f"mwrm_comb_R_#{self.sn}-{self.sub}_{freq:.3f}GHz"

        fig, ax = plt.subplots(11, sharex=True, num=title, figsize=(6, 10))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")
        ax[4].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.lowf, c="blue", label="3-30 kHz")
        ax[4].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[5].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.midf, c="green", label="30-150 kHz")
        ax[5].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[6].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.highf, c="red", label="150-490 kHz")
        ax[6].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.allf, c="black", label="20-490 kHz")
        ax[6].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[7].plot(self.chs[i].Vp.time, self.chs[i].Vp.raw_meankp, c="black", label="raw")
        ax[7].plot(self.chs[i].Vp.time, self.chs[i].Vp.smooth_meankp, c="red", label="smooth")
        ax[7].axhline(0, c="lightgrey", ls="--")
        ax[7].legend(loc="upper right", bbox_to_anchor=(1.5, 1))
        ax[7].fill_between(self.chs[i].Vp.time, self.chs[i].Vp.smooth_meankp - self.chs[i].Vp.error,
                         self.chs[i].Vp.smooth_meankp + self.chs[i].Vp.error, fc="lightgrey")
        # ax[8].plot(self.chs[i].Vp.time, self.chs[i].Vp.Er, c="red")
        # ax[8].axhline(0, c="lightgrey", ls="--")
        ax[9].errorbar(self.chs[i].RAY_OUT.time, self.chs[i].RAY_OUT.kp, self.chs[i].RAY_OUT.kp_error, c="black", ecolor="lightgrey")
        ax[9].axhline(self.chs[i].RAY_OUT.meankp, c="red", ls="--")
        ax[10].plot(self.chs[i].RAY_OUT.time, self.chs[i].RAY_OUT.rho, c="black")

        ax[10].set_xlabel("Time [s]")
        if tstart:
            ax[10].set_xlim(tstart, tend)
        else:
            ax[10].set_xlim(self.ch1.Iamp.time[0], self.ch1.Iamp.time[-1])

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
        ax[7].set_ylabel("V perp\n"
                         "[km/s]")
        ax[7].set_ylim(-5, 5)
        # ax[8].set_ylabel("Er\n"
        #                  "[kV/m]")
        ax[9].set_ylabel("k perp\n"
                         "[cm-1]")
        ax[10].set_ylabel("reff/a99")
        ax[10].set_ylim(0.3, 1.2)

        plt.show()

    def plot_Iamp(self, tstart=False, tend=False, freq_range="midf", cmapname="tab10"):

        title = f"mwrm_comb_R_Iamp_#{self.sn}-{self.sub}_{freq_range}"

        fig, ax = plt.subplots(4+len(self.freqs), sharex=True, num=title, figsize=(6, 12))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        cmap = plt.get_cmap(name=cmapname)
        for i in range(len(self.freqs)):
            if freq_range == "lowf":
                ax[4+i].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.lowf, c=cmap(i), label=f"{self.freqs[i]:.3f} GHz")
            elif freq_range == "midf":
                ax[4+i].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.midf, c=cmap(i), label=f"{self.freqs[i]:.3f} GHz")
            elif freq_range == "highf":
                ax[4+i].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.highf, c=cmap(i), label=f"{self.freqs[i]:.3f} GHz")
            elif freq_range == "allf":
                ax[4 + i].plot(self.chs[i].Iamp.time, self.chs[i].Iamp.allf, c=cmap(i), label=f"{self.freqs[i]:.3f} GHz")
            else:
                print("Wrong command for freq_range\n")
                continue

            ax[4+i].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
            ax[4].set_ylabel("Amplitude\n"
                             "[a.u.]")

        ax[4+len(self.freqs)-1].set_xlabel("Time [s]")
        if tstart:
            ax[4+len(self.freqs)-1].set_xlim(tstart, tend)
        else:
            ax[4+len(self.freqs)-1].set_xlim(self.ch1.Iamp.time[0], self.ch1.Iamp.time[-1])

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")

        plt.show()

    def plot_Vp(self, tstart=False, tend=False, cmapname="tab10"):

        title = f"mwrm_comb_R_Vp_#{self.sn}-{self.sub}"

        fig, ax = plt.subplots(4+len(self.freqs), sharex=True, num=title, figsize=(6, 12))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        cmap = plt.get_cmap(name=cmapname)
        for i in range(len(self.freqs)):
            ax[4+i].plot(self.chs[i].Vp.time, self.chs[i].Vp.raw_meankp, c="black", label="raw")
            ax[4+i].plot(self.chs[i].Vp.time, self.chs[i].Vp.smooth_meankp, c=cmap(i), label="smooth")
            ax[4+i].fill_between(self.chs[i].Vp.time, self.chs[i].Vp.smooth_meankp - self.chs[i].Vp.error,
                               self.chs[i].Vp.smooth_meankp + self.chs[i].Vp.error, fc="lightgrey")
            ax[4+i].axhline(0, c="lightgrey", ls="--")

            ax[4 + i].legend(loc="upper right", bbox_to_anchor=(1.5, 1))
            ax[4+i].set_ylabel(f"V perp\n"
                             f"{self.freqs[i]} GHz\n"
                             f"[km/s]")
            ax[4+i].set_ylim(-10, 10)

        ax[4 + len(self.freqs) - 1].set_xlabel("Time [s]")
        if tstart:
            ax[4 + len(self.freqs) - 1].set_xlim(tstart, tend)
        else:
            ax[4 + len(self.freqs) - 1].set_xlim(self.ch1.Vp.time[0], self.ch1.Vp.time[-1])

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")

        plt.show()

    def plot_Er(self, tstart=False, tend=False, cmapname="tab10"):

        title = f"mwrm_comb_R_Er_#{self.sn}-{self.sub}"

        fig, ax = plt.subplots(4 + len(self.freqs), sharex=True, num=title, figsize=(6, 12))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        cmap = plt.get_cmap(name=cmapname)
        for i in range(len(self.freqs)):
            ax[4 + i].plot(self.chs[i].Vp.time, self.chs[i].Vp.Er, c=cmap(i))
            ax[4 + i].axhline(0, c="lightgrey", ls="--")

            ax[4 + i].legend(loc="upper right", bbox_to_anchor=(1.5, 1))
            ax[4 + i].set_ylabel(f"Er\n"
                                 f"{self.freqs[i]} GHz\n"
                                 f"[kV/m]")

        ax[4 + len(self.freqs) - 1].set_xlabel("Time [s]")
        if tstart:
            ax[4 + len(self.freqs) - 1].set_xlim(tstart, tend)
        else:
            ax[4 + len(self.freqs) - 1].set_xlim(self.ch1.Vp.time[0], self.ch1.Vp.time[-1])

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")

        plt.show()

    def plot_kp_rho(self, tstart=False, tend=False, cmapname="tab10"):

        title = f"mwrm_comb_R_kp_rho_#{self.sn}-{self.sub}"

        fig, ax = plt.subplots(4 + 2, sharex=True, num=title, figsize=(6, 6))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        cmap = plt.get_cmap(name=cmapname)

        for i in range(len(self.freqs)):
            ax[4].errorbar(self.chs[i].RAY_OUT.time, self.chs[i].RAY_OUT.kp, self.chs[i].RAY_OUT.kp_error,
                           c=cmap(i), ecolor="lightgrey", label=f"{self.freqs[i]:.3f} GHz")
            ax[4].axhline(self.chs[i].RAY_OUT.meankp, c=cmap(i), ls="--")
            ax[5].plot(self.chs[i].RAY_OUT.time, self.chs[i].RAY_OUT.rho, c=cmap(i), label=f"{self.freqs[i]:.3f} GHz")
        ax[4].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        ax[5].set_xlabel("Time [s]")
        if tstart:
            ax[5].set_xlim(tstart, tend)
        else:
            ax[5].set_xlim(self.ch1.RAY_OUT.time[0], self.ch1.RAY_OUT.time[-1])

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")
        ax[4].set_ylabel("k perp\n"
                         "[cm-1]")
        ax[5].set_ylabel("reff/a99")
        ax[5].set_ylim(0.3, 1.2)


class highK:

    def __init__(self, sn=174070, sub=1):

        self.sn = sn
        self.sub = sub

        self.egnames = ["mwrm_highK_Iamp", "mwrm_highK_Iamp2", "mwrm_highK_Iamp3"]

        self.eg_ch1 = myEgdb.LoadEG(self.egnames[0], sn=sn, sub=sub, flg_remove=True)
        self.eg_ch2 = myEgdb.LoadEG(self.egnames[1], sn=sn, sub=sub, flg_remove=True)
        self.eg_ch3 = myEgdb.LoadEG(self.egnames[2], sn=sn, sub=sub, flg_remove=True)
        egs = [self.eg_ch1, self.eg_ch2, self.eg_ch3]

        self.ch1 = struct()
        self.ch2 = struct()
        self.ch3 = struct()
        self.chs = [self.ch1, self.ch2, self.ch3]

        for i in range(len(self.chs)):
            self.chs[i].Iamp = struct()
            self.chs[i].t = egs[i].dims(dim=0)
            self.chs[i].Iamp.lowf = egs[i].trace_of(name=f"Amplitude (3-30kHz)", dim=0, other_idxs=[0])
            self.chs[i].Iamp.midf = egs[i].trace_of(name=f"Amplitude (30-150kHz)", dim=0, other_idxs=[0])
            self.chs[i].Iamp.highf = egs[i].trace_of(name=f"Amplitude (100-490kHz)", dim=0, other_idxs=[0])
            self.chs[i].Iamp.midf2 = egs[i].trace_of(name=f"Amplitude (20-200kHz)", dim=0, other_idxs=[0])
            self.chs[i].Iamp.highf2 = egs[i].trace_of(name=f"Amplitude (200-500kHz)", dim=0, other_idxs=[0])

            self.chs[i].rho = egs[i].trace_of(name=f'reff/a99', dim=0, other_idxs=[0])
            self.chs[i].a99 = egs[i].trace_of(name=f'a99', dim=0, other_idxs=[0])
            self.chs[i].fq = egs[i].trace_of(name="\\fq", dim=0, other_idxs=[0])
            self.chs[i].kp = egs[i].trace_of(name=f'wavenumber', dim=0, other_idxs=[0])
            self.chs[i].mod = egs[i].trace_of(name="modulation signal", dim=0, other_idxs=[0])

    def read_param(self):

        self.ech = get_eg.ech_v2(sn=self.sn, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])
        self.nb = get_eg.nb_alldev(sn=self.sn, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])
        self.wp = get_eg.wp(sn=self.sn, sub=self.sub, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])
        self.nel = get_eg.nel(sn=self.sn, sub=self.sub, tstart=self.ch1.Iamp.time[0], tend=self.ch1.Iamp.time[-1])

    def plot_ch(self, ch=1, tstart=False, tend=False):

        title = f"mwrm_highK_#{self.sn}-{self.sub}_ch{ch:d}"

        i = ch - 1
        fig, ax = plt.subplots(12, sharex=True, num=title, figsize=(6, 10))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        ax[4].plot(self.chs[i].t, self.chs[i].Iamp.lowf, c="blue", label="3-30 kHz")
        ax[4].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[5].plot(self.chs[i].t, self.chs[i].Iamp.midf, c="green", label="30-150 kHz")
        ax[5].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[6].plot(self.chs[i].t, self.chs[i].Iamp.highf, c="red", label="100-490 kHz")
        ax[6].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[7].plot(self.chs[i].t, self.chs[i].Iamp.midf2, c="green", label="20-200 kHz")
        ax[7].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[8].plot(self.chs[i].t, self.chs[i].Iamp.highf2, c="red", label="200-500 kHz")
        ax[8].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[9].plot(self.chs[i].t, self.chs[i].mod, c="black")

        ax[10].plot(self.chs[i].t, self.chs[i].kp, c="black")
        ax[11].plot(self.chs[i].t, self.chs[i].rho, c="black")

        ax[11].set_xlabel("Time [s]")
        if tstart:
            ax[11].set_xlim(tstart, tend)
        else:
            ax[11].set_xlim(self.chs[i].t[0], self.chs[i].t[-1])

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
        ax[9].set_ylabel("mod\n"
                         "[V]")
        ax[10].set_ylabel("k perp\n"
                         "[cm-1]")
        ax[11].set_ylabel("reff/a99")
        ax[11].set_ylim(0.3, 1.2)

        plt.show()

    def plot_Iamp(self, tstart=False, tend=False, freq_range="midf", cmapname="tab10"):

        title = f"mwrm_highK_Iamp_#{self.sn}-{self.sub}_{freq_range}"

        fig, ax = plt.subplots(4+len(self.chs), sharex=True, num=title, figsize=(6, 12))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        cmap = plt.get_cmap(name=cmapname)
        for i in range(len(self.chs)):
            if freq_range == "lowf":
                ax[4+i].plot(self.chs[i].t, self.chs[i].Iamp.lowf, c=cmap(i), label=f"ch{i+1:d}")
            elif freq_range == "midf":
                ax[4+i].plot(self.chs[i].t, self.chs[i].Iamp.midf, c=cmap(i), label=f"ch{i+1:d}")
            elif freq_range == "highf":
                ax[4+i].plot(self.chs[i].t, self.chs[i].Iamp.highf, c=cmap(i), label=f"ch{i+1:d}")
            elif freq_range == "midf2":
                ax[4 + i].plot(self.chs[i].t, self.chs[i].Iamp.midf2, c=cmap(i), label=f"ch{i+1:d}")
            elif freq_range == "highf2":
                ax[4 + i].plot(self.chs[i].t, self.chs[i].Iamp.highf2, c=cmap(i), label=f"ch{i+1:d}")
            else:
                print("Wrong command for freq_range\n")
                continue

            ax[4+i].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
            ax[4].set_ylabel("Amplitude\n"
                             "[a.u.]")

        ax[4+len(self.chs)-1].set_xlabel("Time [s]")
        if tstart:
            ax[4+len(self.chs)-1].set_xlim(tstart, tend)
        else:
            ax[4+len(self.chs)-1].set_xlim(self.ch1.t[0], self.ch1.t[-1])

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")

        plt.show()

    def plot_kp_rho(self, tstart=False, tend=False, cmapname="tab10"):

        title = f"mwrm_highK_kp_rho_#{self.sn}-{self.sub}"

        fig, ax = plt.subplots(4 + 2, sharex=True, num=title, figsize=(6, 6))
        fig.suptitle(title)
        fig.subplots_adjust(left=0.2, right=0.7)

        ax[0].plot(self.ech.time, self.ech.total, c="black")
        ax[1].plot(self.nb.time, self.nb.perp, c="lightblue", label="Perp.")
        ax[1].plot(self.nb.time, self.nb.tang, c="purple", label="Tang.")
        ax[1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
        ax[2].plot(self.nel.time, self.nel.nebar, c="black")
        ax[3].plot(self.wp.time, self.wp.wp, c="black")

        cmap = plt.get_cmap(name=cmapname)

        for i in range(len(self.chs)):
            ax[4].plot(self.chs[i].t, self.chs[i].kp, self.chs[i], c=cmap(i), label=f"ch{i+1}")
            ax[5].plot(self.chs[i].t, self.chs[i].rho, c=cmap(i), label=f"ch{i+1}")
        ax[4].legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        ax[5].set_xlabel("Time [s]")
        if tstart:
            ax[5].set_xlim(tstart, tend)
        else:
            ax[5].set_xlim(self.ch1.t[0], self.ch1.t[-1])

        ax[0].set_ylabel("ECH Power\n"
                         "[MW]")
        ax[1].set_ylabel("NBI Power\n"
                         "[MW]")
        ax[2].set_ylabel("nebar\n"
                         "[E19m-3]")
        ax[3].set_ylabel("Wp\n[kJ]")
        ax[4].set_ylabel("k perp\n"
                         "[cm-1]")
        ax[5].set_ylabel("reff/a99")
        ax[5].set_ylim(0.3, 1.2)

# class R_9o:


# class V_9o:


# class comb_R_9o:
