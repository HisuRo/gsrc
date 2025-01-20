import numpy as np
from scipy.interpolate import interp1d  # type: ignore
from scipy.interpolate import griddata  # type: ignore
import seaborn as sns  # type: ignore
import os
import gc
import copy

import matplotlib.pyplot as plt  # type: ignore
from mpl_toolkits import mplot3d  # type: ignore
from matplotlib import cm  # type: ignore

from nasu.myEgdb import LoadEG
from nasu import read, calc, getShotInfo, proc, myEgdb, plot, const, system
import inspect
from nasu.tR import *

# ================================================================================================================================

class struct:
    pass


def Iamp_highk(diaghk, Iamp_range, sn, tstart, tend):

    eghk = LoadEG(diaghk, sn)
    print('\n')

    hktime = eghk.dims(0)
    idx_time = np.where((hktime >= tstart) & (hktime <= tend))[0]
    hktime = hktime[idx_time]

    Iamphk = eghk.trace_of(Iamp_range, dim=0, other_idxs=[0])
    print(Iamphk)
    Iamphk = Iamphk[idx_time]

    modhk = eghk.trace_of('modulation signal', dim=0, other_idxs=[0])
    modhk = modhk[idx_time]

    use_idx = np.where(modhk < -0.5)
    hktime_use = hktime[use_idx]
    Iamphk_use = Iamphk[use_idx]

    del eghk, hktime, idx_time, use_idx, Iamphk
    gc.collect()

    return hktime_use, Iamphk_use


def rho_highk(eghk, tstart, tend):

    hktime = eghk.dims(0)
    idx_time = np.where((hktime >= tstart) & (hktime <= tend))
    hktime = hktime[idx_time]

    rhohk = eghk.trace_of('reff/a99', dim=0, other_idxs=[0])
    rhohk = rhohk[idx_time]

    rhohk_ave = rhohk.mean()
    rhohk_std = rhohk.std()

    return hktime, rhohk, rhohk_ave, rhohk_std


def Iamp_comb_R(diagcombR, freq, Iamp_range, sn, tstart, tend):  # Only mwrm_comb_R_Iamp is available from 2022/1/18.

    egcombR = LoadEG(diagcombR, sn)
    print('\n')

    valnm = Iamp_range + '  ' + freq

    combRtime = egcombR.dims(0)
    idx_time = np.where((combRtime >= tstart) & (combRtime <= tend))
    combRtime = combRtime[idx_time]

    IampcombR = egcombR.trace_of(valnm, dim=0, other_idxs=[0])
    IampcombR = IampcombR[idx_time]

    return combRtime, IampcombR

# Only mwrm_comb_R_Vp is available since 2022/1/18.
# made cwrm_comb_R_RAYOUT available but mwrm_comb_R_Vp unavailable since 2022/6/23
def rho_comb_R(egcombR, freq, tstart, tend):

    rhonm = 'reff/a99 ' + freq

    combRtime = egcombR.dims(0)
    idx_time = np.where((combRtime >= tstart) & (combRtime <= tend))
    combRtime = combRtime[idx_time]

    rhocombR = egcombR.trace_of(rhonm, dim=0, other_idxs=[0])
    rhocombR = rhocombR[idx_time]

    rhocombR_avg = rhocombR.mean()
    rhocombR_std = rhocombR.std()

    return combRtime, rhocombR, rhocombR_avg, rhocombR_std


def wavenumber_comb(egcomb, freq, tstart, tend):

    wavenm = 'k_perp each time  ' + freq
    dwavenm = 'delta k_perp error estimation  ' + freq

    combtime = egcomb.dims(0)
    idx_time = np.where((combtime >= tstart) & (combtime <= tend))
    combtime = combtime[idx_time]

    wavenum_comb = egcomb.trace_of(wavenm, dim=0, other_idxs=[0])
    wavenum_comb = wavenum_comb[idx_time]
    dwavenum_comb = egcomb.trace_of(dwavenm, dim=0, other_idxs=[0])
    dwavenum_comb = dwavenum_comb[idx_time]

    wavenum_comb_avg = np.average(wavenum_comb)
    wavenum_comb_err = np.sqrt(np.sum((dwavenum_comb / 2)**2 +
                                      (wavenum_comb - wavenum_comb_avg)**2)/(len(wavenum_comb) - 1))

    return combtime, wavenum_comb, dwavenum_comb, wavenum_comb_avg, wavenum_comb_err


def Vp_comb_R(egcombR, freq, tstart, tend):

    valnm = 'V perp (smoothing)' + '  ' + freq + '(smoothing) mean k_perp'

    combRtime = egcombR.dims(0)
    idx_time = np.where((combRtime >= tstart) & (combRtime <= tend))
    combRtime = combRtime[idx_time]

    VpcombR = egcombR.trace_of(valnm, dim=0, other_idxs=[0])
    VpcombR = VpcombR[idx_time]

    VpcombR_avg = VpcombR.mean()
    VpcombR_std = VpcombR.std()

    return combRtime, VpcombR, VpcombR_avg, VpcombR_std


def nb(egnb, valnm, tstart, tend):

    nbtime = egnb.dims(0)
    idx_time = np.where((nbtime >= tstart) & (nbtime <= tend))
    nbtime = nbtime[idx_time]

    valnb = egnb.trace_of(valnm, dim=0, other_idxs=[0])
    valnb = valnb[idx_time]

    return nbtime, valnb

def nb_all(egnb, valnm):

    nbtime = egnb.dims(0)
    valnb = egnb.trace_of(valnm, dim=0, other_idxs=[0])

    return nbtime, valnb

def nb_alldev(sn=174070, sub=1, tstart=3.0, tend=6.0):

    o = struct()

    egnb = LoadEG('nb1pwr_temporal', sn=sn, sub=sub)
    o.time, o.nb1 = nb(egnb, 'Pport-through_nb1', tstart, tend)

    egnb = LoadEG('nb2pwr_temporal', sn=sn, sub=sub)
    _, o.nb2 = nb(egnb, 'Pport-through_nb2', tstart, tend)

    egnb = LoadEG('nb3pwr_temporal', sn=sn, sub=sub)
    _, o.nb3 = nb(egnb, 'Pport-through_nb3', tstart, tend)

    egnb = LoadEG('nb4apwr_temporal', sn=sn, sub=sub)
    _, o.nb4a = nb(egnb, 'Pport-through_nb4a', tstart, tend)

    egnb = LoadEG('nb4bpwr_temporal', sn=sn, sub=sub)
    _, o.nb4b = nb(egnb, 'Pport-through_nb4b', tstart, tend)

    egnb = LoadEG('nb5apwr_temporal', sn=sn, sub=sub)
    _, o.nb5a = nb(egnb, 'Pport-through_nb5a', tstart, tend)

    egnb = LoadEG('nb5bpwr_temporal', sn=sn, sub=sub)
    _, o.nb5b = nb(egnb, 'Pport-through_nb5b', tstart, tend)

    o.tang = o.nb1 + o.nb2 + o.nb3
    o.perp = o.nb4a + o.nb4b + o.nb5a + o.nb5b

    return o

def ech(sn, tstart, tend):

    diagech = 'echpw'

    ech_tot = 'Total ECH'

    egech = LoadEG(diagech, sn)

    echtime = egech.dims(0)
    idx_time = np.where((echtime >= tstart) & (echtime <= tend))
    echtime = echtime[idx_time]

    echpw = egech.trace_of(ech_tot, dim=0, other_idxs=[0])
    echpw = echpw[idx_time]

    return echtime, echpw

def ech_all(sn):

    diagech = 'echpw'
    ech_77G_55Uout = '77G_5.5Uout'
    ech_77G_2Our = '77G_2Our'
    ech_154G_2Oll = '154G_2Oll'
    ech_154G_2Oul = '154G_2Oul'
    ech_154G_2Olr = '154G_2Olr'
    ech_tot = 'Total ECH'

    egech = LoadEG(diagech, sn)
    echtime = egech.dims(0)
    ech_77G_55UOut_pw = egech.trace_of(ech_77G_55Uout, dim=0, other_idxs=[0])
    ech_77G_2Our_pw = egech.trace_of(ech_77G_2Our, dim=0, other_idxs=[0])
    ech_154G_2Oll_pw = egech.trace_of(ech_154G_2Oll, dim=0, other_idxs=[0])
    ech_154G_2Oul_pw = egech.trace_of(ech_154G_2Oul, dim=0, other_idxs=[0])
    ech_154G_2Olr_pw = egech.trace_of(ech_154G_2Olr, dim=0, other_idxs=[0])
    ech_tot_pw = egech.trace_of(ech_tot, dim=0, other_idxs=[0])

    return echtime, ech_77G_55UOut_pw, ech_77G_2Our_pw, ech_154G_2Oll_pw, ech_154G_2Oul_pw, ech_154G_2Olr_pw, ech_tot_pw

def ech_v2(sn, tstart, tend, decimate=10):

    egech = LoadEG('echpw', sn)

    o = struct()
    tall = egech.dims(0)
    idx_ts = np.argmin(np.abs(tall - tstart))
    idx_te = np.argmin(np.abs(tall - tend))

    o.time = tall[idx_ts: idx_te+1: decimate]
    o.port55UOut_77G = egech.trace_of('77G_5.5Uout', dim=0, other_idxs=[0])[idx_ts: idx_te+1: decimate]
    o.port2Out_77G = egech.trace_of('77G_2Our', dim=0, other_idxs=[0])[idx_ts: idx_te+1: decimate]
    o.port2OLL_154G = egech.trace_of('154G_2Oll', dim=0, other_idxs=[0])[idx_ts: idx_te+1: decimate]
    o.port2OUL_154G = egech.trace_of('154G_2Oul', dim=0, other_idxs=[0])[idx_ts: idx_te+1: decimate]
    o.port2OLR_154G = egech.trace_of('154G_2Olr', dim=0, other_idxs=[0])[idx_ts: idx_te+1: decimate]
    o.total = egech.trace_of('Total ECH', dim=0, other_idxs=[0])[idx_ts: idx_te+1: decimate]

    return o

def wp(sn=174070, sub=1, tstart=3.0, tend=6.0, decimate=20):

    egwp = LoadEG("wp", sn=sn, sub=sub)
    tall = egwp.dims(dim=0)
    idx_ts = np.argmin(np.abs(tall - tstart))
    idx_te = np.argmin(np.abs(tall - tend))

    o = struct()
    o.time = tall[idx_ts: idx_te+1: decimate]
    o.wp = egwp.trace_of(name="Wp", dim=0, other_idxs=[0])[idx_ts: idx_te+1: decimate]
    o.beta_dia = egwp.trace_of(name="<beta-dia>", dim=0, other_idxs=[0])[idx_ts: idx_te+1: decimate]
    o.beta_vmec = egwp.trace_of(name="<beta-vmec>", dim=0, other_idxs=[0])[idx_ts: idx_te+1: decimate]

    return o

class diamag:

    def __init__(self, sn=174070, sub=1, tstart=3.0, tend=6.0, decimate=20):

        self.diagname = "wp"
        self.sn = sn
        self.sub = sub
        self.ts = tstart
        self.te = tend
        self.dec = decimate
        self.t, list_dat, self.list_dimnms, self.list_valnms, self.list_dimunits, self.list_valunits \
            = read.eg1d(diagnm=self.diagname, sn=self.sn, sub=self.sub)

        _idx, list_dat = proc.getTimeIdxsAndDats(self.t, self.ts, self.te, list_dat, decimate=self.dec)
        self.t = self.t[_idx]
        self.wp, self.beta_dia, self.beta_vmec = list_dat

def nel(sn=174070, sub=1, tstart=3.0, tend=6.0, decimate=10):

    egnel = LoadEG(diagname="fir_nel", sn=sn, sub=sub)
    tall = egnel.dims(dim=0)
    idx_ts = np.argmin(np.abs(tall - tstart))
    idx_te = np.argmin(np.abs(tall - tend))

    o = struct()
    o.time = tall[idx_ts: idx_te + 1: decimate]
    o.nebar = egnel.trace_of(name="ne_bar(.*)", dim=0, other_idxs=[0], including_wildcard=True)[idx_ts: idx_te + 1: decimate]
    o.peak = egnel.trace_of(name="peak", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl3309 = egnel.trace_of(name="nL(3309)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl3399 = egnel.trace_of(name="nL(3399)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl3489 = egnel.trace_of(name="nL(3489)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl3579 = egnel.trace_of(name="nL(3579)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl3669 = egnel.trace_of(name="nL(3669)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl3759 = egnel.trace_of(name="nL(3759)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl3849 = egnel.trace_of(name="nL(3849)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl3939 = egnel.trace_of(name="nL(3939)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl4029 = egnel.trace_of(name="nL(4029)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl4119 = egnel.trace_of(name="nL(4119)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl4209 = egnel.trace_of(name="nL(4209)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl4299 = egnel.trace_of(name="nL(4299)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]
    o.nl4389 = egnel.trace_of(name="nL(4389)", dim=0, other_idxs=[0])[idx_ts: idx_te + 1: decimate]

    return o

class fir_nel:

    def __init__(self, sn=174070, sub=1, tstart=3.0, tend=6.0):

        self.sn = sn
        self.sub = sub
        self.tstart = tstart
        self.tend = tend
        self.inputs_init = {
            "sn" : sn, 
            "subsn" : sub, 
            "tstart" : tstart, 
            "tend" : tend
        }

        self.t, list_dat, self.list_dimnms, self.list_valnms, self.list_dimunits, self.list_valunits \
            = read.eg1d(diagnm="fir_nel", sn=self.sn, sub=self.sub)

        tidxs, list_dat = proc.getTimeIdxsAndDats(self.t, self.tstart, self.tend, list_dat)
        self.t = self.t[tidxs]

        self.nebar, self.peak, self.nl3309, self.nl3399, self.nl3489, self.nl3579, self.nl3669, self.nl3759, \
        self.nl3849, self.nl3939, self.nl4029, self.nl4119, self.nl4209, self.nl4299, self.nl4389 = list_dat

        self.outdir = "C:/python_data/eg/fir_nel"
        proc.ifNotMake(self.outdir)

    def plot_sightline(self, tat=4.5, diag = 'tsmesh', type = "dat", pause=0):

        inputs = copy.deepcopy(self.inputs_init)
        inputs["tat"] = tat

        phi_at = 0.

        eg_tsmesh = myEgdb.LoadEGSimple(diag, self.sn, type=type)

        times = eg_tsmesh.dims(0)
        Rs = eg_tsmesh.dims(1)
        Zs = eg_tsmesh.dims(2)
        phis = eg_tsmesh.dims(3)

        idx_t = np.argmin(np.abs(times - tat))
        idx_phi = np.argmin(np.abs(phis - phi_at))

        reffs = np.zeros((len(Zs), len(Rs)))
        for ii, Z in enumerate(Zs):
            reffs[ii] = eg_tsmesh.trace_of('reff', 1, [idx_t, ii, idx_phi])

        grid_Rs, grid_Zs = np.meshgrid(Rs, Zs)

        # output setting
        figdir = os.path.join(self.outdir, "sightline")
        proc.ifNotMake(figdir)
        fname = f"{self.sn}_{self.sub}_{tat}"
        title = f"#{self.sn}-{self.sub} {tat} s"
        inputs["output_filename"] = fname

        script_path = os.path.abspath(__file__)
        class_name = self.__class__.__name__
        func_name = inspect.currentframe().f_code.co_name
        tmpdir, outdir, logs, now = system.initial_setting_in_nasumodule(script_path, class_name, func_name, outdir_name=figdir)

        fig, ax = plt.subplots(1, num=fname, figsize=(4, 8))
        cs = plt.contour(grid_Rs, grid_Zs, reffs, levels=[0.1 * i - 1. for i in range(21)],
                          cmap=cm.get_cmap('cool'))
        R_fir = [3.309, 3.399, 3.489, 3.579, 3.669, 3.759, 3.849, 3.939, 4.029, 4.119, 4.209, 4.299, 4.389]
        ax.vlines(R_fir, Zs.min(), Zs.max(), ls="--", colors="black")
        plt.axhline(0, ls="--", c="grey")
        plt.clabel(cs, inline=True, fontsize=10, fmt='%1.2f')

        ax.set_xlim(3.2, 4.5)
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title("reff [m]")

        plot.caption(fig, title)
        # plot.capsave(fig, title, fname, path)

        # output # EDIT HERE !!
        outputs = {
            'fig': fig, 
            'Rs': Rs, 
            'Zs': Zs, 
            'reffs': reffs, 
            'R_fir': R_fir
        }

        # systematic output and close
        output_filepath = system.output_pickle_file(outputs, inputs, logs, outdir)
        system.output_fig(fig, outdir, output_filepath, now)

        # plot.check(pause)

        print("DONE !!")

    def ref_to(self, time_ref):

        self.ref = struct()
        self.ref.t = time_ref

        self.ref.avg = struct()
        self.ref.std = struct()
        self.ref.ste = struct()

        list_dat = [self.nebar, self.peak, self.nl3309, self.nl3399, self.nl3489, self.nl3579, self.nl3669, self.nl3759,
                    self.nl3849, self.nl3939, self.nl4029, self.nl4119, self.nl4209, self.nl4299, self.nl4389]
        datRefList, stdRefList, errRefList = calc.timeAverageDatListByRefs(self.t, list_dat, time_ref)

        self.ref.avg.nebar, self.ref.avg.peak, self.ref.avg.nl3309, self.ref.avg.nl3399, self.ref.avg.nl3489, \
        self.ref.avg.nl3579, self.ref.avg.nl3669, self.ref.avg.nl3759, self.ref.avg.nl3849, self.ref.avg.nl3939, \
        self.ref.avg.nl4029, self.ref.avg.nl4119, self.ref.avg.nl4209, self.ref.avg.nl4299, self.ref.avg.nl4389 \
            = datRefList
        self.ref.std.nebar, self.ref.std.peak, self.ref.std.nl3309, self.ref.std.nl3399, self.ref.std.nl3489, \
        self.ref.std.nl3579, self.ref.std.nl3669, self.ref.std.nl3759, self.ref.std.nl3849, self.ref.std.nl3939, \
        self.ref.std.nl4029, self.ref.std.nl4119, self.ref.std.nl4209, self.ref.std.nl4299, self.ref.std.nl4389 \
            = stdRefList
        self.ref.ste.nebar, self.ref.ste.peak, self.ref.ste.nl3309, self.ref.ste.nl3399, self.ref.ste.nl3489, \
        self.ref.ste.nl3579, self.ref.ste.nl3669, self.ref.ste.nl3759, self.ref.ste.nl3849, self.ref.ste.nl3939, \
        self.ref.ste.nl4029, self.ref.ste.nl4119, self.ref.ste.nl4209, self.ref.ste.nl4299, self.ref.ste.nl4389 \
            = errRefList

        return self.ref

class lhdgauss_ray_mwrm:

    def __init__(self, sn=184508, sub=1, num=7):

        self.sn = sn
        self.sub = sub
        self.num = num
        self.diagname = f"LHDGAUSS_RAY_MWRM{self.num:d}"
        self.t, self.f, self.ray, self.step, list_dat4d, \
        self.dimnms, self.valnms, self.dimunits, self.valunits = read.eg4d(diagnm=self.diagname, sn=sn, sub=sub)
        self.x, self.y, self.z, self.kx, self.ky, self.kz, self.kperp, self.kpara, \
        self.reff, self.a99, self.rho, self.Te, self.ne, \
        self.B, self.Bx, self.By, self.Bz = list_dat4d

        self.bx = self.Bx / self.B
        self.by = self.By / self.B
        self.bz = self.Bz / self.B

        self.isExecuted_scattangle = False
        self.isExecuted_scattposition = False
        self.isExecuted_param_along_ray = False

    def plot_rays(self, time=4.5, freq=90, show_b=False):

        idx_t = np.nanargmin(np.abs(self.t - time))  # [s]
        idx_f = np.nanargmin(np.abs(self.f - freq))  # [GHz]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # プロット
        decimate=20
        for i, num in enumerate(self.ray):
            ax.plot(self.x[idx_t, idx_f, i], self.y[idx_t, idx_f, i], self.z[idx_t, idx_f, i], label=num)
            if show_b:
                ax.quiver(self.x[idx_t, idx_f, i][::decimate], self.y[idx_t, idx_f, i][::decimate], self.z[idx_t, idx_f, i][::decimate],
                          self.bx[idx_t, idx_f, i][::decimate], self.by[idx_t, idx_f, i][::decimate], self.bz[idx_t, idx_f, i][::decimate],
                          length=0.1, alpha=0.5, color="black", linewidth=1)

        # 軸ラベルの設定
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')

        ax.legend()

        plt.show()

    def scattangle(self, time=4.5, freq_GHz=90.):

        idx_t = np.nanargmin(np.abs(self.t - time))  # [s]
        idx_f = np.nanargmin(np.abs(self.f - freq_GHz))  # [GHz]
        idx_ray = 0

        self.disp = np.array([np.diff(self.x[idx_t, idx_f, idx_ray]),
                              np.diff(self.y[idx_t, idx_f, idx_ray]),
                              np.diff(self.z[idx_t, idx_f, idx_ray])]).T
        self.b = np.array([self.bx[idx_t, idx_f, idx_ray][:-1],
                           self.by[idx_t, idx_f, idx_ray][:-1],
                           self.bz[idx_t, idx_f, idx_ray][:-1]]).T
        nhor = np.array([0, 0, 1])

        # self.bp = calc.repeat_and_add_lastdim(np.sum(self.b * nhor, axis=-1), 3) * nhor
        # self.bt = self.b - self.bp
        #
        # self.ebt = self.bt / calc.repeat_and_add_lastdim(np.linalg.norm(self.bt, axis=-1), 3)
        # self.disp_t = calc.repeat_and_add_lastdim(np.sum(self.disp * self.ebt, axis=-1), 3) * self.ebt
        # self.disp_pr = self.disp - self.disp_t

        self.disp_p = calc.repeat_and_add_lastdim(np.sum(self.disp * nhor, axis=-1), 3) * nhor
        # self.disp_r = self.disp_pr - self.disp_p
        self.disp_r = self.disp - self.disp_p

        # self.theta_s = np.arccos(np.sum(- self.disp_pr / calc.repeat_and_add_lastdim(np.linalg.norm(self.disp_pr, axis=-1), 3) *
        #                            self.disp_r / calc.repeat_and_add_lastdim(np.linalg.norm(self.disp_r, axis=-1), 3), axis=-1))
        self.theta = np.arccos(
            np.sum(- self.disp / calc.repeat_and_add_lastdim(np.linalg.norm(self.disp, axis=-1), 3) *
                   self.disp_r / calc.repeat_and_add_lastdim(np.linalg.norm(self.disp_r, axis=-1), 3), axis=-1))

        idx_step = np.nanargmin(np.abs(self.z[idx_t, idx_f, idx_ray] - 0.))
        self.thetas = self.theta[idx_step]

        self.isExecuted_scattangle = True

    def scattposition(self, time=4.5, freq_GHz=90., idx_ray=0):

        idx_t = np.nanargmin(np.abs(self.t - time))
        idx_f = np.nanargmin(np.abs(self.f - freq_GHz))

        idx_step = np.nanargmin(np.abs(self.z[idx_t, idx_f, idx_ray] - 0.))

        self.xs = self.x[idx_t, idx_f, idx_ray, idx_step]
        self.ys = self.y[idx_t, idx_f, idx_ray, idx_step]
        self.zs = self.z[idx_t, idx_f, idx_ray, idx_step]
        self.kxs = self.kx[idx_t, idx_f, idx_ray, idx_step]
        self.kys = self.ky[idx_t, idx_f, idx_ray, idx_step]
        self.kzs = self.kz[idx_t, idx_f, idx_ray, idx_step]
        self.kperps = self.kperp[idx_t, idx_f, idx_ray, idx_step]
        self.kparas = self.kpara[idx_t, idx_f, idx_ray, idx_step]
        self.reffs = self.reff[idx_t, idx_f, idx_ray, idx_step]
        self.rhos = self.rho[idx_t, idx_f, idx_ray, idx_step]

        self.isExecuted_scattposition = True

    def param_along_ray(self, time=4.5, freq_GHz=90., ray_num=1):

        idx_t = np.nanargmin(np.abs(self.t - time))
        idx_f = np.nanargmin(np.abs(self.f - freq_GHz))
        idx_ray = ray_num - 1

        self.xray = self.x[idx_t, idx_f, idx_ray]
        self.yray = self.y[idx_t, idx_f, idx_ray]
        self.kxray = self.kx[idx_t, idx_f, idx_ray]
        self.kyray = self.ky[idx_t, idx_f, idx_ray]
        self.kzray = self.kz[idx_t, idx_f, idx_ray]
        self.kperpray = self.kperp[idx_t, idx_f, idx_ray]
        self.kpararay = self.kpara[idx_t, idx_f, idx_ray]
        self.reffray = self.reff[idx_t, idx_f, idx_ray]
        self.rhoray = self.rho[idx_t, idx_f, idx_ray]
        self.Teray = self.Te[idx_t, idx_f, idx_ray]
        self.neray = self.ne[idx_t, idx_f, idx_ray]
        self.Bray = self.B[idx_t, idx_f, idx_ray]
        self.Bxray = self.Bx[idx_t, idx_f, idx_ray]
        self.Byray = self.By[idx_t, idx_f, idx_ray]
        self.Bzray = self.Bz[idx_t, idx_f, idx_ray]

        self.isExecuted_param_along_ray = True

class tsmap:

    def __init__(self, sn=184508, subsn=1, d_colnm="Te", e_colnm="dTe", tstart=3., tend=6., rho_cut=1.2, include_outerside=True):

        self.sn = sn
        self.subsn = subsn
        self.d_colnm = d_colnm
        self.e_colnm = e_colnm
        self.tstart = tstart
        self.tend = tend
        self.rho_cut = rho_cut

        EG = LoadEG(diagname="tsmap_calib", sn=sn, sub=subsn)

        self.t_s = EG.dims(0)
        self.R_m = EG.dims(1)

        self.r_m = EG.trace_of_2d('reff', [0, 1])
        self.rho = EG.trace_of_2d('reff/a99', [0, 1])
        self.d = EG.trace_of_2d(self.d_colnm, [0, 1])
        if e_colnm is not None:
            self.e = EG.trace_of_2d(self.e_colnm, [0, 1])

        self.r_m = np.reshape(self.r_m, EG.dimsize)
        self.rho = np.reshape(self.rho, EG.dimsize)
        self.d = np.reshape(self.d, EG.dimsize)
        self.e = np.reshape(self.e, EG.dimsize)

        self.tR = tR(self.t_s, self.R_m, self.r_m, self.rho, self.d, self.e)
        self.tR.t_window(self.tstart, self.tend, include_outerside=include_outerside)

        self.tR.twin.rm_noisy_ch()
        self.tR.twin.rm_noisy_time()
        self.tR.twin.cut_by_rho(rho_cut=self.rho_cut)

        self.Bax, self.Rax, self.Bq, self.gamma, self.datetime, self.cycle = getShotInfo.info(self.sn)

        """
        # coef = np.abs(self.Bax / 3)
        # self.Br *= coef
        # self.Bz *= coef
        # self.Bphi *= coef

        # self.B = np.sqrt(self.Br ** 2 + self.Bz ** 2 + self.Bphi ** 2)

        # self.pe = const.ee * self.Te * self.ne_calFIR * 1e19  # [kPa]
        # self.pe_err = np.sqrt((self.dTe/self.Te)**2 + (self.dne_calFIR/self.ne_calFIR)**2) * self.pe
        # self.pe_fit = const.ee * self.Te_fit * self.ne_fit * 1e19  # [kPa]
        # self.pe_fit_err = np.sqrt((self.Te_fit_err / self.Te_fit) ** 2 +
        #                           (self.ne_fit_err / self.ne_fit) ** 2) * self.pe_fit

        # self.scaled_resistivity, self.scaled_resistivity_err = calc.power(self.Te, - 3./2., self.dTe)
        # self.scaled_resistivity_fit, self.scaled_resistivity_fit_err \
        #     = calc.power(self.Te_fit, - 3. / 2., self.Te_fit_err)
        """

    def plot_reffa99(self, Rmin=3.5, Rmax=4.7, rhomin=0, rhomax=1.2, drho=0.1):
        levels = np.arange(rhomin, rhomax+drho, drho)
        tg, Rg = np.meshgrid(self.t, self.R)
        fnm=f"#{self.sn}-{self.sub}_{self.tstart}-{self.tend}s_{Rmin}-{Rmax}m"
        plt.subplots(num=fnm)
        cp = plt.contour(tg, Rg, self.reffa99.T, levels=levels)
        plt.clabel(cp, inline=True, fontsize=10)
        plt.title('reff/a99')
        plt.xlabel('Time [s]')
        plt.ylabel('R [m]')
        plt.xlim(self.tstart, self.tend)
        plt.ylim(Rmin, Rmax)
        plt.show()

    def plot_reff(self, Rmin=3.5, Rmax=4.7, reffmin=0, reffmax=0.8, dreff=0.1):
        levels = np.arange(reffmin, reffmax+dreff, dreff)
        tg, Rg = np.meshgrid(self.t, self.R)
        fnm=f"#{self.sn}-{self.sub}_{self.tstart}-{self.tend}s_{Rmin}-{Rmax}m"
        plt.subplots(num=fnm)
        cp = plt.contour(tg, Rg, self.reff.T, levels=levels)
        plt.clabel(cp, inline=True, fontsize=10)
        plt.title('reff [m]')
        plt.xlabel('Time [s]')
        plt.ylabel('R [m]')
        plt.xlim(self.tstart, self.tend)
        plt.ylim(Rmin, Rmax)
        plt.show()

class cxsmap7:

    def __init__(self, sn=184508, sub=1, tstart=3, tend=6, reff_cut=0.7):

        self.sn = sn
        self.sub = sub
        self.ts = tstart
        self.te = tend
        self.reff_cut = reff_cut

        EG = LoadEG(diagname="cxsmap7", sn=sn, sub=sub)

        self.t = EG.dims(0)

        self.ary = EG.trace_of('ary', 1, [0])
        idxs_pol = np.where((self.ary == 1.) | (self.ary == 3.))[0]
        idxs_tor = np.where((self.ary == 5.) | (self.ary == 7.))[0]

        self.pol = struct()
        self.tor = struct()

        self.pol.R = EG.dims(1)[idxs_pol]
        self.tor.R = EG.dims(1)[idxs_tor]
        idxs_sort_pol = np.argsort(self.pol.R)
        idxs_sort_tor = np.argsort(self.tor.R)
        self.pol.R = self.pol.R[idxs_sort_pol]
        self.tor.R = self.tor.R[idxs_sort_tor]

        self.pol.Ti = EG.trace_of_2d('Ti', [0, 1])
        self.pol.Tier = EG.trace_of_2d('Tier', [0, 1])
        self.pol.Vc = EG.trace_of_2d('Vc', [0, 1])
        self.pol.Ver = EG.trace_of_2d('Ver', [0, 1])
        self.pol.inc = EG.trace_of_2d('inc', [0, 1])
        self.pol.icer = EG.trace_of_2d('icer', [0, 1])
        self.pol.Vr = EG.trace_of_2d('Vr', [0, 1])
        self.pol.reff = EG.trace_of_2d('reff', [0, 1])
        self.pol.a99 = EG.trace_of_2d('a99', [0, 1])
        self.pol.p0 = EG.trace_of_2d('p0', [0, 1])
        self.pol.pf = EG.trace_of_2d('pf', [0, 1])
        self.pol.ip = EG.trace_of_2d('ip', [0, 1])
        self.pol.ipf = EG.trace_of_2d('ipf', [0, 1])
        self.pol.Br = EG.trace_of_2d('Br', [0, 1])
        self.pol.Bz = EG.trace_of_2d('Bz', [0, 1])
        self.pol.Bphi = EG.trace_of_2d('Bphi', [0, 1])
        self.pol.dVdreff = EG.trace_of_2d('dVdreff', [0, 1])
        self.pol.Te = EG.trace_of_2d('Te', [0, 1])
        self.pol.ne = EG.trace_of_2d('ne', [0, 1])
        self.pol.t1 = EG.trace_of_2d('t1', [0, 1])

        self.tor.Ti = EG.trace_of_2d('Ti', [0, 1])
        self.tor.Tier = EG.trace_of_2d('Tier', [0, 1])
        self.tor.Vc = EG.trace_of_2d('Vc', [0, 1])
        self.tor.Ver = EG.trace_of_2d('Ver', [0, 1])
        self.tor.inc = EG.trace_of_2d('inc', [0, 1])
        self.tor.icer = EG.trace_of_2d('icer', [0, 1])
        self.tor.Vr = EG.trace_of_2d('Vr', [0, 1])
        self.tor.reff = EG.trace_of_2d('reff', [0, 1])
        self.tor.a99 = EG.trace_of_2d('a99', [0, 1])
        self.tor.p0 = EG.trace_of_2d('p0', [0, 1])
        self.tor.pf = EG.trace_of_2d('pf', [0, 1])
        self.tor.ip = EG.trace_of_2d('ip', [0, 1])
        self.tor.ipf = EG.trace_of_2d('ipf', [0, 1])
        self.tor.Br = EG.trace_of_2d('Br', [0, 1])
        self.tor.Bz = EG.trace_of_2d('Bz', [0, 1])
        self.tor.Bphi = EG.trace_of_2d('Bphi', [0, 1])
        self.tor.dVdreff = EG.trace_of_2d('dVdreff', [0, 1])
        self.tor.Te = EG.trace_of_2d('Te', [0, 1])
        self.tor.ne = EG.trace_of_2d('ne', [0, 1])
        self.tor.t1 = EG.trace_of_2d('t1', [0, 1])

        self.pol.Ti[self.pol.Ti == 0.] = np.nan
        self.pol.reff[np.abs(self.pol.reff) > self.reff_cut] = np.nan
        self.tor.reff[np.abs(self.tor.reff) > self.reff_cut] = np.nan

        varlist_pol = [self.pol.Ti, self.pol.Tier, self.pol.Vc, self.pol.Ver, self.pol.inc, self.pol.icer,
                       self.pol.Vr, self.pol.reff, self.pol.a99,
                       self.pol.p0, self.pol.pf, self.pol.ip, self.pol.ipf,
                       self.pol.Br, self.pol.Bz, self.pol.Bphi, self.pol.dVdreff,
                       self.pol.Te, self.pol.ne, self.pol.t1]
        varlist_tor = [self.tor.Ti, self.tor.Tier, self.tor.Vc, self.tor.Ver, self.tor.inc, self.tor.icer,
                       self.tor.Vr, self.tor.reff, self.tor.a99,
                       self.tor.p0, self.tor.pf, self.tor.ip, self.tor.ipf,
                       self.tor.Br, self.tor.Bz, self.tor.Bphi, self.tor.dVdreff,
                       self.tor.Te, self.tor.ne, self.tor.t1]

        for i in range(len(varlist_pol)):
            varlist_pol[i] = varlist_pol[i].reshape(EG.dimsize)
            varlist_pol[i] = varlist_pol[i][:, idxs_pol]
            varlist_pol[i] = varlist_pol[i][:, idxs_sort_pol]

        for i in range(len(varlist_tor)):
            varlist_tor[i] = varlist_tor[i].reshape(EG.dimsize)
            varlist_tor[i] = varlist_tor[i][:, idxs_tor]
            varlist_tor[i] = varlist_tor[i][:, idxs_sort_tor]

        tidxs = ~np.isnan(varlist_pol[0]).all(axis=1)
        self.t = self.t[tidxs]

        for i in range(len(varlist_pol)):
            varlist_pol[i] = varlist_pol[i][tidxs]
        for i in range(len(varlist_tor)):
            varlist_tor[i] = varlist_tor[i][tidxs]

        _, varlist_pol = proc.getTimeIdxsAndDats(self.t, self.ts, self.te, varlist_pol)
        tidxs, varlist_tor = proc.getTimeIdxsAndDats(self.t, self.ts, self.te, varlist_tor)
        self.t = self.t[tidxs]

        Rpolidxs = ~np.isnan(varlist_pol[7]).any(axis=0)
        Rtoridxs = ~np.isnan(varlist_tor[7]).any(axis=0)
        self.pol.R = self.pol.R[Rpolidxs]
        self.tor.R = self.tor.R[Rtoridxs]

        for i in range(len(varlist_pol)):
            varlist_pol[i] = varlist_pol[i][:, Rpolidxs]
        for i in range(len(varlist_tor)):
            varlist_tor[i] = varlist_tor[i][:, Rtoridxs]

        self.pol.Ti, self.pol.Tier, self.pol.Vc, self.pol.Ver, self.pol.inc, self.pol.icer, \
        self.pol.Vr, self.pol.reff, self.pol.a99, \
        self.pol.p0, self.pol.pf, self.pol.ip, self.pol.ipf, \
        self.pol.Br, self.pol.Bz, self.pol.Bphi, self.pol.dVdreff, \
        self.pol.Te, self.pol.ne, self.pol.t1 = varlist_pol
        self.tor.Ti, self.tor.Tier, self.tor.Vc, self.tor.Ver, self.tor.inc, self.tor.icer, \
        self.tor.Vr, self.tor.reff, self.tor.a99, \
        self.tor.p0, self.tor.pf, self.tor.ip, self.tor.ipf, \
        self.tor.Br, self.tor.Bz, self.tor.Bphi, self.tor.dVdreff, \
        self.tor.Te, self.tor.ne, self.tor.t1 = varlist_tor

        self.pol.reffa99 = self.pol.reff / self.pol.a99
        self.tor.reffa99 = self.tor.reff / self.tor.a99

        self.pol.a99 = self.pol.a99[:, 0]
        self.tor.a99 = self.tor.a99[:, 0]

        self.pol.rho_cut = reff_cut / self.pol.a99
        self.tor.rho_cut = reff_cut / self.tor.a99

        # self.pol.TeTi = self.pol.Te / self.pol.Ti
        # self.tor.TeTi = self.tor.Te / self.tor.Ti

        self.Bax, self.Rax, self.Bq, self.gamma, self.datetime, self.cycle = getShotInfo.info(self.sn)

    def calcgrad(self, polyN=10):

        # a. 2)
        self.polyN = polyN

        _o = calc.polyN_LSM_der(xx=self.tor.reff, yy=self.tor.Ti, polyN=polyN, yErr=self.tor.Tier, parity="even")
        self.tor.Ti_polyfit = _o.yHut
        self.tor.Ti_polyfit_err = _o.yHutErr
        self.tor.dTidreff_polyfit = _o.yHutDer
        self.tor.dTidreff_polyfit_err = _o.yHutDerErr
        self.tor.LTi_polyfit, self.tor.LTi_polyfit_err, self.tor.RLTi_polyfit, self.tor.RLTi_polyfit_err \
            = calc.Lscale(self.tor.Ti_polyfit, self.tor.dTidreff_polyfit, self.Rax,
                          self.tor.Ti_polyfit_err, self.tor.dTidreff_polyfit_err)

        _o = calc.polyN_LSM_der(xx=self.pol.reff, yy=self.pol.Ti, polyN=polyN, yErr=self.pol.Tier, parity="even")
        self.pol.Ti_polyfit = _o.yHut
        self.pol.Ti_polyfit_err = _o.yHutErr
        self.pol.dTidreff_polyfit = _o.yHutDer
        self.pol.dTidreff_polyfit_err = _o.yHutDerErr
        self.pol.LTi_polyfit, self.pol.LTi_polyfit_err, self.pol.RLTi_polyfit, self.pol.RLTi_polyfit_err \
            = calc.Lscale(self.pol.Ti_polyfit, self.pol.dTidreff_polyfit, self.Rax,
                          self.pol.Ti_polyfit_err, self.pol.dTidreff_polyfit_err)

        _o = calc.polyN_LSM_der(xx=self.tor.reff, yy=self.tor.Vc, polyN=polyN, yErr=self.tor.Ver)
        self.tor.Vc_polyfit = _o.yHut
        self.tor.Vc_polyfit_err = _o.yHutErr
        self.tor.dVcdreff_polyfit = _o.yHutDer
        self.tor.dVcdreff_polyfit_err = _o.yHutDerErr
        self.tor.LVc_polyfit, self.tor.LVc_polyfit_err, self.tor.RLVc_polyfit, self.tor.RLVc_polyfit_err \
            = calc.Lscale(self.tor.Vc_polyfit, self.tor.dVcdreff_polyfit, self.Rax,
                          self.tor.Vc_polyfit_err, self.tor.dVcdreff_polyfit_err)

        _o = calc.polyN_LSM_der(xx=self.pol.reff, yy=self.pol.Vc, polyN=polyN, yErr=self.pol.Ver)
        self.pol.Vc_polyfit = _o.yHut
        self.pol.Vc_polyfit_err = _o.yHutErr
        self.pol.dVcdreff_polyfit = _o.yHutDer
        self.pol.dVcdreff_polyfit_err = _o.yHutDerErr
        self.pol.LVc_polyfit, self.pol.LVc_polyfit_err, self.pol.RLVc_polyfit, self.pol.RLVc_polyfit_err \
            = calc.Lscale(self.pol.Vc_polyfit, self.pol.dVcdreff_polyfit, self.Rax,
                          self.pol.Vc_polyfit_err, self.pol.dVcdreff_polyfit_err)

        # # b. 1)
        # self.dreffdR = np.gradient(self.reff, self.R, edge_order=2, axis=-1)
        # self.dTedR_fit = np.gradient(self.Te_fit, self.R, edge_order=2, axis=-1)
        # self.dTedR_fit_err = np.abs(np.gradient(self.Te_fit_err, self.R, edge_order=2, axis=-1))
        # self.dTedreff_fit, self.dTedreff_fit_err = calc.dMdreff(self.dTedR_fit, self.dreffdR, self.dTedR_fit_err)
        # self.LTe_fit, self.LTe_fit_err, self.RLTe_fit, self.RLTe_fit_err \
        #     = calc.Lscale(self.Te_fit, self.dTedreff_fit, self.Rax, self.Te_fit_err, self.dTedreff_fit_err)
        #
        # self.dnedR_fit = np.gradient(self.ne_fit, self.R, edge_order=2, axis=-1)
        # self.dnedR_fit_err = np.abs(np.gradient(self.ne_fit_err, self.R, edge_order=2, axis=-1))
        # self.dnedreff_fit, self.dnedreff_fit_err = calc.dMdreff(self.dnedR_fit, self.dreffdR, self.dnedR_fit_err)
        # self.Lne_fit, self.Lne_fit_err, self.RLne_fit, self.RLne_fit_err \
        #     = calc.Lscale(self.ne_fit, self.dnedreff_fit, self.Rax, self.ne_fit_err, self.dnedreff_fit_err)
        #
        # self.dpedR_fit = np.abs(np.gradient(self.pe_fit, self.R, edge_order=2, axis=-1))
        # self.dpedR_fit_err = np.abs(np.gradient(self.pe_fit_err, self.R, edge_order=2, axis=-1))
        # self.dpedreff_fit, self.dpedreff_fit_err = calc.dMdreff(self.dpedR_fit, self.dreffdR, self.dpedR_fit_err)
        # self.Lpe_fit, self.Lpe_fit_err, self.RLpe_fit, self.RLpe_fit_err \
        #     = calc.Lscale(self.pe_fit, self.dpedreff_fit, self.Rax, self.pe_fit_err, self.dpedreff_fit_err)
        #
        # self.dBzdR = np.gradient(self.Bz, self.R, edge_order=2, axis=-1)
        # self.dBzdreff, _ = calc.dMdreff(self.dBzdR, self.dreffdR)
        # self.LBz, _, self.RLBz, _ \
        #     = calc.Lscale(self.Bz, self.dBzdreff, self.Rax)
        #
        # self.dBphidR = np.gradient(self.Bphi, self.R, edge_order=2, axis=-1)
        # self.dBphidreff, _ = calc.dMdreff(self.dBphidR, self.dreffdR)
        # self.LBphi, _, self.RLBphi, _ \
        #     = calc.Lscale(self.Bphi, self.dBphidreff, self.Rax)
        #
        # self.dBdR = np.gradient(self.B, self.R, edge_order=2, axis=-1)
        # self.dBdreff, _ = calc.dMdreff(self.dBdR, self.dreffdR)
        # self.LB, _, self.RLB, _ \
        #     = calc.Lscale(self.B, self.dBdreff, self.Rax)

    def tat(self, time=4.5, include_grad=True):

        self.at = struct()
        self.at.pol = struct()
        self.at.tor = struct()
        datlist = [self.t,
                   self.pol.Ti, self.pol.Tier, self.pol.Vc, self.pol.Ver, self.pol.inc, self.pol.icer,
                   self.pol.Vr, self.pol.reff, self.pol.a99,
                   self.pol.p0, self.pol.pf, self.pol.ip, self.pol.ipf,
                   self.pol.Br, self.pol.Bz, self.pol.Bphi, self.pol.dVdreff,
                   self.pol.Te, self.pol.ne, self.pol.t1,
                   self.pol.reffa99,
                   self.tor.Ti, self.tor.Tier, self.tor.Vc, self.tor.Ver, self.tor.inc, self.tor.icer,
                   self.tor.Vr, self.tor.reff, self.tor.a99,
                   self.tor.p0, self.tor.pf, self.tor.ip, self.tor.ipf,
                   self.tor.Br, self.tor.Bz, self.tor.Bphi, self.tor.dVdreff,
                   self.tor.Te, self.tor.ne, self.tor.t1,
                   self.tor.reffa99]

        _, datlist_at = proc.getTimeIdxAndDats(self.t, time, datlist)
        self.at.t, \
        self.at.pol.Ti, self.at.pol.Tier, self.at.pol.Vc, self.at.pol.Ver, self.at.pol.inc, self.at.pol.icer, \
        self.at.pol.Vr, self.at.pol.reff, self.at.pol.a99, \
        self.at.pol.p0, self.at.pol.pf, self.at.pol.ip, self.at.pol.ipf, \
        self.at.pol.Br, self.at.pol.Bz, self.at.pol.Bphi, self.at.pol.dVdreff, \
        self.at.pol.Te, self.at.pol.ne, self.at.pol.t1, \
        self.at.pol.reffa99, \
        self.at.tor.Ti, self.at.tor.Tier, self.at.tor.Vc, self.at.tor.Ver, self.at.tor.inc, self.at.tor.icer, \
        self.at.tor.Vr, self.at.tor.reff, self.at.tor.a99, \
        self.at.tor.p0, self.at.tor.pf, self.at.tor.ip, self.at.tor.ipf, \
        self.at.tor.Br, self.at.tor.Bz, self.at.tor.Bphi, self.at.tor.dVdreff, \
        self.at.tor.Te, self.at.tor.ne, self.at.tor.t1, \
        self.at.tor.reffa99 = datlist_at

        if include_grad:
            datlist = [self.pol.Ti_polyfit, self.pol.Ti_polyfit_err,
                       self.pol.dTidreff_polyfit, self.pol.dTidreff_polyfit_err,
                       self.pol.LTi_polyfit, self.pol.LTi_polyfit_err,
                       self.pol.RLTi_polyfit, self.pol.RLTi_polyfit_err,
                       self.pol.Vc_polyfit, self.pol.Vc_polyfit_err,
                       self.pol.dVcdreff_polyfit, self.pol.dVcdreff_polyfit_err,
                       self.pol.LVc_polyfit, self.pol.LVc_polyfit_err,
                       self.pol.RLVc_polyfit, self.pol.RLVc_polyfit_err,
                       self.tor.Ti_polyfit, self.tor.Ti_polyfit_err,
                       self.tor.dTidreff_polyfit, self.tor.dTidreff_polyfit_err,
                       self.tor.LTi_polyfit, self.tor.LTi_polyfit_err,
                       self.tor.RLTi_polyfit, self.tor.RLTi_polyfit_err,
                       self.tor.Vc_polyfit, self.tor.Vc_polyfit_err,
                       self.tor.dVcdreff_polyfit, self.tor.dVcdreff_polyfit_err,
                       self.tor.LVc_polyfit, self.tor.LVc_polyfit_err,
                       self.tor.RLVc_polyfit, self.tor.RLVc_polyfit_err
                       ]
            _, datlist_at = proc.getTimeIdxAndDats(self.t, time, datlist)
            self.at.pol.Ti_polyfit, self.at.pol.Ti_polyfit_err, \
            self.at.pol.dTidreff_polyfit, self.at.pol.dTidreff_polyfit_err, \
            self.at.pol.LTi_polyfit, self.at.pol.LTi_polyfit_err, \
            self.at.pol.RLTi_polyfit, self.at.pol.RLTi_polyfit_err, \
            self.at.pol.Vc_polyfit, self.at.pol.Vc_polyfit_err, \
            self.at.pol.dVcdreff_polyfit, self.at.pol.dVcdreff_polyfit_err, \
            self.at.pol.LVc_polyfit, self.at.pol.LVc_polyfit_err, \
            self.at.pol.RLVc_polyfit, self.at.pol.RLVc_polyfit_err, \
            self.at.tor.Ti_polyfit, self.at.tor.Ti_polyfit_err, \
            self.at.tor.dTidreff_polyfit, self.at.tor.dTidreff_polyfit_err, \
            self.at.tor.LTi_polyfit, self.at.tor.LTi_polyfit_err, \
            self.at.tor.RLTi_polyfit, self.at.tor.RLTi_polyfit_err, \
            self.at.tor.Vc_polyfit, self.at.tor.Vc_polyfit_err, \
            self.at.tor.dVcdreff_polyfit, self.at.tor.dVcdreff_polyfit_err, \
            self.at.tor.LVc_polyfit, self.at.tor.LVc_polyfit_err, \
            self.at.tor.RLVc_polyfit, self.at.tor.RLVc_polyfit_err = datlist_at

    def calc_teti(self, use_tsfit=True):

        self.tsmap = tsmap(self.sn, self.sub, self.ts, self.te, rho_cut=1.2)
        self.tsmap.calcgrad()

        self.tsmap.pol = struct()
        self.tsmap.tor = struct()

        if use_tsfit:
            self.tsmap.pol.Te_intp = griddata((np.repeat(self.tsmap.t, len(self.tsmap.R)), np.tile(self.tsmap.R, len(self.tsmap.t))),
                                              self.tsmap.Te_fit.ravel(), (self.t[:, None], self.pol.R[None, :]),
                                              method='linear', fill_value=np.nan)
            self.tsmap.pol.Te_intp_err = griddata((np.repeat(self.tsmap.t, len(self.tsmap.R)), np.tile(self.tsmap.R, len(self.tsmap.t))),
                                                  self.tsmap.Te_fit_err.ravel(), (self.t[:, None], self.pol.R[None, :]),
                                                  method='linear', fill_value=np.nan)
            self.tsmap.tor.Te_intp = griddata((np.repeat(self.tsmap.t, len(self.tsmap.R)), np.tile(self.tsmap.R, len(self.tsmap.t))),
                                              self.tsmap.Te_fit.ravel(), (self.t[:, None], self.tor.R[None, :]),
                                              method='linear', fill_value=np.nan)
            self.tsmap.tor.Te_intp_err = griddata((np.repeat(self.tsmap.t, len(self.tsmap.R)), np.tile(self.tsmap.R, len(self.tsmap.t))),
                                                  self.tsmap.Te_fit_err.ravel(), (self.t[:, None], self.tor.R[None, :]),
                                                  method='linear', fill_value=np.nan)
        else:
            self.tsmap.pol.Te_intp = griddata((np.repeat(self.tsmap.t, len(self.tsmap.R)), np.tile(self.tsmap.R, len(self.tsmap.t))),
                                              self.tsmap.Te_polyfit.ravel(), (self.t[:, None], self.pol.R[None, :]),
                                              method='linear', fill_value=np.nan)
            self.tsmap.pol.Te_intp_err = griddata((np.repeat(self.tsmap.t, len(self.tsmap.R)), np.tile(self.tsmap.R, len(self.tsmap.t))),
                                                  self.tsmap.Te_polyfit_err.ravel(), (self.t[:, None], self.pol.R[None, :]),
                                                  method='linear', fill_value=np.nan)
            self.tsmap.tor.Te_intp = griddata((np.repeat(self.tsmap.t, len(self.tsmap.R)), np.tile(self.tsmap.R, len(self.tsmap.t))),
                                              self.tsmap.Te_polyfit.ravel(), (self.t[:, None], self.tor.R[None, :]),
                                              method='linear', fill_value=np.nan)
            self.tsmap.tor.Te_intp_err = griddata((np.repeat(self.tsmap.t, len(self.tsmap.R)), np.tile(self.tsmap.R, len(self.tsmap.t))),
                                                  self.tsmap.Te_polyfit_err.ravel(), (self.t[:, None], self.tor.R[None, :]),
                                                  method='linear', fill_value=np.nan)
        self.pol.teti, self.pol.teti_err = calc.Tratio(self.tsmap.pol.Te_intp, self.pol.Ti,
                                                       self.tsmap.pol.Te_intp_err, self.pol.Tier)
        self.tor.teti, self.tor.teti_err = calc.Tratio(self.tsmap.tor.Te_intp, self.tor.Ti,
                                                       self.tsmap.tor.Te_intp_err, self.tor.Tier)

    def t_window(self, tstart=4.4, tend=4.5, include_grad=False, include_teti=False):

        self.twin = struct()
        self.twin.tstart = tstart
        self.twin.tend = tend
        self.twin.pol = struct()
        self.twin.tor = struct()

        datlist = [self.t, self.pol.Ti, self.pol.Tier, self.pol.Vc, self.pol.Ver,
                   self.pol.inc, self.pol.icer,
                   self.pol.Vr, self.pol.reff,
                   self.pol.p0, self.pol.pf, self.pol.ip, self.pol.ipf,
                   self.pol.Br, self.pol.Bz, self.pol.Bphi, self.pol.dVdreff,
                   self.pol.Te, self.pol.ne, self.pol.t1,
                   self.pol.reffa99,
                   self.tor.Ti, self.tor.Tier, self.tor.Vc, self.tor.Ver, self.tor.inc, self.tor.icer,
                   self.tor.Vr, self.tor.reff,
                   self.tor.p0, self.tor.pf, self.tor.ip, self.tor.ipf,
                   self.tor.Br, self.tor.Bz, self.tor.Bphi, self.tor.dVdreff,
                   self.tor.Te, self.tor.ne, self.tor.t1,
                   self.tor.reffa99]
        _idxs, datlist_win = proc.getTimeIdxsAndDats(self.t, self.twin.tstart, self.twin.tend, datlist)
        self.twin.t, self.twin.pol.Ti, self.twin.pol.Tier, self.twin.pol.Vc, self.twin.pol.Ver, \
        self.twin.pol.inc, self.twin.pol.icer, \
        self.twin.pol.Vr, self.twin.pol.reff, \
        self.twin.pol.p0, self.twin.pol.pf, self.twin.pol.ip, self.twin.pol.ipf, \
        self.twin.pol.Br, self.twin.pol.Bz, self.twin.pol.Bphi, self.twin.pol.dVdreff, \
        self.twin.pol.Te, self.twin.pol.ne, self.twin.pol.t1, \
        self.twin.pol.reffa99, \
        self.twin.tor.Ti, self.twin.tor.Tier, self.twin.tor.Vc, self.twin.tor.Ver, \
        self.twin.tor.inc, self.twin.tor.icer, \
        self.twin.tor.Vr, self.twin.tor.reff, \
        self.twin.tor.p0, self.twin.tor.pf, self.twin.tor.ip, self.twin.tor.ipf, \
        self.twin.tor.Br, self.twin.tor.Bz, self.twin.tor.Bphi, self.twin.tor.dVdreff, \
        self.twin.tor.Te, self.twin.tor.ne, self.twin.tor.t1, \
        self.twin.tor.reffa99 = datlist_win

        if include_grad:
            datlist = [self.pol.Ti_polyfit, self.pol.Ti_polyfit_err,
                       self.pol.dTidreff_polyfit, self.pol.dTidreff_polyfit_err,
                       self.pol.LTi_polyfit, self.pol.LTi_polyfit_err,
                       self.pol.RLTi_polyfit, self.pol.RLTi_polyfit_err,
                       self.pol.Vc_polyfit, self.pol.Vc_polyfit_err,
                       self.pol.dVcdreff_polyfit, self.pol.dVcdreff_polyfit_err,
                       self.pol.LVc_polyfit, self.pol.LVc_polyfit_err,
                       self.pol.RLVc_polyfit, self.pol.RLVc_polyfit_err]
            _idxs, datlist_win = proc.getTimeIdxsAndDats(self.pol.t, self.twin.tstart, self.twin.tend, datlist)
            self.twin.pol.Ti_polyfit, self.twin.pol.Ti_polyfit_err, \
            self.twin.pol.dTidreff_polyfit, self.twin.pol.dTidreff_polyfit_err, \
            self.twin.pol.LTi_polyfit, self.twin.pol.LTi_polyfit_err, \
            self.twin.pol.RLTi_polyfit, self.twin.pol.RLTi_polyfit_err, \
            self.twin.pol.Vc_polyfit, self.twin.pol.Vc_polyfit_err, \
            self.twin.pol.dVcdreff_polyfit, self.twin.pol.dVcdreff_polyfit_err, \
            self.twin.pol.LVc_polyfit, self.twin.pol.LVc_polyfit_err, \
            self.twin.pol.RLVc_polyfit, self.twin.pol.RLVc_polyfit_err = datlist_win

            datlist = [self.tor.Ti_polyfit, self.tor.Ti_polyfit_err,
                       self.tor.dTidreff_polyfit, self.tor.dTidreff_polyfit_err,
                       self.tor.LTi_polyfit, self.tor.LTi_polyfit_err,
                       self.tor.RLTi_polyfit, self.tor.RLTi_polyfit_err,
                       self.tor.Vc_polyfit, self.tor.Vc_polyfit_err,
                       self.tor.dVcdreff_polyfit, self.tor.dVcdreff_polyfit_err,
                       self.tor.LVc_polyfit, self.tor.LVc_polyfit_err,
                       self.tor.RLVc_polyfit, self.tor.RLVc_polyfit_err]
            _idxs, datlist_win = proc.getTimeIdxsAndDats(self.tor.t, self.twin.tstart, self.twin.tend, datlist)
            self.twin.tor.Ti_polyfit, self.twin.tor.Ti_polyfit_err, \
            self.twin.tor.dTidreff_polyfit, self.twin.tor.dTidreff_polyfit_err, \
            self.twin.tor.LTi_polyfit, self.twin.tor.LTi_polyfit_err, \
            self.twin.tor.RLTi_polyfit, self.twin.tor.RLTi_polyfit_err, \
            self.twin.tor.Vc_polyfit, self.twin.tor.Vc_polyfit_err, \
            self.twin.tor.dVcdreff_polyfit, self.twin.tor.dVcdreff_polyfit_err, \
            self.twin.tor.LVc_polyfit, self.twin.tor.LVc_polyfit_err, \
            self.twin.tor.RLVc_polyfit, self.twin.tor.RLVc_polyfit_err = datlist_win

        if include_teti:
            datlist = [self.pol.teti, self.pol.teti_err]
            _idxs, datlist_win = proc.getTimeIdxsAndDats(self.pol.t, self.twin.tstart, self.twin.tend, datlist)
            self.twin.pol.teti, self.twin.pol.teti_err = datlist_win

            datlist = [self.tor.teti, self.tor.teti_err]
            _idxs, datlist_win = proc.getTimeIdxsAndDats(self.tor.t, self.twin.tstart, self.twin.tend, datlist)
            self.twin.tor.teti, self.twin.tor.teti_err = datlist_win

        self.twin.pol.avg = struct()
        self.twin.pol.std = struct()
        self.twin.pol.ste = struct()
        self.twin.tor.avg = struct()
        self.twin.tor.std = struct()
        self.twin.tor.ste = struct()

        self.twin.pol.avg.reff, self.twin.pol.std.reff, self.twin.pol.ste.reff \
            = calc.average(self.twin.pol.reff, err=None, axis=0)
        self.twin.pol.avg.reffa99, self.twin.pol.std.reffa99, self.twin.pol.ste.reffa99 \
            = calc.average(self.twin.pol.reffa99, err=None, axis=0)
        self.twin.pol.avg.Ti, self.twin.pol.std.Ti, self.twin.pol.ste.Ti \
            = calc.average(self.twin.pol.Ti, err=self.twin.pol.Tier, axis=0)
        self.twin.pol.avg.Vc, self.twin.pol.std.Vc, self.twin.pol.ste.Vc \
            = calc.average(self.twin.pol.Vc, err=self.twin.pol.Ver, axis=0)

        self.twin.tor.avg.reff, self.twin.tor.std.reff, self.twin.tor.ste.reff \
            = calc.average(self.twin.tor.reff, err=None, axis=0)
        self.twin.tor.avg.reffa99, self.twin.tor.std.reffa99, self.twin.tor.ste.reffa99 \
            = calc.average(self.twin.tor.reffa99, err=None, axis=0)
        self.twin.tor.avg.Ti, self.twin.tor.std.Ti, self.twin.tor.ste.Ti \
            = calc.average(self.twin.tor.Ti, err=self.twin.tor.Tier, axis=0)
        self.twin.tor.avg.Vc, self.twin.tor.std.Vc, self.twin.tor.ste.Vc \
            = calc.average(self.twin.tor.Vc, err=self.twin.tor.Ver, axis=0)

        if include_grad:
            self.twin.pol.avg.Ti_polyfit, self.twin.pol.std.Ti_polyfit, self.twin.pol.ste.Ti_polyfit \
                = calc.average(self.twin.pol.Ti_polyfit, err=self.twin.pol.Ti_polyfit_err, axis=0)
            self.twin.pol.avg.dTidreff_polyfit, self.twin.pol.std.dTidreff_polyfit, self.twin.pol.ste.dTidreff_polyfit \
                = calc.average(self.twin.pol.dTidreff_polyfit, err=self.twin.pol.dTidreff_polyfit_err, axis=0)
            self.twin.pol.avg.LTi_polyfit, self.twin.pol.std.LTi_polyfit, self.twin.pol.ste.LTi_polyfit \
                = calc.average(self.twin.pol.LTi_polyfit, err=self.twin.pol.LTi_polyfit_err, axis=0)
            self.twin.pol.avg.RLTi_polyfit, self.twin.pol.std.RLTi_polyfit, self.twin.pol.ste.RLTi_polyfit \
                = calc.average(self.twin.pol.RLTi_polyfit, err=self.twin.pol.RLTi_polyfit_err, axis=0)

            self.twin.pol.avg.Vc_polyfit, self.twin.pol.std.Vc_polyfit, self.twin.pol.ste.Vc_polyfit \
                = calc.average(self.twin.pol.Vc_polyfit, err=self.twin.pol.Vc_polyfit_err, axis=0)
            self.twin.pol.avg.dVcdreff_polyfit, self.twin.pol.std.dVcdreff_polyfit, self.twin.pol.ste.dVcdreff_polyfit \
                = calc.average(self.twin.pol.dVcdreff_polyfit, err=self.twin.pol.dVcdreff_polyfit_err, axis=0)
            self.twin.pol.avg.LVc_polyfit, self.twin.pol.std.LVc_polyfit, self.twin.pol.ste.LVc_polyfit \
                = calc.average(self.twin.pol.LVc_polyfit, err=self.twin.pol.LVc_polyfit_err, axis=0)
            self.twin.pol.avg.RLVc_polyfit, self.twin.pol.std.RLVc_polyfit, self.twin.pol.ste.RLVc_polyfit \
                = calc.average(self.twin.pol.RLVc_polyfit, err=self.twin.pol.RLVc_polyfit_err, axis=0)

            self.twin.tor.avg.Ti_polyfit, self.twin.tor.std.Ti_polyfit, self.twin.tor.ste.Ti_polyfit \
                = calc.average(self.twin.tor.Ti_polyfit, err=self.twin.tor.Ti_polyfit_err, axis=0)
            self.twin.tor.avg.dTidreff_polyfit, self.twin.tor.std.dTidreff_polyfit, self.twin.tor.ste.dTidreff_polyfit \
                = calc.average(self.twin.tor.dTidreff_polyfit, err=self.twin.tor.dTidreff_polyfit_err, axis=0)
            self.twin.tor.avg.LTi_polyfit, self.twin.tor.std.LTi_polyfit, self.twin.tor.ste.LTi_polyfit \
                = calc.average(self.twin.tor.LTi_polyfit, err=self.twin.tor.LTi_polyfit_err, axis=0)
            self.twin.tor.avg.RLTi_polyfit, self.twin.tor.std.RLTi_polyfit, self.twin.tor.ste.RLTi_polyfit \
                = calc.average(self.twin.tor.RLTi_polyfit, err=self.twin.tor.RLTi_polyfit_err, axis=0)

            self.twin.tor.avg.Vc_polyfit, self.twin.tor.std.Vc_polyfit, self.twin.tor.ste.Vc_polyfit \
                = calc.average(self.twin.tor.Vc_polyfit, err=self.twin.tor.Vc_polyfit_err, axis=0)
            self.twin.tor.avg.dVcdreff_polyfit, self.twin.tor.std.dVcdreff_polyfit, self.twin.tor.ste.dVcdreff_polyfit \
                = calc.average(self.twin.tor.dVcdreff_polyfit, err=self.twin.tor.dVcdreff_polyfit_err, axis=0)
            self.twin.tor.avg.LVc_polyfit, self.twin.tor.std.LVc_polyfit, self.twin.tor.ste.LVc_polyfit \
                = calc.average(self.twin.tor.LVc_polyfit, err=self.twin.tor.LVc_polyfit_err, axis=0)
            self.twin.tor.avg.RLVc_polyfit, self.twin.tor.std.RLVc_polyfit, self.twin.tor.ste.RLVc_polyfit \
                = calc.average(self.twin.tor.RLVc_polyfit, err=self.twin.tor.RLVc_polyfit_err, axis=0)

        if include_teti:
            self.twin.pol.avg.teti, self.twin.pol.std.teti, self.twin.pol.ste.teti \
                = calc.average(self.twin.pol.teti, err=self.twin.pol.teti_err, axis=0)
            self.twin.tor.avg.teti, self.twin.tor.std.teti, self.twin.tor.ste.teti \
                = calc.average(self.twin.tor.teti, err=self.twin.tor.teti_err, axis=0)

    def R_window(self, Rat=4.1, dR=0.106, include_outerside=False, include_grad=False, include_teti=False):

        self.Rwin = struct()
        self.Rwin.pol = struct()
        self.Rwin.tor = struct()
        self.Rwin.Rat = Rat
        self.Rwin.dR = dR
        self.Rwin.Rin = Rat - 0.5 * dR
        self.Rwin.Rout = Rat + 0.5 * dR

        datlist = [self.pol.Ti, self.pol.Tier, self.pol.Vc, self.pol.Ver, self.pol.inc, self.pol.icer,
                   self.pol.Vr, self.pol.reff,
                   self.pol.p0, self.pol.pf, self.pol.ip, self.pol.ipf,
                   self.pol.Br, self.pol.Bz, self.pol.Bphi, self.pol.dVdreff,
                   self.pol.Te, self.pol.ne, self.pol.t1,
                   self.pol.reffa99]
        _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.pol.R, x_start=self.Rwin.Rin, x_end=self.Rwin.Rout,
                                                                Ys_list=datlist, include_outerside=include_outerside)
        self.Rwin.pol.Ti, self.Rwin.pol.Tier, self.Rwin.pol.Vc, self.Rwin.pol.Ver, \
        self.Rwin.pol.inc, self.Rwin.pol.icer, \
        self.Rwin.pol.Vr, self.Rwin.pol.reff, \
        self.Rwin.pol.p0, self.Rwin.pol.pf, self.Rwin.pol.ip, self.Rwin.pol.ipf, \
        self.Rwin.pol.Br, self.Rwin.pol.Bz, self.Rwin.pol.Bphi, self.Rwin.pol.dVdreff, \
        self.Rwin.pol.Te, self.Rwin.pol.ne, self.Rwin.pol.t1, \
        self.Rwin.pol.reffa99 = datlist_win
        self.Rwin.pol.R = self.pol.R[_idxs]

        datlist = [self.tor.Ti, self.tor.Tier, self.tor.Vc, self.tor.Ver, self.tor.inc, self.tor.icer,
                   self.tor.Vr, self.tor.reff,
                   self.tor.p0, self.tor.pf, self.tor.ip, self.tor.ipf,
                   self.tor.Br, self.tor.Bz, self.tor.Bphi, self.tor.dVdreff,
                   self.tor.Te, self.tor.ne, self.tor.t1,
                   self.tor.reffa99]
        _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.tor.R, x_start=self.Rwin.Rin,
                                                                x_end=self.Rwin.Rout,
                                                                Ys_list=datlist, include_outerside=include_outerside)
        self.Rwin.tor.Ti, self.Rwin.tor.Tier, self.Rwin.tor.Vc, self.Rwin.tor.Ver, \
        self.Rwin.tor.inc, self.Rwin.tor.icer, \
        self.Rwin.tor.Vr, self.Rwin.tor.reff, \
        self.Rwin.tor.p0, self.Rwin.tor.pf, self.Rwin.tor.ip, self.Rwin.tor.ipf, \
        self.Rwin.tor.Br, self.Rwin.tor.Bz, self.Rwin.tor.Bphi, self.Rwin.tor.dVdreff, \
        self.Rwin.tor.Te, self.Rwin.tor.ne, self.Rwin.tor.t1, \
        self.Rwin.tor.reffa99 = datlist_win
        self.Rwin.tor.R = self.tor.R[_idxs]

        if include_grad:
            datlist = [self.pol.Ti_polyfit, self.pol.Ti_polyfit_err,
                       self.pol.dTidreff_polyfit, self.pol.dTidreff_polyfit_err,
                       self.pol.LTi_polyfit, self.pol.LTi_polyfit_err,
                       self.pol.RLTi_polyfit, self.pol.RLTi_polyfit_err,
                       self.pol.Vc_polyfit, self.pol.Vc_polyfit_err,
                       self.pol.dVcdreff_polyfit, self.pol.dVcdreff_polyfit_err,
                       self.pol.LVc_polyfit, self.pol.LVc_polyfit_err,
                       self.pol.RLVc_polyfit, self.pol.RLVc_polyfit_err]
            _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.pol.R, x_start=self.Rwin.Rin,
                                                                    x_end=self.Rwin.Rout,
                                                                    Ys_list=datlist,
                                                                    include_outerside=include_outerside)
            self.Rwin.pol.Ti_polyfit, self.Rwin.pol.Ti_polyfit_err, \
            self.Rwin.pol.dTidreff_polyfit, self.Rwin.pol.dTidreff_polyfit_err, \
            self.Rwin.pol.LTi_polyfit, self.Rwin.pol.LTi_polyfit_err, \
            self.Rwin.pol.RLTi_polyfit, self.Rwin.pol.RLTi_polyfit_err, \
            self.Rwin.pol.Vc_polyfit, self.Rwin.pol.Vc_polyfit_err, \
            self.Rwin.pol.dVcdreff_polyfit, self.Rwin.pol.dVcdreff_polyfit_err, \
            self.Rwin.pol.LVc_polyfit, self.Rwin.pol.LVc_polyfit_err, \
            self.Rwin.pol.RLVc_polyfit, self.Rwin.pol.RLVc_polyfit_err = datlist_win

            datlist = [self.tor.Ti_polyfit, self.tor.Ti_polyfit_err,
                       self.tor.dTidreff_polyfit, self.tor.dTidreff_polyfit_err,
                       self.tor.LTi_polyfit, self.tor.LTi_polyfit_err,
                       self.tor.RLTi_polyfit, self.tor.RLTi_polyfit_err,
                       self.tor.Vc_polyfit, self.tor.Vc_polyfit_err,
                       self.tor.dVcdreff_polyfit, self.tor.dVcdreff_polyfit_err,
                       self.tor.LVc_polyfit, self.tor.LVc_polyfit_err,
                       self.tor.RLVc_polyfit, self.tor.RLVc_polyfit_err]
            _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.tor.R, x_start=self.Rwin.Rin,
                                                                    x_end=self.Rwin.Rout,
                                                                    Ys_list=datlist,
                                                                    include_outerside=include_outerside)
            self.Rwin.tor.Ti_polyfit, self.Rwin.tor.Ti_polyfit_err, \
            self.Rwin.tor.dTidreff_polyfit, self.Rwin.tor.dTidreff_polyfit_err, \
            self.Rwin.tor.LTi_polyfit, self.Rwin.tor.LTi_polyfit_err, \
            self.Rwin.tor.RLTi_polyfit, self.Rwin.tor.RLTi_polyfit_err, \
            self.Rwin.tor.Vc_polyfit, self.Rwin.tor.Vc_polyfit_err, \
            self.Rwin.tor.dVcdreff_polyfit, self.Rwin.tor.dVcdreff_polyfit_err, \
            self.Rwin.tor.LVc_polyfit, self.Rwin.tor.LVc_polyfit_err, \
            self.Rwin.tor.RLVc_polyfit, self.Rwin.tor.RLVc_polyfit_err = datlist_win

        if include_teti:
            datlist = [self.pol.teti, self.pol.teti_err]
            _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.pol.R, x_start=self.Rwin.Rin,
                                                                    x_end=self.Rwin.Rout,
                                                                    Ys_list=datlist,
                                                                    include_outerside=include_outerside)
            self.Rwin.pol.teti, self.Rwin.pol.teti_err = datlist_win

            datlist = [self.tor.teti, self.tor.teti_err]
            _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.tor.R, x_start=self.Rwin.Rin,
                                                                    x_end=self.Rwin.Rout,
                                                                    Ys_list=datlist,
                                                                    include_outerside=include_outerside)
            self.Rwin.tor.teti, self.Rwin.tor.teti_err = datlist_win

        self.Rwin.pol.reffin = np.ravel(self.Rwin.pol.reff[:, 0])
        self.Rwin.pol.reffout = np.ravel(self.Rwin.pol.reff[:, -1])
        self.Rwin.pol.reffa99in = np.ravel(self.Rwin.pol.reffa99[:, 0])
        self.Rwin.pol.reffa99out = np.ravel(self.Rwin.pol.reffa99[:, -1])
        self.Rwin.tor.reffin = np.ravel(self.Rwin.tor.reff[:, 0])
        self.Rwin.tor.reffout = np.ravel(self.Rwin.tor.reff[:, -1])
        self.Rwin.tor.reffa99in = np.ravel(self.Rwin.tor.reffa99[:, 0])
        self.Rwin.tor.reffa99out = np.ravel(self.Rwin.tor.reffa99[:, -1])

        self.Rwin.pol.avg = struct()
        self.Rwin.pol.std = struct()
        self.Rwin.pol.ste = struct()
        self.Rwin.tor.avg = struct()
        self.Rwin.tor.std = struct()
        self.Rwin.tor.ste = struct()

        self.Rwin.pol.avg.reff, self.Rwin.pol.std.reff, self.Rwin.pol.ste.reff \
            = calc.average(self.Rwin.pol.reff, err=None, axis=1)
        self.Rwin.pol.avg.reffa99, self.Rwin.pol.std.reffa99, self.Rwin.pol.ste.reffa99 \
            = calc.average(self.Rwin.pol.reffa99, err=None, axis=1)
        self.Rwin.pol.avg.Ti, self.Rwin.pol.std.Ti, self.Rwin.pol.ste.Ti \
            = calc.average(self.Rwin.pol.Ti, err=self.Rwin.pol.Tier, axis=1)
        self.Rwin.pol.avg.Vc, self.Rwin.pol.std.Vc, self.Rwin.pol.ste.Vc \
            = calc.average(self.Rwin.pol.Vc, err=self.Rwin.pol.Ver, axis=1)

        self.Rwin.tor.avg.reff, self.Rwin.tor.std.reff, self.Rwin.tor.ste.reff \
            = calc.average(self.Rwin.tor.reff, err=None, axis=1)
        self.Rwin.tor.avg.reffa99, self.Rwin.tor.std.reffa99, self.Rwin.tor.ste.reffa99 \
            = calc.average(self.Rwin.tor.reffa99, err=None, axis=1)
        self.Rwin.tor.avg.Ti, self.Rwin.tor.std.Ti, self.Rwin.tor.ste.Ti \
            = calc.average(self.Rwin.tor.Ti, err=self.Rwin.tor.Tier, axis=1)
        self.Rwin.tor.avg.Vc, self.Rwin.tor.std.Vc, self.Rwin.tor.ste.Vc \
            = calc.average(self.Rwin.tor.Vc, err=self.Rwin.tor.Ver, axis=1)


        if include_grad:
            self.Rwin.pol.avg.Ti_polyfit, self.Rwin.pol.std.Ti_polyfit, self.Rwin.pol.ste.Ti_polyfit \
                = calc.average(self.Rwin.pol.Ti_polyfit, err=self.Rwin.pol.Ti_polyfit_err, axis=1)
            self.Rwin.pol.avg.dTidreff_polyfit, self.Rwin.pol.std.dTidreff_polyfit, self.Rwin.pol.ste.dTidreff_polyfit \
                = calc.average(self.Rwin.pol.dTidreff_polyfit, err=self.Rwin.pol.dTidreff_polyfit_err, axis=1)
            self.Rwin.pol.avg.LTi_polyfit, self.Rwin.pol.std.LTi_polyfit, self.Rwin.pol.ste.LTi_polyfit \
                = calc.average(self.Rwin.pol.LTi_polyfit, err=self.Rwin.pol.LTi_polyfit_err, axis=1)
            self.Rwin.pol.avg.RLTi_polyfit, self.Rwin.pol.std.RLTi_polyfit, self.Rwin.pol.ste.RLTi_polyfit \
                = calc.average(self.Rwin.pol.RLTi_polyfit, err=self.Rwin.pol.RLTi_polyfit_err, axis=1)

            self.Rwin.pol.avg.Vc_polyfit, self.Rwin.pol.std.Vc_polyfit, self.Rwin.pol.ste.Vc_polyfit \
                = calc.average(self.Rwin.pol.Vc_polyfit, err=self.Rwin.pol.Vc_polyfit_err, axis=1)
            self.Rwin.pol.avg.dVcdreff_polyfit, self.Rwin.pol.std.dVcdreff_polyfit, self.Rwin.pol.ste.dVcdreff_polyfit \
                = calc.average(self.Rwin.pol.dVcdreff_polyfit, err=self.Rwin.pol.dVcdreff_polyfit_err, axis=1)
            self.Rwin.pol.avg.LVc_polyfit, self.Rwin.pol.std.LVc_polyfit, self.Rwin.pol.ste.LVc_polyfit \
                = calc.average(self.Rwin.pol.LVc_polyfit, err=self.Rwin.pol.LVc_polyfit_err, axis=1)
            self.Rwin.pol.avg.RLVc_polyfit, self.Rwin.pol.std.RLVc_polyfit, self.Rwin.pol.ste.RLVc_polyfit \
                = calc.average(self.Rwin.pol.RLVc_polyfit, err=self.Rwin.pol.RLVc_polyfit_err, axis=1)

            self.Rwin.tor.avg.Ti_polyfit, self.Rwin.tor.std.Ti_polyfit, self.Rwin.tor.ste.Ti_polyfit \
                = calc.average(self.Rwin.tor.Ti_polyfit, err=self.Rwin.tor.Ti_polyfit_err, axis=1)
            self.Rwin.tor.avg.dTidreff_polyfit, self.Rwin.tor.std.dTidreff_polyfit, self.Rwin.tor.ste.dTidreff_polyfit \
                = calc.average(self.Rwin.tor.dTidreff_polyfit, err=self.Rwin.tor.dTidreff_polyfit_err, axis=1)
            self.Rwin.tor.avg.LTi_polyfit, self.Rwin.tor.std.LTi_polyfit, self.Rwin.tor.ste.LTi_polyfit \
                = calc.average(self.Rwin.tor.LTi_polyfit, err=self.Rwin.tor.LTi_polyfit_err, axis=1)
            self.Rwin.tor.avg.RLTi_polyfit, self.Rwin.tor.std.RLTi_polyfit, self.Rwin.tor.ste.RLTi_polyfit \
                = calc.average(self.Rwin.tor.RLTi_polyfit, err=self.Rwin.tor.RLTi_polyfit_err, axis=1)

            self.Rwin.tor.avg.Vc_polyfit, self.Rwin.tor.std.Vc_polyfit, self.Rwin.tor.ste.Vc_polyfit \
                = calc.average(self.Rwin.tor.Vc_polyfit, err=self.Rwin.tor.Vc_polyfit_err, axis=1)
            self.Rwin.tor.avg.dVcdreff_polyfit, self.Rwin.tor.std.dVcdreff_polyfit, self.Rwin.tor.ste.dVcdreff_polyfit \
                = calc.average(self.Rwin.tor.dVcdreff_polyfit, err=self.Rwin.tor.dVcdreff_polyfit_err, axis=1)
            self.Rwin.tor.avg.LVc_polyfit, self.Rwin.tor.std.LVc_polyfit, self.Rwin.tor.ste.LVc_polyfit \
                = calc.average(self.Rwin.tor.LVc_polyfit, err=self.Rwin.tor.LVc_polyfit_err, axis=1)
            self.Rwin.tor.avg.RLVc_polyfit, self.Rwin.tor.std.RLVc_polyfit, self.Rwin.tor.ste.RLVc_polyfit \
                = calc.average(self.Rwin.tor.RLVc_polyfit, err=self.Rwin.tor.RLVc_polyfit_err, axis=1)

        if include_teti:
            self.Rwin.pol.avg.teti, self.Rwin.pol.std.teti, self.Rwin.pol.ste.teti \
                = calc.average(self.Rwin.pol.teti, err=self.Rwin.pol.teti_err, axis=1)
            self.Rwin.tor.avg.teti, self.Rwin.tor.std.teti, self.Rwin.tor.ste.teti \
                = calc.average(self.Rwin.tor.teti, err=self.Rwin.tor.teti_err, axis=1)

    def tR_window(self, tstart=4.4, tend=4.5, Rat=4.1, dR=0.106,
                  include_outerside=False, include_grad=False, include_teti=False):

        self.tRwin = struct()
        self.tRwin.pol = struct()
        self.tRwin.tor = struct()

        self.tRwin.tstart = tstart
        self.tRwin.tend = tend

        self.tRwin.Rat = Rat
        self.tRwin.dR = dR
        self.tRwin.Rin = Rat - 0.5 * dR
        self.tRwin.Rout = Rat + 0.5 * dR

        datlist = [self.pol.Ti, self.pol.Tier, self.pol.Vc, self.pol.Ver, self.pol.inc, self.pol.icer,
                   self.pol.Vr, self.pol.reff,
                   self.pol.p0, self.pol.pf, self.pol.ip, self.pol.ipf,
                   self.pol.Br, self.pol.Bz, self.pol.Bphi, self.pol.dVdreff,
                   self.pol.Te, self.pol.ne, self.pol.t1,
                   self.pol.reffa99]
        _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.pol.R, x_start=self.tRwin.Rin, x_end=self.tRwin.Rout,
                                                                Ys_list=datlist, include_outerside=include_outerside)
        self.tRwin.pol.Ti, self.tRwin.pol.Tier, self.tRwin.pol.Vc, self.tRwin.pol.Ver, \
        self.tRwin.pol.inc, self.tRwin.pol.icer, \
        self.tRwin.pol.Vr, self.tRwin.pol.reff, \
        self.tRwin.pol.p0, self.tRwin.pol.pf, self.tRwin.pol.ip, self.tRwin.pol.ipf, \
        self.tRwin.pol.Br, self.tRwin.pol.Bz, self.tRwin.pol.Bphi, self.tRwin.pol.dVdreff, \
        self.tRwin.pol.Te, self.tRwin.pol.ne, self.tRwin.pol.t1, \
        self.tRwin.pol.reffa99 = datlist_win
        self.tRwin.pol.R = self.pol.R[_idxs]

        datlist = [self.tor.Ti, self.tor.Tier, self.tor.Vc, self.tor.Ver, self.tor.inc, self.tor.icer,
                   self.tor.Vr, self.tor.reff,
                   self.tor.p0, self.tor.pf, self.tor.ip, self.tor.ipf,
                   self.tor.Br, self.tor.Bz, self.tor.Bphi, self.tor.dVdreff,
                   self.tor.Te, self.tor.ne, self.tor.t1,
                   self.tor.reffa99]
        _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.tor.R, x_start=self.tRwin.Rin,
                                                                x_end=self.tRwin.Rout,
                                                                Ys_list=datlist, include_outerside=include_outerside)
        self.tRwin.tor.Ti, self.tRwin.tor.Tier, self.tRwin.tor.Vc, self.tRwin.tor.Ver, \
        self.tRwin.tor.inc, self.tRwin.tor.icer, \
        self.tRwin.tor.Vr, self.tRwin.tor.reff, \
        self.tRwin.tor.p0, self.tRwin.tor.pf, self.tRwin.tor.ip, self.tRwin.tor.ipf, \
        self.tRwin.tor.Br, self.tRwin.tor.Bz, self.tRwin.tor.Bphi, self.tRwin.tor.dVdreff, \
        self.tRwin.tor.Te, self.tRwin.tor.ne, self.tRwin.tor.t1, \
        self.tRwin.tor.reffa99 = datlist_win
        self.tRwin.tor.R = self.tor.R[_idxs]

        if include_grad:
            datlist = [self.pol.Ti_polyfit, self.pol.Ti_polyfit_err,
                       self.pol.dTidreff_polyfit, self.pol.dTidreff_polyfit_err,
                       self.pol.LTi_polyfit, self.pol.LTi_polyfit_err,
                       self.pol.RLTi_polyfit, self.pol.RLTi_polyfit_err,
                       self.pol.Vc_polyfit, self.pol.Vc_polyfit_err,
                       self.pol.dVcdreff_polyfit, self.pol.dVcdreff_polyfit_err,
                       self.pol.LVc_polyfit, self.pol.LVc_polyfit_err,
                       self.pol.RLVc_polyfit, self.pol.RLVc_polyfit_err]
            _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.pol.R, x_start=self.tRwin.Rin,
                                                                    x_end=self.tRwin.Rout,
                                                                    Ys_list=datlist,
                                                                    include_outerside=include_outerside)
            self.tRwin.pol.Ti_polyfit, self.tRwin.pol.Ti_polyfit_err, \
            self.tRwin.pol.dTidreff_polyfit, self.tRwin.pol.dTidreff_polyfit_err, \
            self.tRwin.pol.LTi_polyfit, self.tRwin.pol.LTi_polyfit_err, \
            self.tRwin.pol.RLTi_polyfit, self.tRwin.pol.RLTi_polyfit_err, \
            self.tRwin.pol.Vc_polyfit, self.tRwin.pol.Vc_polyfit_err, \
            self.tRwin.pol.dVcdreff_polyfit, self.tRwin.pol.dVcdreff_polyfit_err, \
            self.tRwin.pol.LVc_polyfit, self.tRwin.pol.LVc_polyfit_err, \
            self.tRwin.pol.RLVc_polyfit, self.tRwin.pol.RLVc_polyfit_err = datlist_win

            datlist = [self.tor.Ti_polyfit, self.tor.Ti_polyfit_err,
                       self.tor.dTidreff_polyfit, self.tor.dTidreff_polyfit_err,
                       self.tor.LTi_polyfit, self.tor.LTi_polyfit_err,
                       self.tor.RLTi_polyfit, self.tor.RLTi_polyfit_err,
                       self.tor.Vc_polyfit, self.tor.Vc_polyfit_err,
                       self.tor.dVcdreff_polyfit, self.tor.dVcdreff_polyfit_err,
                       self.tor.LVc_polyfit, self.tor.LVc_polyfit_err,
                       self.tor.RLVc_polyfit, self.tor.RLVc_polyfit_err]
            _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.tor.R, x_start=self.tRwin.Rin,
                                                                    x_end=self.tRwin.Rout,
                                                                    Ys_list=datlist,
                                                                    include_outerside=include_outerside)
            self.tRwin.tor.Ti_polyfit, self.tRwin.tor.Ti_polyfit_err, \
            self.tRwin.tor.dTidreff_polyfit, self.tRwin.tor.dTidreff_polyfit_err, \
            self.tRwin.tor.LTi_polyfit, self.tRwin.tor.LTi_polyfit_err, \
            self.tRwin.tor.RLTi_polyfit, self.tRwin.tor.RLTi_polyfit_err, \
            self.tRwin.tor.Vc_polyfit, self.tRwin.tor.Vc_polyfit_err, \
            self.tRwin.tor.dVcdreff_polyfit, self.tRwin.tor.dVcdreff_polyfit_err, \
            self.tRwin.tor.LVc_polyfit, self.tRwin.tor.LVc_polyfit_err, \
            self.tRwin.tor.RLVc_polyfit, self.tRwin.tor.RLVc_polyfit_err = datlist_win

        if include_teti:
            datlist = [self.pol.teti, self.pol.teti_err]
            _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.pol.R, x_start=self.tRwin.Rin,
                                                                    x_end=self.tRwin.Rout,
                                                                    Ys_list=datlist,
                                                                    include_outerside=include_outerside)
            self.tRwin.pol.teti, self.tRwin.pol.teti_err = datlist_win

            datlist = [self.tor.teti, self.tor.teti_err]
            _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.tor.R, x_start=self.tRwin.Rin,
                                                                    x_end=self.tRwin.Rout,
                                                                    Ys_list=datlist,
                                                                    include_outerside=include_outerside)
            self.tRwin.tor.teti, self.tRwin.tor.teti_err = datlist_win

        self.tRwin.pol.reffin = np.ravel(self.tRwin.pol.reff[:, 0])
        self.tRwin.pol.reffout = np.ravel(self.tRwin.pol.reff[:, -1])
        self.tRwin.pol.reffa99in = np.ravel(self.tRwin.pol.reffa99[:, 0])
        self.tRwin.pol.reffa99out = np.ravel(self.tRwin.pol.reffa99[:, -1])
        self.tRwin.tor.reffin = np.ravel(self.tRwin.tor.reff[:, 0])
        self.tRwin.tor.reffout = np.ravel(self.tRwin.tor.reff[:, -1])
        self.tRwin.tor.reffa99in = np.ravel(self.tRwin.tor.reffa99[:, 0])
        self.tRwin.tor.reffa99out = np.ravel(self.tRwin.tor.reffa99[:, -1])


        datlist = [self.t, self.tRwin.pol.Ti, self.tRwin.pol.Tier, self.tRwin.pol.Vc, self.tRwin.pol.Ver,
                   self.tRwin.pol.inc, self.tRwin.pol.icer,
                   self.tRwin.pol.Vr, self.tRwin.pol.reff,
                   self.tRwin.pol.p0, self.tRwin.pol.pf, self.tRwin.pol.ip, self.tRwin.pol.ipf,
                   self.tRwin.pol.Br, self.tRwin.pol.Bz, self.tRwin.pol.Bphi, self.tRwin.pol.dVdreff,
                   self.tRwin.pol.Te, self.tRwin.pol.ne, self.tRwin.pol.t1,
                   self.tRwin.pol.reffa99,
                   self.tRwin.pol.reffin, self.tRwin.pol.reffout,
                   self.tRwin.pol.reffa99in, self.tRwin.pol.reffa99out,
                   self.tRwin.tor.Ti, self.tRwin.tor.Tier, self.tRwin.tor.Vc, self.tRwin.tor.Ver, self.tRwin.tor.inc, self.tRwin.tor.icer,
                   self.tRwin.tor.Vr, self.tRwin.tor.reff,
                   self.tRwin.tor.p0, self.tRwin.tor.pf, self.tRwin.tor.ip, self.tRwin.tor.ipf,
                   self.tRwin.tor.Br, self.tRwin.tor.Bz, self.tRwin.tor.Bphi, self.tRwin.tor.dVdreff,
                   self.tRwin.tor.Te, self.tRwin.tor.ne, self.tRwin.tor.t1,
                   self.tRwin.tor.reffa99,
                   self.tRwin.tor.reffin, self.tRwin.tor.reffout,
                   self.tRwin.tor.reffa99in, self.tRwin.tor.reffa99out]
        _idxs, datlist_win = proc.getTimeIdxsAndDats(self.t, self.tRwin.tstart, self.tRwin.tend, datlist)
        self.tRwin.t, self.tRwin.pol.Ti, self.tRwin.pol.Tier, self.tRwin.pol.Vc, self.tRwin.pol.Ver, \
        self.tRwin.pol.inc, self.tRwin.pol.icer, \
        self.tRwin.pol.Vr, self.tRwin.pol.reff, \
        self.tRwin.pol.p0, self.tRwin.pol.pf, self.tRwin.pol.ip, self.tRwin.pol.ipf, \
        self.tRwin.pol.Br, self.tRwin.pol.Bz, self.tRwin.pol.Bphi, self.tRwin.pol.dVdreff, \
        self.tRwin.pol.Te, self.tRwin.pol.ne, self.tRwin.pol.t1, \
        self.tRwin.pol.reffa99, \
        self.tRwin.pol.reffin, self.tRwin.pol.reffout, \
        self.tRwin.pol.reffa99in, self.tRwin.pol.reffa99out, \
        self.tRwin.tor.Ti, self.tRwin.tor.Tier, self.tRwin.tor.Vc, self.tRwin.tor.Ver, \
        self.tRwin.tor.inc, self.tRwin.tor.icer, \
        self.tRwin.tor.Vr, self.tRwin.tor.reff, \
        self.tRwin.tor.p0, self.tRwin.tor.pf, self.tRwin.tor.ip, self.tRwin.tor.ipf, \
        self.tRwin.tor.Br, self.tRwin.tor.Bz, self.tRwin.tor.Bphi, self.tRwin.tor.dVdreff, \
        self.tRwin.tor.Te, self.tRwin.tor.ne, self.tRwin.tor.t1, \
        self.tRwin.tor.reffa99, \
        self.tRwin.tor.reffin, self.tRwin.tor.reffout, \
        self.tRwin.tor.reffa99in, self.tRwin.tor.reffa99out = datlist_win

        if include_grad:
            datlist = [self.tRwin.pol.Ti_polyfit, self.tRwin.pol.Ti_polyfit_err,
                       self.tRwin.pol.dTidreff_polyfit, self.tRwin.pol.dTidreff_polyfit_err,
                       self.tRwin.pol.LTi_polyfit, self.tRwin.pol.LTi_polyfit_err,
                       self.tRwin.pol.RLTi_polyfit, self.tRwin.pol.RLTi_polyfit_err,
                       self.tRwin.pol.Vc_polyfit, self.tRwin.pol.Vc_polyfit_err,
                       self.tRwin.pol.dVcdreff_polyfit, self.tRwin.pol.dVcdreff_polyfit_err,
                       self.tRwin.pol.LVc_polyfit, self.tRwin.pol.LVc_polyfit_err,
                       self.tRwin.pol.RLVc_polyfit, self.tRwin.pol.RLVc_polyfit_err]
            _idxs, datlist_win = proc.getTimeIdxsAndDats(self.t, self.tRwin.tstart, self.tRwin.tend, datlist)
            self.tRwin.pol.Ti_polyfit, self.tRwin.pol.Ti_polyfit_err, \
            self.tRwin.pol.dTidreff_polyfit, self.tRwin.pol.dTidreff_polyfit_err, \
            self.tRwin.pol.LTi_polyfit, self.tRwin.pol.LTi_polyfit_err, \
            self.tRwin.pol.RLTi_polyfit, self.tRwin.pol.RLTi_polyfit_err, \
            self.tRwin.pol.Vc_polyfit, self.tRwin.pol.Vc_polyfit_err, \
            self.tRwin.pol.dVcdreff_polyfit, self.tRwin.pol.dVcdreff_polyfit_err, \
            self.tRwin.pol.LVc_polyfit, self.tRwin.pol.LVc_polyfit_err, \
            self.tRwin.pol.RLVc_polyfit, self.tRwin.pol.RLVc_polyfit_err = datlist_win

            datlist = [self.tRwin.tor.Ti_polyfit, self.tRwin.tor.Ti_polyfit_err,
                       self.tRwin.tor.dTidreff_polyfit, self.tRwin.tor.dTidreff_polyfit_err,
                       self.tRwin.tor.LTi_polyfit, self.tRwin.tor.LTi_polyfit_err,
                       self.tRwin.tor.RLTi_polyfit, self.tRwin.tor.RLTi_polyfit_err,
                       self.tRwin.tor.Vc_polyfit, self.tRwin.tor.Vc_polyfit_err,
                       self.tRwin.tor.dVcdreff_polyfit, self.tRwin.tor.dVcdreff_polyfit_err,
                       self.tRwin.tor.LVc_polyfit, self.tRwin.tor.LVc_polyfit_err,
                       self.tRwin.tor.RLVc_polyfit, self.tRwin.tor.RLVc_polyfit_err]
            _idxs, datlist_win = proc.getTimeIdxsAndDats(self.t, self.tRwin.tstart, self.tRwin.tend, datlist)
            self.tRwin.tor.Ti_polyfit, self.tRwin.tor.Ti_polyfit_err, \
            self.tRwin.tor.dTidreff_polyfit, self.tRwin.tor.dTidreff_polyfit_err, \
            self.tRwin.tor.LTi_polyfit, self.tRwin.tor.LTi_polyfit_err, \
            self.tRwin.tor.RLTi_polyfit, self.tRwin.tor.RLTi_polyfit_err, \
            self.tRwin.tor.Vc_polyfit, self.tRwin.tor.Vc_polyfit_err, \
            self.tRwin.tor.dVcdreff_polyfit, self.tRwin.tor.dVcdreff_polyfit_err, \
            self.tRwin.tor.LVc_polyfit, self.tRwin.tor.LVc_polyfit_err, \
            self.tRwin.tor.RLVc_polyfit, self.tRwin.tor.RLVc_polyfit_err = datlist_win

        if include_teti:
            datlist = [self.tRwin.pol.teti, self.tRwin.pol.teti_err]
            _idxs, datlist_win = proc.getTimeIdxsAndDats(self.t, self.tRwin.tstart, self.tRwin.tend, datlist)
            self.tRwin.pol.teti, self.tRwin.pol.teti_err = datlist_win

            datlist = [self.tRwin.tor.teti, self.tRwin.tor.teti_err]
            _idxs, datlist_win = proc.getTimeIdxsAndDats(self.t, self.tRwin.tstart, self.tRwin.tend, datlist)
            self.tRwin.tor.teti, self.tRwin.tor.teti_err = datlist_win

        self.tRwin.pol.avg = struct()
        self.tRwin.pol.std = struct()
        self.tRwin.pol.ste = struct()
        self.tRwin.tor.avg = struct()
        self.tRwin.tor.std = struct()
        self.tRwin.tor.ste = struct()

        self.tRwin.pol.avg.reff, self.tRwin.pol.std.reff, self.tRwin.pol.ste.reff \
            = calc.average(self.tRwin.pol.reff, err=None)
        self.tRwin.pol.avg.reffa99, self.tRwin.pol.std.reffa99, self.tRwin.pol.ste.reffa99 \
            = calc.average(self.tRwin.pol.reffa99, err=None)
        self.tRwin.pol.avg.Ti, self.tRwin.pol.std.Ti, self.tRwin.pol.ste.Ti \
            = calc.average(self.tRwin.pol.Ti, err=self.tRwin.pol.Tier)
        self.tRwin.pol.avg.Vc, self.tRwin.pol.std.Vc, self.tRwin.pol.ste.Vc \
            = calc.average(self.tRwin.pol.Vc, err=self.tRwin.pol.Ver)

        self.tRwin.tor.avg.reff, self.tRwin.tor.std.reff, self.tRwin.tor.ste.reff \
            = calc.average(self.tRwin.tor.reff, err=None)
        self.tRwin.tor.avg.reffa99, self.tRwin.tor.std.reffa99, self.tRwin.tor.ste.reffa99 \
            = calc.average(self.tRwin.tor.reffa99, err=None)
        self.tRwin.tor.avg.Ti, self.tRwin.tor.std.Ti, self.tRwin.tor.ste.Ti \
            = calc.average(self.tRwin.tor.Ti, err=self.tRwin.tor.Tier)
        self.tRwin.tor.avg.Vc, self.tRwin.tor.std.Vc, self.tRwin.tor.ste.Vc \
            = calc.average(self.tRwin.tor.Vc, err=self.tRwin.tor.Ver)


        if include_grad:
            self.tRwin.pol.avg.Ti_polyfit, self.tRwin.pol.std.Ti_polyfit, self.tRwin.pol.ste.Ti_polyfit \
                = calc.average(self.tRwin.pol.Ti_polyfit, err=self.tRwin.pol.Ti_polyfit_err)
            self.tRwin.pol.avg.dTidreff_polyfit, self.tRwin.pol.std.dTidreff_polyfit, self.tRwin.pol.ste.dTidreff_polyfit \
                = calc.average(self.tRwin.pol.dTidreff_polyfit, err=self.tRwin.pol.dTidreff_polyfit_err)
            self.tRwin.pol.avg.LTi_polyfit, self.tRwin.pol.std.LTi_polyfit, self.tRwin.pol.ste.LTi_polyfit \
                = calc.average(self.tRwin.pol.LTi_polyfit, err=self.tRwin.pol.LTi_polyfit_err)
            self.tRwin.pol.avg.RLTi_polyfit, self.tRwin.pol.std.RLTi_polyfit, self.tRwin.pol.ste.RLTi_polyfit \
                = calc.average(self.tRwin.pol.RLTi_polyfit, err=self.tRwin.pol.RLTi_polyfit_err)

            self.tRwin.pol.avg.Vc_polyfit, self.tRwin.pol.std.Vc_polyfit, self.tRwin.pol.ste.Vc_polyfit \
                = calc.average(self.tRwin.pol.Vc_polyfit, err=self.tRwin.pol.Vc_polyfit_err)
            self.tRwin.pol.avg.dVcdreff_polyfit, self.tRwin.pol.std.dVcdreff_polyfit, self.tRwin.pol.ste.dVcdreff_polyfit \
                = calc.average(self.tRwin.pol.dVcdreff_polyfit, err=self.tRwin.pol.dVcdreff_polyfit_err)
            self.tRwin.pol.avg.LVc_polyfit, self.tRwin.pol.std.LVc_polyfit, self.tRwin.pol.ste.LVc_polyfit \
                = calc.average(self.tRwin.pol.LVc_polyfit, err=self.tRwin.pol.LVc_polyfit_err)
            self.tRwin.pol.avg.RLVc_polyfit, self.tRwin.pol.std.RLVc_polyfit, self.tRwin.pol.ste.RLVc_polyfit \
                = calc.average(self.tRwin.pol.RLVc_polyfit, err=self.tRwin.pol.RLVc_polyfit_err)

            self.tRwin.tor.avg.Ti_polyfit, self.tRwin.tor.std.Ti_polyfit, self.tRwin.tor.ste.Ti_polyfit \
                = calc.average(self.tRwin.tor.Ti_polyfit, err=self.tRwin.tor.Ti_polyfit_err)
            self.tRwin.tor.avg.dTidreff_polyfit, self.tRwin.tor.std.dTidreff_polyfit, self.tRwin.tor.ste.dTidreff_polyfit \
                = calc.average(self.tRwin.tor.dTidreff_polyfit, err=self.tRwin.tor.dTidreff_polyfit_err)
            self.tRwin.tor.avg.LTi_polyfit, self.tRwin.tor.std.LTi_polyfit, self.tRwin.tor.ste.LTi_polyfit \
                = calc.average(self.tRwin.tor.LTi_polyfit, err=self.tRwin.tor.LTi_polyfit_err)
            self.tRwin.tor.avg.RLTi_polyfit, self.tRwin.tor.std.RLTi_polyfit, self.tRwin.tor.ste.RLTi_polyfit \
                = calc.average(self.tRwin.tor.RLTi_polyfit, err=self.tRwin.tor.RLTi_polyfit_err)

            self.tRwin.tor.avg.Vc_polyfit, self.tRwin.tor.std.Vc_polyfit, self.tRwin.tor.ste.Vc_polyfit \
                = calc.average(self.tRwin.tor.Vc_polyfit, err=self.tRwin.tor.Vc_polyfit_err)
            self.tRwin.tor.avg.dVcdreff_polyfit, self.tRwin.tor.std.dVcdreff_polyfit, self.tRwin.tor.ste.dVcdreff_polyfit \
                = calc.average(self.tRwin.tor.dVcdreff_polyfit, err=self.tRwin.tor.dVcdreff_polyfit_err)
            self.tRwin.tor.avg.LVc_polyfit, self.tRwin.tor.std.LVc_polyfit, self.tRwin.tor.ste.LVc_polyfit \
                = calc.average(self.tRwin.tor.LVc_polyfit, err=self.tRwin.tor.LVc_polyfit_err)
            self.tRwin.tor.avg.RLVc_polyfit, self.tRwin.tor.std.RLVc_polyfit, self.tRwin.tor.ste.RLVc_polyfit \
                = calc.average(self.tRwin.tor.RLVc_polyfit, err=self.tRwin.tor.RLVc_polyfit_err)

        if include_teti:
            self.tRwin.pol.avg.teti, self.tRwin.pol.std.teti, self.tRwin.pol.ste.teti \
                = calc.average(self.tRwin.pol.teti, err=self.tRwin.pol.teti_err)
            self.tRwin.tor.avg.teti, self.tRwin.tor.std.teti, self.tRwin.tor.ste.teti \
                = calc.average(self.tRwin.tor.teti, err=self.tRwin.tor.teti_err)

class cxs7_Er:

    def __init__(self, sn=184508, sub=1, tstart=3, tend=6, Er_lim=50):

        self.sn = sn
        self.sub = sub
        self.ts = tstart
        self.te = tend
        self.Er_lim = Er_lim

        self.t, self.R, list_dat, self.dimnms, self.valnms, self.dimunits, self.valunits \
            = read.eg2d("cxs7_Er", sn, sub)
        self.dt = self.t[1] - self.t[0]
        self.dR = self.R[1] - self.R[0]

        list_dat[10][list_dat[10] == 0.] = np.nan
        list_dat[10][np.abs(list_dat[10]) > self.Er_lim] = np.nan

        tidxs, list_dat = proc.getTimeIdxsAndDats(self.t, self.ts, self.te, list_dat)
        self.t = self.t[tidxs]

        # tidxs = ~np.isnan(list_dat[10]).all(axis=1)
        # self.t = self.t[tidxs]
        # for i in range(len(list_dat)):
        #     list_dat[i] = list_dat[i][tidxs]

        # Ridxs = ~np.isnan(list_dat[10]).any(axis=0)
        # self.R = self.R[Ridxs]
        # for i in range(len(list_dat)):
        #     list_dat[i] = list_dat[i][:, Ridxs]

        self.reff, self.reffa99, self.Erdia, self.Erdiaerr, self.Ervt, self.Ervterr, self.Ervp, self.Ervperr, \
        self.Ervperp, self.Ervperperr, self.Er, self.Ererr, self.Ereff, self.Erefferr, self.Inc, self.Incerr, \
        self.Tip, self.Tiperr, self.Vt, self.Vterr, self.Vp, self.Vperr, self.Br, self.Bz, self.Bphi = list_dat

        self.Bax, self.Rax, self.Bq, self.gamma, self.datetime, self.cycle = getShotInfo.info(self.sn)

        self.Br *= self.Bax / 3
        self.Bz *= self.Bax / 3
        self.Bphi *= self.Bax / 3

        self.B = np.sqrt(self.Br**2 + self.Bz**2 + self.Bphi**2)

        self.vExB, self.vExB_err = calc.divide(self.Er * self.Bphi, self.B**2, self.Ererr * self.Bphi)
        self.vExBeff, self.vExBeff_err = calc.divide(self.Ereff * self.Bphi, self.B**2, self.Erefferr * self.Bphi)

        self.dirnm = f"cxs7_Er"
        proc.ifNotMake(self.dirnm)

    def tat(self, time=4.5, include_grad=True):

        self.at = struct()
        datlist = [self.t,
                   self.reff, self.reffa99, self.Erdia, self.Erdiaerr, self.Ervt, self.Ervterr, self.Ervp, self.Ervperr,
                   self.Ervperp, self.Ervperperr, self.Er, self.Ererr, self.Ereff, self.Erefferr, self.Inc, self.Incerr,
                   self.Tip, self.Tiperr, self.Vt, self.Vterr, self.Vp, self.Vperr, self.Br, self.Bz, self.Bphi, 
                   self.B, self.vExB, self.vExB_err, self.vExBeff, self.vExBeff_err]

        _, datlist_at = proc.getTimeIdxAndDats(self.t, time, datlist)
        self.at.t, \
        self.at.reff, self.at.reffa99, self.at.Erdia, self.at.Erdiaerr, \
        self.at.Ervt, self.at.Ervterr, self.at.Ervp, self.at.Ervperr, \
        self.at.Ervperp, self.at.Ervperperr, self.at.Er, self.at.Ererr, \
        self.at.Ereff, self.at.Erefferr, self.at.Inc, self.at.Incerr, \
        self.at.Tip, self.at.Tiperr, self.at.Vt, self.at.Vterr, \
        self.at.Vp, self.at.Vperr, self.at.Br, self.at.Bz, self.at.Bphi, \
        self.at.B, self.at.vExB, self.at.vExB_err, self.at.vExBeff, self.at.vExBeff_err = datlist_at

        if include_grad:

            datlist = [self.Er_polyfit, self.Er_polyfit_err, self.dErdreff_polyfit, self.dErdreff_polyfit_err,
                       self.LEr_polyfit, self.LEr_polyfit_err, self.RLEr_polyfit, self.RLEr_polyfit_err,
                       self.Ereff_polyfit, self.Ereff_polyfit_err, self.dEreffdreff_polyfit, self.dEreffdreff_polyfit_err,
                       self.LEreff_polyfit, self.LEreff_polyfit_err, self.RLEreff_polyfit, self.RLEreff_polyfit_err,
                       self.vExB_polyfit, self.vExB_polyfit_err, self.dvExBdreff_polyfit, self.dvExBdreff_polyfit_err,
                       self.LvExB_polyfit, self.LvExB_polyfit_err, self.RLvExB_polyfit, self.RLvExB_polyfit_err,
                       self.vExBeff_polyfit, self.vExBeff_polyfit_err,
                       self.dvExBeffdreff_polyfit, self.dvExBeffdreff_polyfit_err,
                       self.LvExBeff_polyfit, self.LvExBeff_polyfit_err,
                       self.RLvExBeff_polyfit, self.RLvExBeff_polyfit_err
                       ]
            _, datlist_at = proc.getTimeIdxAndDats(self.t, time, datlist)
            self.at.Er_polyfit, self.at.Er_polyfit_err, self.at.dErdreff_polyfit, self.at.dErdreff_polyfit_err, \
            self.at.LEr_polyfit, self.at.LEr_polyfit_err, self.at.RLEr_polyfit, self.at.RLEr_polyfit_err, \
            self.at.Ereff_polyfit, self.at.Ereff_polyfit_err, \
            self.at.dEreffdreff_polyfit, self.at.dEreffdreff_polyfit_err, \
            self.at.LEreff_polyfit, self.at.LEreff_polyfit_err, \
            self.at.RLEreff_polyfit, self.at.RLEreff_polyfit_err, \
            self.at.vExB_polyfit, self.at.vExB_polyfit_err, \
            self.at.dvExBdreff_polyfit, self.at.dvExBdreff_polyfit_err, \
            self.at.LvExB_polyfit, self.at.LvExB_polyfit_err, \
            self.at.RLvExB_polyfit, self.at.RLvExB_polyfit_err, \
            self.at.vExBeff_polyfit, self.at.vExBeff_polyfit_err, \
            self.at.dvExBeffdreff_polyfit, self.at.dvExBeffdreff_polyfit_err, \
            self.at.LvExBeff_polyfit, self.at.LvExBeff_polyfit_err, \
            self.at.RLvExBeff_polyfit, self.at.RLvExBeff_polyfit_err = datlist_at

    def calcgrad(self, polyN=10):

        # a. 2)
        self.polyN = polyN

        _o = calc.polyN_LSM_der(xx=self.reff, yy=self.Er, polyN=polyN, yErr=self.Ererr, parity="even")
        self.Er_polyfit = _o.yHut
        self.Er_polyfit_err = _o.yHutErr
        self.dErdreff_polyfit = _o.yHutDer
        self.dErdreff_polyfit_err = _o.yHutDerErr
        self.LEr_polyfit, self.LEr_polyfit_err, self.RLEr_polyfit, self.RLEr_polyfit_err \
            = calc.Lscale(self.Er_polyfit, self.dErdreff_polyfit, self.Rax,
                          self.Er_polyfit_err, self.dErdreff_polyfit_err)

        _o = calc.polyN_LSM_der(xx=self.reff, yy=self.Ereff, polyN=polyN, yErr=self.Erefferr, parity="even")
        self.Ereff_polyfit = _o.yHut
        self.Ereff_polyfit_err = _o.yHutErr
        self.dEreffdreff_polyfit = _o.yHutDer
        self.dEreffdreff_polyfit_err = _o.yHutDerErr
        self.LEreff_polyfit, self.LEreff_polyfit_err, self.RLEreff_polyfit, self.RLEreff_polyfit_err \
            = calc.Lscale(self.Ereff_polyfit, self.dEreffdreff_polyfit, self.Rax,
                          self.Ereff_polyfit_err, self.dEreffdreff_polyfit_err)
        
        _o = calc.polyN_LSM_der(xx=self.reff, yy=self.vExB, polyN=polyN, yErr=self.vExB_err, parity="even")
        self.vExB_polyfit = _o.yHut
        self.vExB_polyfit_err = _o.yHutErr
        self.dvExBdreff_polyfit = _o.yHutDer
        self.dvExBdreff_polyfit_err = _o.yHutDerErr
        self.LvExB_polyfit, self.LvExB_polyfit_err, self.RLvExB_polyfit, self.RLvExB_polyfit_err \
            = calc.Lscale(self.vExB_polyfit, self.dvExBdreff_polyfit, self.Rax,
                          self.vExB_polyfit_err, self.dvExBdreff_polyfit_err)

        _o = calc.polyN_LSM_der(xx=self.reff, yy=self.vExBeff, polyN=polyN, yErr=self.vExBeff_err, parity="even")
        self.vExBeff_polyfit = _o.yHut
        self.vExBeff_polyfit_err = _o.yHutErr
        self.dvExBeffdreff_polyfit = _o.yHutDer
        self.dvExBeffdreff_polyfit_err = _o.yHutDerErr
        self.LvExBeff_polyfit, self.LvExBeff_polyfit_err, self.RLvExBeff_polyfit, self.RLvExBeff_polyfit_err \
            = calc.Lscale(self.vExBeff_polyfit, self.dvExBeffdreff_polyfit, self.Rax,
                          self.vExBeff_polyfit_err, self.dvExBeffdreff_polyfit_err)

    def plot_vExB(self, vExB_lim=5, cmap="coolwarm", pause=0):

        self.vExB_lim = vExB_lim
        self.cmap = cmap

        figdir = os.path.join(self.dirnm, "vExB")
        proc.ifNotMake(figdir)
        fnm = f"{self.sn}_{self.sub}_{self.ts}_{self.te}_{self.vExB_lim}"
        title = f"#{self.sn}-{self.sub}"
        path = os.path.join(figdir, f"{fnm}.png")
        fig, ax = plt.subplots(1)

        im = ax.pcolormesh(np.append(self.t - 0.5 * self.dt, self.t[-1] + 0.5 * self.dt),
                           np.append(self.R - 0.5 * self.dR, self.R[-1] + 0.5 * self.dR),
                           self.vExBeff.T, cmap=self.cmap, vmin=-self.vExB_lim, vmax=self.vExB_lim)
        cbar = plt.colorbar(im)
        cbar.set_label("vExB [km/s]")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("R [m]")

        plot.caption(fig, title)
        plot.capsave(fig, title, fnm, path)

        plot.check(pause)



class LID_cur:

    def __init__(self, sn=184508, sub=1):
        self.t, list_dat, self.list_dimnms, self.list_valnms, self.list_dimunits, self.list_valunits \
            = read.eg1d("LID_cur", sn, sub=1)
        if list_dat != None:
            self.A, self.B1, self.B2 = list_dat

    def tat(self, time=3.):

        self.at = struct()
        datlist = [self.t, self.A, self.B1, self.B2]
        _idx, datlist = proc.getTimeIdxAndDats(time=self.t, time_at=time, datList=datlist)
        self.at.t, self.at.A, self.at.B1, self.at.B2 = datlist

    def time_window(self, tstart=4., tend=5.):

        self.twin = struct()
        self.twin.tstart = tstart
        self.twin.tend = tend
        datlist = [self.t, self.A, self.B1, self.B2]
        _idx, datlist = proc.getTimeIdxsAndDats(self.t, self.twin.tstart, self.twin.tend, datlist)
        self.twin.t, self.twin.A, self.twin.B1, self.twin.B2 = datlist

        self.twin.avg = struct()
        self.twin.std = struct()
        self.twin.ste = struct()
        self.twin.avg.t, self.twin.std.t, self.twin.ste.t = calc.average(self.twin.t)
        self.twin.avg.A, self.twin.std.A, self.twin.ste.A = calc.average(self.twin.A)
        self.twin.avg.B1, self.twin.std.B1, self.twin.ste.B1 = calc.average(self.twin.B1)
        self.twin.avg.B2, self.twin.std.B2, self.twin.ste.B2 = calc.average(self.twin.B2)

class ece():

    def __init__(self, sn=185857, sub=1, tstart=3, tend=6, fluc_thresh=0.1):

        self.diagname = "ece_fast"
        self.sn = sn
        self.sub = sub
        self.ts = tstart
        self.te = tend
        self.t, self.R, list_dat, self.list_dimnms, self.list_valnms, self.list_dimunits, self.list_valunits \
            = read.eg2d(diagnm=self.diagname, sn=self.sn, sub=self.sub)
        if list_dat != None:
            tidxs, list_dat = proc.getTimeIdxsAndDats(self.t, tstart, tend, list_dat)
            self.t = self.t[tidxs]
            self.Te, self.fece, self.calib, self.diag_number, self.ADC_ch, self.rho_vacuum = list_dat
        else:
            self.Te, self.fece, self.calib, self.diag_number, self.ADC_ch, self.rho_vacuum \
                = [None, None, None, None, None, None]

        # fluc = 2 * np.std(self.Te[self.t.size // 2 - 10: self.t.size // 2 + 10], axis=0)
        # avg_tmp = 2 * np.average(self.Te[self.t.size // 2 - 10: self.t.size // 2 + 10], axis=0)
        # # eceDelRIdxs = [0, 1, 3, 18, 19, 25, 31, 33, 39, 40, 41, 42, 44, 45, 46, 47, 48, 52, 53, 55, 56, 57, 59]
        # eceDelRIdxs = np.where((fluc >= fluc_thresh) | (fluc / avg_tmp > fluc_thresh) | (avg_tmp < 0))[0]

        # self.R = np.delete(self.R, eceDelRIdxs)
        # self.Te = np.delete(self.Te, eceDelRIdxs)
        # self.fece = np.delete(self.fece, eceDelRIdxs)
        # self.calib = np.delete(self.calib, eceDelRIdxs)
        # self.diag_number = np.delete(self.diag_number, eceDelRIdxs)
        # self.ADC_ch = np.delete(self.ADC_ch, eceDelRIdxs)
        # self.rho_vacuum = np.delete(self.rho_vacuum, eceDelRIdxs)

        _idx1 = np.where(self.diag_number[0] == 1.)[0]
        _idx2 = np.where(self.diag_number[0] == 2.)[0]
        _idx3 = np.where(self.diag_number[0] == 3.)[0]

        self.radh = calc.struct()
        self.radl = calc.struct()
        self.radm = calc.struct()

        self.radh.R = self.R[_idx1]
        self.radh.Te = self.Te[:, _idx1]
        self.radh.fece = self.fece[0][_idx1]
        self.radh.calib = self.calib[:, _idx1]
        self.radh.ADC_ch = self.ADC_ch[0][_idx1]
        self.radh.rho_vacuum = self.rho_vacuum[0][_idx1]

        self.radl.R = self.R[_idx2]
        self.radl.Te = self.Te[:, _idx2]
        self.radl.fece = self.fece[0][_idx2]
        self.radl.calib = self.calib[:, _idx2]
        self.radl.ADC_ch = self.ADC_ch[0][_idx2]
        self.radl.rho_vacuum = self.rho_vacuum[0][_idx2]

        self.radm.R = self.R[_idx3]
        self.radm.Te = self.Te[:, _idx3]
        self.radm.fece = self.fece[0][_idx3]
        self.radm.calib = self.calib[:, _idx3]
        self.radm.ADC_ch = self.ADC_ch[0][_idx3]
        self.radm.rho_vacuum = self.rho_vacuum[0][_idx3]

        Ridxs_sort = np.argsort(self.R)
        self.R = self.R[Ridxs_sort]
        self.Te = self.Te[:, Ridxs_sort]
        self.fece = self.fece[0][Ridxs_sort]
        self.calib = self.calib[:, Ridxs_sort]
        self.ADC_ch = self.ADC_ch[0][Ridxs_sort]
        self.rho_vacuum = self.rho_vacuum[0][Ridxs_sort]

        """
        # self.dirbase = "ece"
        # proc.ifNotMake(self.dirbase)
        # self.fnm_base = f"{sn}_{sub}_{tstart}_{tend}_{fluc_thresh}"
        # self.figtitle = f"#{sn}-{sub} {tstart}-{tend}s"
        """


    def t_window(self, tstart=4, tend=5):

        self.twin = struct()
        self.twin.ts = tstart
        self.twin.te = tend
        datlist = [self.t, self.Te, self.calib]
        _, datlist_win = proc.getTimeIdxsAndDats(time=self.t, startTime=self.twin.ts,
                                                 endTime=self.twin.te, datList=datlist)
        self.twin.t, self.twin.Te, self.twin.calib = datlist_win

        self.twin.avg = struct()
        self.twin.std = struct()
        self.twin.ste = struct()

        self.twin.avg.Te, self.twin.std.Te, self.twin.ste.Te = \
            calc.average(self.twin.Te, err=None, axis=0)
        self.twin.avg.calib, self.twin.std.calib, self.twin.ste.calib = \
            calc.average(self.twin.calib, err=None, axis=0)
        
        return self.twin

    def calibration(self, tstart=4, tend=5, use_tsfit=True):

        self.tsmap = tsmap(self.sn, self.sub, self.ts, self.te)
        self.t_window(tstart, tend)
        if use_tsfit:
            self.tsmap.t_window(tstart, tend, include_grad=False)
            self.Te_ts = interp1d(self.tsmap.R, self.tsmap.twin.avg.Te_fit,
                                  bounds_error=False, fill_value="extrapolate")(self.R)
            self.Te_ts_err = interp1d(self.tsmap.R, self.tsmap.twin.ste.Te_fit,
                                  bounds_error=False, fill_value="extrapolate")(self.R)
            self.reffa99 = interp1d(self.tsmap.R, self.tsmap.twin.avg.reffa99,
                                  bounds_error=False, fill_value="extrapolate")(self.R)
            self.reffa99_err = interp1d(self.tsmap.R, self.tsmap.twin.ste.reffa99,
                                  bounds_error=False, fill_value="extrapolate")(self.R)
        else:
            self.tsmap.t_window(tstart, tend, include_grad=True)
            self.Te_ts = interp1d(self.tsmap.R, self.tsmap.twin.avg.Te_polyfit,
                                  bounds_error=False, fill_value="extrapolate")(self.R)
            self.Te_ts_err = interp1d(self.tsmap.R, self.tsmap.twin.ste.Te_polyfit,
                                  bounds_error=False, fill_value="extrapolate")(self.R)
            self.reffa99 = interp1d(self.tsmap.R, self.tsmap.twin.avg.reffa99,
                                  bounds_error=False, fill_value="extrapolate")(self.R)
            self.reffa99_err = interp1d(self.tsmap.R, self.tsmap.twin.ste.reffa99,
                                  bounds_error=False, fill_value="extrapolate")(self.R)

        self.tscal, self.tscal_err = calc.divide(self.Te_ts, self.twin.avg.Te, self.Te_ts_err, self.twin.ste.Te)
        self.Te_tscal, self.Te_tscal_err = calc.multiple(self.Te, np.abs(self.tscal), B_err=self.tscal_err)

    def Radius_at(self, Rat=4.0, iscalib=True):

        self.Rat = struct()
        self.Rat.R = Rat
        Ridx = np.argmin(np.abs(self.R - self.Rat.R))

        self.Rat.Te = self.Te[:, Ridx]
        self.Rat.rho_vacuum = self.rho_vacuum[Ridx]

        if iscalib:
            self.Rat.Te_tscal = self.Te_tscal[:, Ridx]
            self.Rat.Te_tscal_err = self.Te_tscal_err[:, Ridx]
            self.Rat.reffa99 = self.reffa99[Ridx]
            self.Rat.reffa99_err = self.reffa99_err[Ridx]

        return self.Rat

    """
    # def plot_ch(self, ADC_ch=1, pause=0):

    #     plot.set("notebook", "ticks")

    #     figdir = os.path.join(self.dirbase, "plot_ch")
    #     proc.ifNotMake(figdir)
    #     fnm = self.fnm_base + f"{ADC_ch}.png"
    #     path = os.path.join(figdir, fnm)

    #     fig, ax = plt.subplots(1)
    #     if ADC_ch == 1:
    #         for i in range(len(self.radh.R)):
    #             ax.plot(self.t, self.radh.Te[:, i],
    #                     label=f"{self.radh.R[i]}m")
    #     elif ADC_ch == 2:
    #         for i in range(len(self.radl.R)):
    #             ax.plot(self.t, self.radl.Te[:, i],
    #                     label=f"{self.radl.R[i]}m")
    #     elif ADC_ch == 3:
    #         for i in range(len(self.radm.R)):
    #             ax.plot(self.t, self.radm.Te[:, i],
    #                     label=f"{self.radm.R[i]}m")
    #     else:
    #         exit()

    #     ax.set_xlabel("Time [s]")
    #     ax.set_ylabel("Te ECE [keV]")
    #     fig.legend()

    #     plot.caption(fig, self.figtitle)
    #     plot.capsave(fig, self.figtitle, fnm, path)
    #     plot.check(pause)
    """
    
    """
    # def plot(self, R_in=3.8, R_out=4.4, dR=0.1, iscalib=True, pause=0):

    #     plot.set("notebook", "ticks")

    #     figdir = os.path.join(self.dirbase, "plot")
    #     proc.ifNotMake(figdir)
    #     fnm = self.fnm_base + f"{R_in}_{R_out}_{dR}.png"
    #     path = os.path.join(figdir, fnm)

    #     fig, ax = plt.subplots(1)
    #     R_arr = np.round(np.arange(R_in, R_out + dR, dR), 4)
    #     if iscalib:
    #         for R in R_arr:
    #             self.Radius_at(R, iscalib=iscalib)
    #             ax.errorbar(self.t, self.Rat.Te_tscal, self.Rat.Te_tscal_err,
    #                         ecolor="lightgrey", label=f"{self.Rat.R}m")
    #     else:
    #         for R in np.arange(R_in, R_out + dR, dR):
    #             self.Radius_at(R, iscalib=iscalib)
    #             ax.plot(self.t, self.Rat.Te,
    #                     label=f"{self.Rat.R}m")

    #     ax.set_xlabel("Time [s]")
    #     if iscalib:
    #         ax.set_ylabel("Te ECE [keV] tscalib")
    #     else:
    #         ax.set_ylabel("Te ECE [keV]")
    #     ax.legend()

    #     plot.caption(fig, self.figtitle)
    #     plot.capsave(fig, self.figtitle, fnm, path)
    #     plot.check(pause)
    """

    def R_window(self, Rat=4.0, dR=0.106, include_outerside=False):

        self.Rwin = struct()
        self.Rwin.Rat = Rat
        self.Rwin.dR = dR
        self.Rwin.Rin = Rat - 0.5 * dR
        self.Rwin.Rout = Rat + 0.5 * dR

        datlist = [self.Te_tscal, self.Te_tscal_err]
        Ridxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.R, x_start=self.Rwin.Rin, x_end=self.Rwin.Rout,
                                                                Ys_list=datlist, include_outerside=include_outerside)
        self.Rwin.Te_tscal, self.Rwin.Te_tscal_err = datlist_win
        self.Rwin.R = self.R[Ridxs]
        self.Rwin.rho_vacuum = self.rho_vacuum[Ridxs]
        self.Rwin.reffa99 = self.reffa99[Ridxs]

        self.Rwin.reffa99in = self.Rwin.reffa99[0]
        self.Rwin.reffa99out = self.Rwin.reffa99[-1]

        self.Rwin.avg = struct()
        self.Rwin.std = struct()
        self.Rwin.ste = struct()

        self.Rwin.avg.Te_tscal, self.Rwin.std.Te_tscal, self.Rwin.ste.Te_tscal \
            = calc.average(self.Rwin.Te_tscal, err=self.Rwin.Te_tscal_err, axis=1)

        return self.Rwin

class gas_puf:

    def __init__(self, sn=187979, sub=1, tstart=3., tend=6.):

        self.diagname = "gas_puf"
        self.sn = sn
        self.sub = sub
        self.ts = tstart
        self.te = tend

        self.t, list_dat, self.list_dimnms, self.list_valnms, self.list_dimunits, self.list_valunits \
            = read.eg1d(diagnm=self.diagname, sn=sn, sub=sub)

        tidxs, list_dat = proc.getTimeIdxsAndDats(self.t, self.ts, self.te, list_dat)

        self.t = self.t[tidxs]
        self.Ar35Lm, self.H235Ll, self.He35Ll, self.D235Ll, self.Ne95Lm, self.N235Lm, self.X35Lm, self.Ar55Ls, \
        self.H255Lm, self.He55Lm, self.D255Lm, self.Ne55Ls, self.N255Ls, self.X55Ls, self.Ar95Ls, self.H295Ls, \
        self.He95Ls, self.D295Ls, self.Ne95Ls, self.N295Ls = list_dat

        self.Ar = self.Ar35Lm + self.Ar55Ls + self.Ar95Ls
        self.H2 = self.H235Ll + self.H255Lm + self.H295Ls
        self.He = self.He35Ll + self.He55Lm + self.He55Lm
        self.D2 = self.D235Ll + self.D255Lm + self.D295Ls
        self.N2 = self.N235Lm + self.N255Ls + self.N295Ls
        self.Ne = self.Ne95Lm + self.Ne55Ls + self.Ne95Ls
        self.X = self.X35Lm + self.X55Ls

    def plot(self, axes, species=["Ar", "H2", "He", "D2", "N2", "Ne", "X"]):

        for s in species:
            if s == "Ar":
                axes.plot(self.t, self.Ar, c="gold", label=s)
            elif s == "H2":
                axes.plot(self.t, self.H2, c="magenta", label=s)
            elif s == "He":
                axes.plot(self.t, self.He, c="dodgerblue", label=s)
            elif s == "D2":
                axes.plot(self.t, self.D2, c="limegreen", label=s)
            elif s == "N2":
                axes.plot(self.t, self.N2, c="lightpink", label=s)
            elif s == "Ne":
                axes.plot(self.t, self.Ne, c="lightblue", label=s)
            elif s == "X":
                axes.plot(self.t, self.X, c="black", label=s)
        axes.legend()

class pci_ch4_fft:

    def __init__(self, sn, sub, tstart, tend, sampling="slow"):
        # sampling: "fast", "slow"

        self.sn = sn
        self.sub = sub
        self.ts = tstart
        self.te = tend
        self.sampling = sampling

        self.diagname = 'pci_ch4_fft'
        self.diagname += f"_{self.sampling}"
        self.pciEg = myEgdb.LoadEG(self.diagname, self.sn, self.sub)

        self.t, list_dat, self.dimnms, self.valnms, self.dimunits, self.valunits \
            = read.eg1d(self.diagname, self.sn, self.sub)

        tidxs, list_dat = proc.getTimeIdxsAndDats(self.t, self.ts, self.te, list_dat)
        self.t = self.t[tidxs]

        if sampling == "fast":
            self.midf, self.highf, self.allf = list_dat
        elif sampling == "slow":
            self.lowf, self.midf, self.highf, self.allf = list_dat

    def ref_to_fir_nel(self, Rfir=4.1):

        Rfirs = np.array([3.309, 3.399, 3.489, 3.579,
                          3.669, 3.759, 3.849, 3.939,
                          4.029, 4.119, 4.209, 4.299, 4.389])

        self.fir = fir_nel(self.sn, self.sub, self.ts, self.te)
        self.fir.ref_to(self.t)
        idx_dat = np.argmin(np.abs(Rfirs - Rfir))
        datlist = [self.fir.ref.avg.nl3309, self.fir.ref.avg.nl3399, self.fir.ref.avg.nl3489, self.fir.ref.avg.nl3579,
                   self.fir.ref.avg.nl3669, self.fir.ref.avg.nl3759, self.fir.ref.avg.nl3849, self.fir.ref.avg.nl3939,
                   self.fir.ref.avg.nl4029, self.fir.ref.avg.nl4119, self.fir.ref.avg.nl4209, self.fir.ref.avg.nl4299,
                   self.fir.ref.avg.nl4389]
        self.nel = datlist[idx_dat]

        if self.sampling == "fast":
            self.midf_nel = self.midf / self.nel
            self.highf_nel = self.highf / self.nel
            self.allf_nel = self.allf / self.nel
        elif self.sampling == "slow":
            self.lowf_nel = self.lowf / self.nel
            self.midf_nel = self.midf / self.nel
            self.highf_nel = self.highf / self.nel
            self.allf_nel = self.allf / self.nel
