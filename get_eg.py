import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import os
import gc

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from nasu.myEgdb import LoadEG
from nasu import read, calc, getShotInfo, proc


ee = 1.602176634  # [10^-19 C]


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


def wavenumber_combR(egcombR, freq, tstart, tend):

    wavenm = 'k_perp each time  ' + freq
    dwavenm = 'delta k_perp error estimation  ' + freq

    combRtime = egcombR.dims(0)
    idx_time = np.where((combRtime >= tstart) & (combRtime <= tend))
    combRtime = combRtime[idx_time]

    wavenum_combR = egcombR.trace_of(wavenm, dim=0, other_idxs=[0])
    wavenum_combR = wavenum_combR[idx_time]
    dwavenum_combR = egcombR.trace_of(dwavenm, dim=0, other_idxs=[0])
    dwavenum_combR = dwavenum_combR[idx_time]

    wavenum_combR_avg = np.average(wavenum_combR)
    wavenum_combR_err = np.sqrt(np.sum((dwavenum_combR / 2)**2 + (wavenum_combR - wavenum_combR_avg)**2)/(len(wavenum_combR) - 1))

    return combRtime, wavenum_combR, dwavenum_combR, wavenum_combR_avg, wavenum_combR_err


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

    def __init__(self, sn=184508, sub=1):

        self.sn = sn
        self.sub = sub

        EG = LoadEG(diagname="tsmap_calib", sn=sn, sub=sub)

        self.t = EG.dims(0)
        self.R = EG.dims(1)

        self.reff = EG.trace_of_2d('reff', [0, 1])
        self.reffa99 = EG.trace_of_2d('reff/a99', [0, 1])
        self.Te = EG.trace_of_2d('Te', [0, 1])
        self.dTe = EG.trace_of_2d('dTe', [0, 1])
        self.ne_calFIR = EG.trace_of_2d('ne_calFIR', [0, 1])
        self.dne_calFIR = EG.trace_of_2d('dne_calFIR', [0, 1])
        self.Te_fit = EG.trace_of_2d('Te_fit', [0, 1])
        self.Te_fit_err = EG.trace_of_2d('Te_fit_err', [0, 1])
        self.ne_fit = EG.trace_of_2d('ne_fit', [0, 1])
        self.ne_fit_err = EG.trace_of_2d('ne_fit_err', [0, 1])
        self.Br = EG.trace_of_2d('Br', [0, 1])
        self.Bz = EG.trace_of_2d('Bz', [0, 1])
        self.Bphi = EG.trace_of_2d('Bphi', [0, 1])

        self.reffa99[self.reffa99 > 1.5] = np.nan
        self.Te[self.Te == 0.] = np.nan
        self.dTe[self.dTe == 0.] = np.nan
        self.ne_calFIR[self.ne_calFIR == 0.] = np.nan
        self.dne_calFIR[self.dne_calFIR == 0.] = np.nan
        self.Te_fit[self.Te_fit == 0.] = np.nan
        self.Te_fit_err[self.Te_fit_err == 0.] = np.nan
        self.ne_fit[self.ne_fit == 0.] = np.nan
        self.ne_fit_err[self.ne_fit_err == 0.] = np.nan  # 重み付き平均のときのエラーを消すため。
        self.Br[self.Br == 0.] = np.nan
        self.Bz[self.Bz == 0.] = np.nan
        self.Bphi[self.Bphi == 0.] = np.nan

        self.reff = np.reshape(self.reff, EG.dimsize)
        self.reffa99 = np.reshape(self.reffa99, EG.dimsize)
        self.Te = np.reshape(self.Te, EG.dimsize)
        self.dTe = np.reshape(self.dTe, EG.dimsize)
        self.ne_calFIR = np.reshape(self.ne_calFIR, EG.dimsize)
        self.dne_calFIR = np.reshape(self.dne_calFIR, EG.dimsize)
        self.Te_fit = np.reshape(self.Te_fit, EG.dimsize)
        self.Te_fit_err = np.reshape(self.Te_fit_err, EG.dimsize)
        self.ne_fit = np.reshape(self.ne_fit, EG.dimsize)
        self.ne_fit_err = np.reshape(self.ne_fit_err, EG.dimsize)
        self.Br = np.reshape(self.Br, EG.dimsize)
        self.Bz = np.reshape(self.Bz, EG.dimsize)
        self.Bphi = np.reshape(self.Bphi, EG.dimsize)

        self.Bax, self.Rax, self.Bq, self.gamma, self.datetime, self.cycle = getShotInfo.info(self.sn)
        coef = np.abs(self.Bax / 3)
        self.Br *= coef
        self.Bz *= coef
        self.Bphi *= coef

        self.B = np.sqrt(self.Br ** 2 + self.Bz ** 2 + self.Bphi ** 2)

        self.pe = ee * self.Te * self.ne_calFIR  # [kPa]
        self.pe_err = np.sqrt((self.dTe/self.Te)**2 + (self.dne_calFIR/self.ne_calFIR)**2) * self.pe
        self.pe_fit = ee * self.Te_fit * self.ne_fit  # [kPa]
        self.pe_fit_err = np.sqrt((self.Te_fit_err / self.Te_fit) ** 2 +
                                  (self.ne_fit_err / self.ne_fit) ** 2) * self.pe_fit

    def tat(self, time=4.5):

        self.at = struct()
        datlist = [self.t, self.reff, self.reffa99, self.Te, self.dTe, self.ne_calFIR, self.dne_calFIR,
                   self.Te_fit, self.Te_fit_err, self.ne_fit, self.ne_fit_err, self.Br, self.Bz, self.Bphi, self.B]
        _, datlist_at = proc.getTimeIdxAndDats(self.t, time, datlist)
        self.at.t, self.at.reff, self.at.reffa99, \
        self.at.Te, self.at.dTe, self.at.ne_calFIR, self.at.dne_calFIR, \
        self.at.Te_fit, self.at.Te_fit_err, self.at.ne_fit, self.at.ne_fit_err, \
        self.at.Br, self.at.Bz, self.at.Bphi, self.at.B = datlist_at

    def plot_reffa99(self, tstart=3, tend=6, Rmin=3.5, Rmax=4.7, rhomin=0, rhomax=1.2, drho=0.1):
        levels = np.arange(rhomin, rhomax+drho, drho)
        tg, Rg = np.meshgrid(self.t, self.R)
        fnm=f"#{self.sn}-{self.sub}_{tstart}-{tend}s_{Rmin}-{Rmax}m"
        plt.figure(num=fnm)
        cp = plt.contour(tg, Rg, self.reffa99.T, levels=levels)
        plt.clabel(cp, inline=True, fontsize=10)
        plt.title('reff/a99')
        plt.xlabel('Time [s]')
        plt.ylabel('R [m]')
        plt.xlim(tstart, tend)
        plt.ylim(Rmin, Rmax)
        plt.show()

    def time_window(self, tstart=4.4, tend=4.5):

        self.twin = struct()
        self.twin.tstart = tstart
        self.twin.tend = tend
        datlist = [self.t, self.reff, self.reffa99, self.Te, self.dTe, self.ne_calFIR, self.dne_calFIR,
                   self.Te_fit, self.Te_fit_err, self.ne_fit, self.ne_fit_err, self.Br, self.Bz, self.Bphi, self.B,
                   self.pe, self.pe_err, self.pe_fit, self.pe_fit_err]
        _, datlist_win = proc.getTimeIdxsAndDats(time=self.t, startTime=tstart, endTime=tend, datList=datlist)
        self.twin.t, self.twin.reff, self.twin.reffa99, \
        self.twin.Te, self.twin.dTe, self.twin.ne_calFIR, self.twin.dne_calFIR, \
        self.twin.Te_fit, self.twin.Te_fit_err, self.twin.ne_fit, self.twin.ne_fit_err, \
        self.twin.Br, self.twin.Bz, self.twin.Bphi, self.twin.B, \
        self.twin.pe, self.twin.pe_err, self.twin.pe_fit, self.twin.pe_fit_err = datlist_win

        self.twin.avg = struct()
        self.twin.std = struct()
        self.twin.ste = struct()

        self.twin.avg.reff, self.twin.std.reff, self.twin.ste.reff \
            = calc.average(self.twin.reff, err=None, axis=0)
        self.twin.avg.reffa99, self.twin.std.reffa99, self.twin.ste.reffa99 \
            = calc.average(self.twin.reffa99, err=None, axis=0)
        self.twin.avg.Te, self.twin.std.Te, self.twin.ste.Te \
            = calc.average(self.twin.Te, err=self.twin.dTe, axis=0)
        self.twin.avg.ne_calFIR, self.twin.std.ne_calFIR, self.twin.ste.ne_calFIR \
            = calc.average(self.twin.ne_calFIR, err=self.twin.dne_calFIR, axis=0)
        self.twin.avg.Te_fit, self.twin.std.Te_fit, self.twin.ste.Te_fit \
            = calc.average(self.twin.Te_fit, err=self.twin.Te_fit_err, axis=0)
        self.twin.avg.ne_fit, self.twin.std.ne_fit, self.twin.ste.ne_fit \
            = calc.average(self.twin.ne_fit, err=self.twin.ne_fit_err, axis=0)
        self.twin.avg.Br, self.twin.std.Br, self.twin.ste.Br \
            = calc.average(self.twin.Br, err=None, axis=0)
        self.twin.avg.Bz, self.twin.std.Bz, self.twin.ste.Bz \
            = calc.average(self.twin.Bz, err=None, axis=0)
        self.twin.avg.Bphi, self.twin.std.Bphi, self.twin.ste.Bphi \
            = calc.average(self.twin.Bphi, err=None, axis=0)
        self.twin.avg.B, self.twin.std.B, self.twin.ste.B \
            = calc.average(self.twin.B, err=None, axis=0)
        self.twin.avg.pe, self.twin.std.pe, self.twin.ste.pe \
            = calc.average(self.twin.pe, err=self.twin.pe_err, axis=0)
        self.twin.avg.pe_fit, self.twin.std.pe_fit, self.twin.ste.pe_fit \
            = calc.average(self.twin.pe_fit, err=self.twin.pe_fit_err, axis=0)

    def R_window(self, Rat=4.1, dR=0.106, include_outerside=False, include_grad=False):
        # R平均の処理
        # tsmapで取得したデータの指定したR区間内での平均。
        # 勾配も平均処理するかどうかはオプションとする。（勾配も平均する場合は、先に勾配を計算しておく。）
        # reff/a99とRの対応関係を別で確認してからRの範囲を決める。

        self.Rwin = struct()
        self.Rwin.Rat = Rat
        self.Rwin.dR = dR
        self.Rwin.Rin = Rat - 0.5 * dR
        self.Rwin.Rout = Rat + 0.5 * dR

        datlist = [self.reff, self.reffa99, self.Te, self.dTe, self.ne_calFIR, self.dne_calFIR,
                   self.Te_fit, self.Te_fit_err, self.ne_fit, self.ne_fit_err, self.Br, self.Bz, self.Bphi, self.B,
                   self.pe, self.pe_err, self.pe_fit, self.pe_fit_err]
        _idxs, datlist_win = proc.getXIdxsAndYs_2dalongLastAxis(xx=self.R, x_start=self.Rwin.Rin, x_end=self.Rwin.Rout,
                                                                Ys_list=datlist, include_outerside=include_outerside)
        self.Rwin.R = self.R[_idxs]
        self.Rwin.reff, self.Rwin.reffa99, \
        self.Rwin.Te, self.Rwin.dTe, self.Rwin.ne_calFIR, self.Rwin.dne_calFIR, \
        self.Rwin.Te_fit, self.Rwin.Te_fit_err, self.Rwin.ne_fit, self.Rwin.ne_fit_err, \
        self.Rwin.Br, self.Rwin.Bz, self.Rwin.Bphi, self.Rwin.B, \
        self.Rwin.pe, self.Rwin.pe_err, self.Rwin.pe_fit, self.Rwin.pe_fit_err = datlist_win

        self.Rwin.reffin = np.ravel(self.Rwin.reff[:, 0])
        self.Rwin.reffout = np.ravel(self.Rwin.reff[:, -1])
        self.Rwin.reffa99in = np.ravel(self.Rwin.reffa99[:, 0])
        self.Rwin.reffa99out = np.ravel(self.Rwin.reffa99[:, -1])

        self.Rwin.avg = struct()
        self.Rwin.std = struct()
        self.Rwin.ste = struct()

        self.Rwin.avg.reff, self.Rwin.std.reff, self.Rwin.ste.reff \
            = calc.average(self.Rwin.reff, err=None, axis=1)
        self.Rwin.avg.reffa99, self.Rwin.std.reffa99, self.Rwin.ste.reffa99 \
            = calc.average(self.Rwin.reffa99, err=None, axis=1)
        self.Rwin.avg.Te, self.Rwin.std.Te, self.Rwin.ste.Te \
            = calc.average(self.Rwin.Te, err=self.Rwin.dTe, axis=1)
        self.Rwin.avg.ne_calFIR, self.Rwin.std.ne_calFIR, self.Rwin.ste.ne_calFIR \
            = calc.average(self.Rwin.ne_calFIR, err=self.Rwin.dne_calFIR, axis=1)
        self.Rwin.avg.Te_fit, self.Rwin.std.Te_fit, self.Rwin.ste.Te_fit \
            = calc.average(self.Rwin.Te_fit, err=self.Rwin.Te_fit_err, axis=1)
        self.Rwin.avg.ne_fit, self.Rwin.std.ne_fit, self.Rwin.ste.ne_fit \
            = calc.average(self.Rwin.ne_fit, err=self.Rwin.ne_fit_err, axis=1)
        self.Rwin.avg.Br, self.Rwin.std.Br, self.Rwin.ste.Br \
            = calc.average(self.Rwin.Br, err=None, axis=1)
        self.Rwin.avg.Bz, self.Rwin.std.Bz, self.Rwin.ste.Bz \
            = calc.average(self.Rwin.Bz, err=None, axis=1)
        self.Rwin.avg.Bphi, self.Rwin.std.Bphi, self.Rwin.ste.Bphi \
            = calc.average(self.Rwin.Bphi, err=None, axis=1)
        self.Rwin.avg.B, self.Rwin.std.B, self.Rwin.ste.B \
            = calc.average(self.Rwin.B, err=None, axis=1)
        self.Rwin.avg.pe, self.Rwin.std.pe, self.Rwin.ste.pe \
            = calc.average(self.Rwin.pe, err=self.Rwin.pe_err, axis=1)
        self.Rwin.avg.pe_fit, self.Rwin.std.pe_fit, self.Rwin.ste.pe_fit \
            = calc.average(self.Rwin.pe_fit, err=self.Rwin.pe_fit_err, axis=1)

    # Te, or ne_calFIR 分布データの処理
    # ※ 勾配計算は, 物理量をMとして, dM/dR と　dreff/dRをそれぞれ計算し、(dM/dR) / (dreff/dR) で dM/dreffを計算するのが良い。
    #    （データ仕様の都合上）
    # a. 生データのまま
    #   1) 局所直線フィッティングで勾配をそのまま計算する。スケール長計算時のTe or neの値には0次項を使う。
    #   2) 全点n次多項式フィッティングを行う。導関数から勾配を計算する。
    # b. Te_fit or ne_fit を使う。
    #   1) 2次中心差分で勾配を計算する。
    #   2) 5点ステンシル中心差分で勾配を計算する。
    # c. 平均処理（時間平均・ショット平均・空間平均（移動平均）などで処理）してばらつきを小さくしたデータを使う。
    #   1) 局所直線フィッティングで勾配を計算する。スケール長計算時のTe or neの値には0次項を使う。
    #   2) 全点n次多項式フィッティングを行う。導関数から勾配を計算する。
    # a., b.はEGデータからそのまま計算できるので、自動で全て行う。
    # c.はオプションとして実装しておく。平均処理した後に行えるように。

class cxsmap7:

    def __init__(self, sn=184508, sub=1):

        self.sn = sn
        self.sub = sub

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
        self.tor.reff[np.abs(self.tor.reff) > 1.5] = np.nan
        self.pol.reff[np.abs(self.pol.reff) > 1.5] = np.nan

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
        Rpolidxs = ~np.isnan(varlist_pol[7]).all(axis=0)
        Rtoridxs = ~np.isnan(varlist_tor[7]).all(axis=0)
        self.t = self.t[tidxs]
        self.pol.R = self.pol.R[Rpolidxs]
        self.tor.R = self.tor.R[Rtoridxs]

        for i in range(len(varlist_pol)):
            varlist_pol[i] = varlist_pol[i][tidxs]
            varlist_pol[i] = varlist_pol[i][:, Rpolidxs]

        for i in range(len(varlist_tor)):
            varlist_tor[i] = varlist_tor[i][tidxs]
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

        self.pol.TeTi = self.pol.Te / self.pol.Ti
        self.tor.TeTi = self.tor.Te / self.tor.Ti

    def tat(self, time=4.5):

        self.at = struct()
        self.at.pol = struct()
        self.at.tor = struct()
        datlist = [self.t,
                   self.pol.Ti, self.pol.Tier, self.pol.Vc, self.pol.Ver, self.pol.inc, self.pol.icer,
                   self.pol.Vr, self.pol.reff, self.pol.a99,
                   self.pol.p0, self.pol.pf, self.pol.ip, self.pol.ipf,
                   self.pol.Br, self.pol.Bz, self.pol.Bphi, self.pol.dVdreff,
                   self.pol.Te, self.pol.ne, self.pol.t1,
                   self.pol.reffa99, self.pol.TeTi,
                   self.tor.Ti, self.tor.Tier, self.tor.Vc, self.tor.Ver, self.tor.inc, self.tor.icer,
                   self.tor.Vr, self.tor.reff, self.tor.a99,
                   self.tor.p0, self.tor.pf, self.tor.ip, self.tor.ipf,
                   self.tor.Br, self.tor.Bz, self.tor.Bphi, self.tor.dVdreff,
                   self.tor.Te, self.tor.ne, self.tor.t1,
                   self.tor.reffa99, self.tor.TeTi]

        _, datlist_at = proc.getTimeIdxAndDats(self.t, time, datlist)
        self.at.t, \
        self.at.pol.Ti, self.at.pol.Tier, self.at.pol.Vc, self.at.pol.Ver, self.at.pol.inc, self.at.pol.icer, \
        self.at.pol.Vr, self.at.pol.reff, self.at.pol.a99, \
        self.at.pol.p0, self.at.pol.pf, self.at.pol.ip, self.at.pol.ipf, \
        self.at.pol.Br, self.at.pol.Bz, self.at.pol.Bphi, self.at.pol.dVdreff, \
        self.at.pol.Te, self.at.pol.ne, self.at.pol.t1, \
        self.at.pol.reffa99, self.at.pol.TeTi, \
        self.at.tor.Ti, self.at.tor.Tier, self.at.tor.Vc, self.at.tor.Ver, self.at.tor.inc, self.at.tor.icer, \
        self.at.tor.Vr, self.at.tor.reff, self.at.tor.a99, \
        self.at.tor.p0, self.at.tor.pf, self.at.tor.ip, self.at.tor.ipf, \
        self.at.tor.Br, self.at.tor.Bz, self.at.tor.Bphi, self.at.tor.dVdreff, \
        self.at.tor.Te, self.at.tor.ne, self.at.tor.t1, \
        self.at.tor.reffa99, self.at.tor.TeTi = datlist_at
