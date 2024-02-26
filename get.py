import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

from nasu.myEgdb import LoadEG
from nasu import read, calc


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
