import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

from nasu.myEgdb import LoadEG
from nasu import read, calc


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

    del idx_time, eghk

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


def rho_comb_R(egcombR, freq, tstart, tend):  # Only mwrm_comb_R_Vp is available from 2022/1/18.

    valnm = 'reff/a99' + '  ' + freq

    combRtime = egcombR.dims(0)
    idx_time = np.where((combRtime >= tstart) & (combRtime <= tend))
    combRtime = combRtime[idx_time]

    rhocombR = egcombR.trace_of(valnm, dim=0, other_idxs=[0])
    rhocombR = rhocombR[idx_time]

    rhocombR_avg = rhocombR.mean()
    rhocombR_std = rhocombR.std()

    return combRtime, rhocombR, rhocombR_avg, rhocombR_std


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
    ech_tot = 'Total ECH'

    egech = LoadEG(diagech, sn)
    echtime = egech.dims(0)
    echpw = egech.trace_of(ech_tot, dim=0, other_idxs=[0])

    return echtime, echpw