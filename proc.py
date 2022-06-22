import numpy as np
from scipy.interpolate import interp1d
from scipy import signal, fft

import gc


def decimate_timeSeries(ti_raw, sr_raw, dt_dec):
    size_raw = ti_raw.size
    if size_raw != sr_raw.size:
        print('Series data size is different from time data size.')
        print(exit())
    dt_raw = ti_raw[1] - ti_raw[0]
    Ndec = int(dt_dec / dt_raw + 0.5)
    idxs_dec = np.arange(0, size_raw, Ndec)
    ti_dec = ti_raw[idxs_dec]
    sr_dec = sr_raw[idxs_dec]
    return ti_dec, sr_dec


def takeat_rho(dat, err, time, rho, rho_at):

    idxs_at = (range(time.size), np.nanargmin(np.abs(rho - rho_at), axis=1))
    dat_at = dat[idxs_at]
    err_at = err[idxs_at]

    idxs_del = np.where((rho_at > np.nanmax(rho, axis=1)) | (rho_at < np.nanmin(rho, axis=1)))[0]
    dat_at[idxs_del] = np.nan
    err_at[idxs_del] = np.nan

    return dat_at, err_at


def takeat_rho_R(R, rho, rho_at):

    idxs_at = np.nanargmin(np.abs(rho - rho_at), axis=1)
    R_at = R[idxs_at]

    idxs_del = np.where((rho_at > np.nanmax(rho, axis=1)) | (rho_at < np.nanmin(rho, axis=1)))[0]
    R_at[idxs_del] = np.nan

    return R_at


def takeat_tlist(dat, err, R, reff, rho, time, tlist):
    idxs_tlist = [np.nanargmin(np.abs(time - t)) for t in tlist]
    time_tl = time[idxs_tlist]
    R_tl = R[idxs_tlist]
    reff_tl = reff[idxs_tlist]
    rho_tl = rho[idxs_tlist]
    dat_tl = dat[idxs_tlist]
    err_tl = err[idxs_tlist]
    return time_tl, R_tl, reff_tl, rho_tl, dat_tl, err_tl


def interpolate1d(time, val, err, t_ref):

    f = interp1d(time, val, bounds_error=False, fill_value=np.nan)
    fer = interp1d(time, err, bounds_error=False, fill_value=np.nan)
    val_intp = f(t_ref)
    err_intp = fer(t_ref)

    return val_intp, err_intp


def adjustVt_comb2(ch, sn, Vt0):

    Vt = 0

    if ch == '29.1G':
        if sn >= 177865:
            Vt = 1.1
    elif ch == '32.0G':
        if sn >= 177865:
            Vt = 1.2
    elif ch == '33.4G':
        if sn >= 176306 and sn < 177866:
            Vt = 1.2
        elif sn >= 177866:
            Vt = 1.0
    elif ch == '34.8G':
        if sn >= 176308 and sn < 177866:
            Vt = 1.2
        elif sn >= 177866:
            Vt = 1.0
    elif ch == '36.9G':
        if sn >= 176307 and sn < 177866:
            Vt = 1.2
        elif sn >= 177866:
            Vt = 1.0
    else:
        Vt = Vt0

    print(f'Vt was changed from {Vt0:.1f} to {Vt:.1f}\n')

    return Vt


def outlier_Sk(dat_Sk, diag, chIQ, path_mod, time_nb, dat_nb5a, time_fDSk):

    if diag == 'MWRM-PXI' and chIQ[0] == 1:
        BSmod = np.loadtxt(path_mod, delimiter=',').T
        time_mod, dat_mod = BSmod
        idx_del = np.where(dat_mod < 0.75)[0]
        dat_Sk[idx_del] = np.nan
    elif diag == 'MWRM-COMB':
        dat_nb5a_fDSk = interp1d(time_nb, dat_nb5a)(time_fDSk)
        dat_diff_nb5a = np.diff(dat_nb5a_fDSk)
        idx_del = np.where(dat_diff_nb5a < -2)[0]
        dat_Sk[idx_del] = np.nan

    return dat_Sk
