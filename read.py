import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
from parse import parse
from nasu.myEgdb import LoadEG
import nasu.LHDRetrieve as LHDR
from nasu import getShotInfo, proc, myEgdb
import inspect


class struct:
    pass


def inputSnOnTerminal():
    snstart = int(input('Start Shot # >>> '))
    snend = int(input('End Shot (if take only one shot, enter 0) # >>> '))
    if snend == 0:
        snend = snstart
    return snstart, snend


def colorLabelFdSk(idx_dev, ch, frangeSk_k):
    label_fD = 'Doppler Shift [kHz]'
    if idx_dev == 0:
        color_fDSk = 'red'
        label_Sk = f'$I_{{amp,BS(3-O),ch{ch:d}}}$\n' \
                   f'{frangeSk_k[0]:d}-{frangeSk_k[1]:d} kHz (a.u.)'
    elif idx_dev == 1:
        color_fDSk = 'blue'
        label_Sk = f'$I_{{amp,DBS(3-O),{ch:s}}}$\n' \
                   f'{frangeSk_k[0]:d}-{frangeSk_k[1]:d} kHz (a.u.)'
    elif idx_dev == 2:
        color_fDSk = 'green'
        label_Sk = f'$I_{{amp,DBS(9-O),{ch:s}}}$\n' \
                   f'{frangeSk_k[0]:d}-{frangeSk_k[1]:d} kHz (a.u.)'
    else:
        exit()

    return color_fDSk, label_Sk, label_fD


def tsFromLocal(filepath):

    datDf = pd.read_csv(filepath)
    columns = datDf.columns.values
    print(columns)
    time_ts = datDf[columns[0]].unique()
    R_ts = datDf[columns[1]].unique()
    datDf = datDf.pivot(columns[0], columns[1])
    reff_ts = datDf[columns[2]].to_numpy()
    rho_ts = datDf[columns[3]].to_numpy()
    dat_Te = datDf[columns[4]].to_numpy()
    err_Te = datDf[columns[5]].to_numpy()
    dat_ne = datDf[columns[6]].to_numpy()
    err_ne = datDf[columns[7]].to_numpy()

    return time_ts, R_ts, reff_ts, rho_ts, dat_Te, err_Te, dat_ne, err_ne


def cxs7FromLocal(filepath):

    datDf = pd.read_csv(filepath)
    columns = datDf.columns.values
    print(columns)
    time_cxs7 = datDf[columns[0]].unique()
    R_cxs7 = datDf[columns[1]].unique()
    datDf = datDf.pivot(columns[0], columns[1])
    reff_cxs7 = datDf[columns[2]].to_numpy()
    rho_cxs7 = datDf[columns[3]].to_numpy()
    dat_Ti = datDf[columns[4]].to_numpy()
    err_Ti = datDf[columns[5]].to_numpy()

    return time_cxs7, R_cxs7, reff_cxs7, rho_cxs7, dat_Ti, err_Ti


def ece(EG):
    time = EG.dims(0)
    R = EG.dims(1)
    shape_tR = (time.size, R.size)

    Te = EG.trace_of_2d('Te', [0, 1])
    Te[Te <= 0.] = np.nan
    Te = np.reshape(Te, shape_tR)

    idxs_sort = np.argsort(R)
    R = R[idxs_sort]
    Te = Te[:, idxs_sort]

    return time, R, Te


def fftSetting(inputfile):
    Nfft_pw, Nfft, window, Nens, OVR, NOV = input_FFT(inputfile)
    frangefD_k, frangeSk_k = input_fDSk(inputfile)
    input_df = pd.read_csv(inputfile, header=None, index_col=0)
    flim_k = int(input_df.at['flimk', 1])
    return Nfft_pw, Nfft, window, Nens, OVR, NOV, frangefD_k, frangeSk_k, flim_k


def ei_setting(idx_dev, ch, sn):

    dT = dT_Retrieve(idx_dev, ch, sn)

    if idx_dev == 0:
        color_fDSk = 'red'
        label_Sk = f'$I_{{amp,BS(3-O),ch{ch:d}}}$\n(a.u.)'
    elif idx_dev == 1:
        color_fDSk = 'blue'
        label_Sk = f'$I_{{amp,DBS(3-O),{ch:s}}}$\n(a.u.)'
    elif idx_dev == 2:
        color_fDSk = 'green'
        label_Sk = f'$I_{{amp,DBS(9-O),{ch:s}}}$\n(a.u.)'

    return dT, color_fDSk, label_Sk


def dT_Retrieve(idx_dev, ch, sn):

    if idx_dev == 0 and ch == 3 and sn <= 179616:
        dT = '1e-06'
    else:
        dT = '4e-07'

    return dT


def dirs():

    cwd = os.getcwd()
    files = os.listdir(cwd)
    dirs = [os.path.join(cwd, d) for d in files
            if os.path.isdir(os.path.join(cwd, d)) and d != '.idea' and d != '__pycache__' and d != '.git']

    return dirs


def calibPrms_df(path_calibPrms):
    idxs_calibPrms = ['LO freq. (MHz)', 'LO level (dBm)', 'LO VGA (V)', 'RF VGA (V)']
    calibPrms_df = pd.read_csv(path_calibPrms, header=0, index_col=idxs_calibPrms)

    return calibPrms_df


def calibPrms_df_v2(path_calibPrms):
    idxs_calibPrms = ['LO freq. (MHz)']
    calibPrms_df = pd.read_csv(path_calibPrms, header=0, index_col=idxs_calibPrms)

    return calibPrms_df


def field_from_tsmap_calib(EG, sn):

    time = EG.dims(0)
    R = EG.dims(1)
    shape_tR = (time.size, R.size)

    reff = EG.trace_of_2d('reff', [0, 1])
    rho = EG.trace_of_2d('reff/a99', [0, 1])
    Br = EG.trace_of_2d('Br', [0, 1])
    Bz = EG.trace_of_2d('Bz', [0, 1])
    Bphi = EG.trace_of_2d('Bphi', [0, 1])

    rho[rho > 1.5] = np.nan
    Br[Br == 0.] = np.nan
    Bz[Bz == 0.] = np.nan
    Bphi[Bphi == 0.] = np.nan

    reff = np.reshape(reff, shape_tR)
    rho = np.reshape(rho, shape_tR)
    Br = np.reshape(Br, shape_tR)
    Bz = np.reshape(Bz, shape_tR)
    Bphi = np.reshape(Bphi, shape_tR)

    Bt = getShotInfo.info(sn)[0]
    coef = Bt/3
    Br = coef * Br
    Bz = coef * Bz
    Bphi = coef * Bphi

    B = np.sqrt(Br ** 2 + Bz ** 2 + Bphi ** 2)

    return time, R, reff, rho, B, Br, Bz, Bphi


def nTB_from_tsmap_calib(EG, sn):

    time, R, reff, rho, B, Br, Bz, Bphi = field_from_tsmap_calib(EG, sn)
    time, R, reff, rho, \
    dat_Te, err_Te, dat_ne, err_ne, \
    dat_Tefit, err_Tefit, dat_nefit, err_nefit = tsmap_calib(EG)

    return time, R, reff, rho, \
           dat_Te, err_Te, dat_ne, err_ne, \
           dat_Tefit, err_Tefit, dat_nefit, err_nefit, \
           B, Br, Bz, Bphi


def field_from_tsmap_reff(EG, sn):

    time = EG.dims(0)
    R = EG.dims(1)
    shape_tR = (time.size, R.size)

    reff = EG.trace_of_2d('reff', [0, 1])
    Br = EG.trace_of_2d('Br', [0, 1])
    Bz = EG.trace_of_2d('Bz', [0, 1])
    Bphi = EG.trace_of_2d('Bphi', [0, 1])

    Br[Br == 0.] = np.nan
    Bz[Bz == 0.] = np.nan
    Bphi[Bphi == 0.] = np.nan

    reff = np.reshape(reff, shape_tR)
    Br = np.reshape(Br, shape_tR)
    Bz = np.reshape(Bz, shape_tR)
    Bphi = np.reshape(Bphi, shape_tR)

    Bt = getShotInfo.info(sn)[0]
    coef = np.abs(Bt/3)
    Br = coef * Br
    Bz = coef * Bz
    Bphi = coef * Bphi

    B = np.sqrt(Br ** 2 + Bz ** 2 + Bphi ** 2)

    return time, R, reff, B, Br, Bz, Bphi


def tsmap_calib(EG):

    time = EG.dims(0)
    R = EG.dims(1)
    shape_tR = (time.size, R.size)

    reff = EG.trace_of_2d('reff', [0, 1])
    rho = EG.trace_of_2d('reff/a99', [0, 1])
    dat_Te = EG.trace_of_2d('Te', [0, 1])
    err_Te = EG.trace_of_2d('dTe', [0, 1])
    dat_ne = EG.trace_of_2d('ne_calFIR', [0, 1])
    err_ne = EG.trace_of_2d('dne_calFIR', [0, 1])
    dat_Tefit = EG.trace_of_2d('Te_fit', [0, 1])
    err_Tefit = EG.trace_of_2d('Te_fit_err', [0, 1])
    dat_nefit = EG.trace_of_2d('ne_fit', [0, 1])
    err_nefit = EG.trace_of_2d('ne_fit_err', [0, 1])

    rho[rho > 1.5] = np.nan
    dat_Te[dat_Te == 0.] = np.nan
    err_Te[err_Te == 0.] = np.nan
    dat_ne[dat_ne == 0.] = np.nan
    err_ne[err_ne == 0.] = np.nan
    dat_Tefit[dat_Tefit == 0.] = np.nan
    err_Tefit[err_Tefit == 0.] = np.nan
    dat_nefit[dat_nefit == 0.] = np.nan
    err_nefit[err_nefit == 0.] = np.nan

    reff = np.reshape(reff, shape_tR)
    rho = np.reshape(rho, shape_tR)
    dat_Te = np.reshape(dat_Te, shape_tR)
    err_Te = np.reshape(err_Te, shape_tR)
    dat_ne = np.reshape(dat_ne, shape_tR)
    err_ne = np.reshape(err_ne, shape_tR)
    dat_Tefit = np.reshape(dat_Tefit, shape_tR)
    err_Tefit = np.reshape(err_Tefit, shape_tR)
    dat_nefit = np.reshape(dat_nefit, shape_tR)
    err_nefit = np.reshape(err_nefit, shape_tR)

    return time, R, reff, rho, \
           dat_Te, err_Te, dat_ne, err_ne, \
           dat_Tefit, err_Tefit, dat_nefit, err_nefit


def cxsmap7(EG):

    proc.suggestNewVer(1, 'cxsmap7')

    time = EG.dims(0)
    R = EG.dims(1)
    shape_tR = (time.size, R.size)

    ary = EG.trace_of('ary', 1, [0])
    idxs_pol = np.where((ary == 1.) | (ary == 3.))[0]
    idxs_tor = np.where((ary == 5.) | (ary == 7.))[0]

    a99 = EG.trace_of_2d('a99', [0, 1])
    reff = EG.trace_of_2d('reff', [0, 1])
    a99[a99 == 0.] = np.nan
    reff[np.abs(reff) > 1.5] = np.nan
    a99 = a99.reshape(shape_tR)
    reff = reff.reshape(shape_tR)

    a99b = ~np.isnan(a99).all(axis=1)
    time = time[a99b]
    a99 = a99[a99b]
    reff = reff[a99b]

    dat_Ti = EG.trace_of_2d('Ti', [0, 1])
    err_Ti = EG.trace_of_2d('Tier', [0, 1])
    dat_Ti[dat_Ti == 0.] = np.nan
    err_Ti[err_Ti == 0.] = np.nan
    dat_Ti = dat_Ti.reshape(shape_tR)
    err_Ti = err_Ti.reshape(shape_tR)

    dat_Ti = dat_Ti[a99b]
    err_Ti = err_Ti[a99b]

    rho = reff / a99
    R_pol = R[idxs_pol]
    R_tor = R[idxs_tor]
    reff_pol = reff[:, idxs_pol]
    reff_tor = reff[:, idxs_tor]
    rho_pol = rho[:, idxs_pol]
    rho_tor = rho[:, idxs_tor]
    a99_pol = a99[:, idxs_pol]
    a99_tor = a99[:, idxs_tor]
    dat_Tipol = dat_Ti[:, idxs_pol]
    dat_Titor = dat_Ti[:, idxs_tor]
    err_Tipol = err_Ti[:, idxs_pol]
    err_Titor = err_Ti[:, idxs_tor]

    idxs_sort_pol = np.argsort(R_pol)
    idxs_sort_tor = np.argsort(R_tor)
    R_pol = R_pol[idxs_sort_pol]
    reff_pol = reff_pol[:, idxs_sort_pol]
    rho_pol = rho_pol[:, idxs_sort_pol]
    a99_pol = a99_pol[:, idxs_sort_pol]
    dat_Tipol = dat_Tipol[:, idxs_sort_pol]
    err_Tipol = err_Tipol[:, idxs_sort_pol]
    R_tor = R_tor[idxs_sort_tor]
    reff_tor = reff_tor[:, idxs_sort_tor]
    rho_tor = rho_tor[:, idxs_sort_tor]
    a99_tor = a99_tor[:, idxs_sort_tor]
    dat_Titor = dat_Titor[:, idxs_sort_tor]
    err_Titor = err_Titor[:, idxs_sort_tor]

    return time, R_pol, R_tor, \
           reff_pol, reff_tor, rho_pol, rho_tor, a99_pol, a99_tor, \
           dat_Tipol, err_Tipol, dat_Titor, err_Titor


def cxsmap7_v1(EG):

    time = EG.dims(0)
    R = EG.dims(1)
    shape_tR = (time.size, R.size)

    ary = EG.trace_of('ary', 1, [0])
    idxs_pol = np.where((ary == 1.) | (ary == 3.))[0]
    idxs_tor = np.where((ary == 5.) | (ary == 7.))[0]

    a99 = EG.trace_of_2d('a99', [0, 1])
    reff = EG.trace_of_2d('reff', [0, 1])
    a99[a99 == 0.] = np.nan
    reff[np.abs(reff) > 1.5] = np.nan
    a99 = a99.reshape(shape_tR)
    reff = reff.reshape(shape_tR)

    a99b = ~np.isnan(a99).all(axis=1)
    time = time[a99b]
    a99 = a99[a99b]
    reff = reff[a99b]

    dat_Ti = EG.trace_of_2d('Ti', [0, 1])
    err_Ti = EG.trace_of_2d('Tier', [0, 1])
    dat_Vc = EG.trace_of_2d('Vc', [0, 1])
    err_Vc = EG.trace_of_2d('Ver', [0, 1])
    dat_Ti[dat_Ti == 0.] = np.nan
    err_Ti[err_Ti == 0.] = np.nan
    dat_Ti = dat_Ti.reshape(shape_tR)
    err_Ti = err_Ti.reshape(shape_tR)
    dat_Vc = dat_Vc.reshape(shape_tR)
    err_Vc = err_Vc.reshape(shape_tR)

    dat_Ti = dat_Ti[a99b]
    err_Ti = err_Ti[a99b]
    dat_Vc = dat_Vc[a99b]
    err_Vc = err_Vc[a99b]

    rho = reff / a99
    R_pol = R[idxs_pol]
    R_tor = R[idxs_tor]
    reff_pol = reff[:, idxs_pol]
    reff_tor = reff[:, idxs_tor]
    rho_pol = rho[:, idxs_pol]
    rho_tor = rho[:, idxs_tor]
    a99 = a99[:, 0]
    dat_Tipol = dat_Ti[:, idxs_pol]
    dat_Titor = dat_Ti[:, idxs_tor]
    err_Tipol = err_Ti[:, idxs_pol]
    err_Titor = err_Ti[:, idxs_tor]
    dat_Vcpol = dat_Vc[:, idxs_pol]
    dat_Vctor = dat_Vc[:, idxs_tor]
    err_Vcpol = err_Vc[:, idxs_pol]
    err_Vctor = err_Vc[:, idxs_tor]

    idxs_sort_pol = np.argsort(R_pol)
    idxs_sort_tor = np.argsort(R_tor)
    R_pol = R_pol[idxs_sort_pol]
    reff_pol = reff_pol[:, idxs_sort_pol]
    rho_pol = rho_pol[:, idxs_sort_pol]
    dat_Tipol = dat_Tipol[:, idxs_sort_pol]
    err_Tipol = err_Tipol[:, idxs_sort_pol]
    dat_Vcpol = dat_Vcpol[:, idxs_sort_pol]
    err_Vcpol = err_Vcpol[:, idxs_sort_pol]
    R_tor = R_tor[idxs_sort_tor]
    reff_tor = reff_tor[:, idxs_sort_tor]
    rho_tor = rho_tor[:, idxs_sort_tor]
    dat_Titor = dat_Titor[:, idxs_sort_tor]
    err_Titor = err_Titor[:, idxs_sort_tor]
    dat_Vctor = dat_Vctor[:, idxs_sort_tor]
    err_Vctor = err_Vctor[:, idxs_sort_tor]

    return time, R_pol, R_tor, \
           reff_pol, reff_tor, rho_pol, rho_tor, a99, \
           dat_Tipol, err_Tipol, dat_Titor, err_Titor, \
           dat_Vcpol, err_Vcpol, dat_Vctor, err_Vctor


def cxsmap9(EG):

    time = EG.dims(0)
    R = EG.dims(1)
    shape_tR = (time.size, R.size)

    a99 = EG.trace_of_2d('a99', [0, 1])
    reff = EG.trace_of_2d('reff', [0, 1])
    a99[a99 == 0.] = np.nan
    reff[np.abs(reff) > 1.5] = np.nan
    a99 = a99.reshape(shape_tR)
    reff = reff.reshape(shape_tR)

    a99b = ~np.isnan(a99).all(axis=1)
    time = time[a99b]
    a99 = a99[a99b]
    reff = reff[a99b]

    dat_Ti = EG.trace_of_2d('Ti', [0, 1])
    err_Ti = EG.trace_of_2d('Tier', [0, 1])
    dat_Ti[dat_Ti == 0.] = np.nan
    err_Ti[err_Ti == 0.] = np.nan
    dat_Ti = dat_Ti.reshape(shape_tR)
    err_Ti = err_Ti.reshape(shape_tR)

    dat_Ti = dat_Ti[a99b]
    err_Ti = err_Ti[a99b]

    rho = reff / a99

    idxs_sort = np.argsort(R)
    R = R[idxs_sort]
    reff = reff[:, idxs_sort]
    rho = rho[:, idxs_sort]
    a99 = a99[:, idxs_sort]
    dat_Ti = dat_Ti[:, idxs_sort]
    err_Ti = err_Ti[:, idxs_sort]

    return time, R, \
           reff, rho, a99, \
           dat_Ti, err_Ti


def cxs7_Er(egEr):
    proc.suggestNewVer(2, 'cxs7_Er')

    time = egEr.dims(0)
    RR = egEr.dims(1)
    cols = ['reff', 'reff/a99', 'Erdia', 'Erdiaerr', 'Ervt', 'Ervterr', 'Ervp', 'Ervperr', 'Ervperp', 'Ervperperr',
            'Er', 'Ererr',
            '<Er>', '<Er>err', 'Inc', 'Incerr', 'Tip', 'Tiperr', 'Vt', 'Vterr', 'Vp', 'Vperr', 'Br', 'Bz', 'Bphi']
    list_dat = [0] * len(cols)
    for ii in range(len(list_dat)):
        dat = egEr.trace_of_2d(cols[ii], [0, 1])
        dat = np.reshape(dat, (len(time), len(RR)))
        list_dat[ii] = dat

    return time, RR, list_dat, cols


def cxs7_Er_v2(egEr):
    proc.suggestNewVer(3, 'cxs7_Er')

    time = egEr.dims(0)
    RR = egEr.dims(1)
    cols = ['reff', 'reff/a99', 'Erdia', 'Erdiaerr', 'Ervt', 'Ervterr', 'Ervp', 'Ervperr', 'Ervperp', 'Ervperperr',
            'Er', 'Ererr',
            '<Er>', '<Er>err', 'Inc', 'Incerr', 'Tip', 'Tiperr', 'Vt', 'Vterr', 'Vp', 'Vperr', 'Br', 'Bz', 'Bphi']
    list_dat = [0] * len(cols)
    for ii in range(len(list_dat)):
        dat = egEr.trace_of_2d(cols[ii], [0, 1])
        dat = np.reshape(dat, (len(time), len(RR)))
        dat[dat == 0.] = np.nan
        list_dat[ii] = dat

    return time, RR, list_dat, cols


def cxs7_Er_v3(diagnm, sn):

    time, RR, list_dat, list_dimnms, list_valnms, list_dimunits, list_valunits = eg2d(diagnm, sn)
    for ii, dat in enumerate(list_dat):
        dat[dat == 0.] = np.nan
        list_dat[ii] = dat

    return time, RR, list_dat, list_dimnms, list_valnms, list_dimunits, list_valunits


def eg1d(diagnm, sn):

    eg = myEgdb.LoadEG(diagnm, sn)
    dim0 = eg.dims(0)
    list_dimnms = eg.dimnames
    list_valnms = eg.valnames
    list_dimunits = eg.dimunits
    list_valunits = eg.valunits
    list_dat = [0]*len(list_valnms)
    for ii, valnm in enumerate(list_valnms):
        dat = eg.trace_of(valnm, 0, [0])
        list_dat[ii] = dat

    return dim0, list_dat, list_dimnms, list_valnms, list_dimunits, list_valunits


def eg2d(diagnm, sn, sub=1):

    eg = myEgdb.LoadEG(diagname=diagnm, sn=sn, sub=sub)
    dim0 = eg.dims(0)
    dim1 = eg.dims(1)
    list_dimnms = eg.dimnames
    list_valnms = eg.valnames
    list_dimunits = eg.dimunits
    list_valunits = eg.valunits
    list_dat = [0]*len(list_valnms)
    for ii, valnm in enumerate(list_valnms):
        dat = eg.trace_of_2d(valnm, [0, 1])
        dat = np.reshape(dat, eg.dimsize)
        list_dat[ii] = dat

    return dim0, dim1, list_dat, list_dimnms, list_valnms, list_dimunits, list_valunits


def eg3d(diagnm, sn):
    eg = myEgdb.LoadEG(diagnm, sn)
    dim0 = eg.dims(0)
    dim1 = eg.dims(1)
    dim2 = eg.dims(2)
    list_dimnms = eg.dimnames
    list_valnms = eg.valnames
    list_dimunits = eg.dimunits
    list_valunits = eg.valunits
    list_dat3d = [0] * len(list_valnms)
    for ii, valnm in enumerate(list_valnms):
        dat = eg.trace_of_3d(valnm, [0, 1, 2])
        dat = np.reshape(dat, eg.dimsize)
        list_dat3d[ii] = dat

    return dim0, dim1, dim2, list_dat3d, list_dimnms, list_valnms, list_dimunits, list_valunits


def eg4d(diagnm, sn, sub):
    eg = myEgdb.LoadEG(diagname=diagnm, sn=sn, sub=sub)
    dim0 = eg.dims(0)
    dim1 = eg.dims(1)
    dim2 = eg.dims(2)
    dim3 = eg.dims(3)
    list_dimnms = eg.dimnames
    list_valnms = eg.valnames
    list_dimunits = eg.dimunits
    list_valunits = eg.valunits
    list_dat4d = [0] * len(list_valnms)
    for ii, valnm in enumerate(list_valnms):
        dat = eg.trace_of_4d(valnm, [0, 1, 2, 3])
        dat = np.reshape(dat, eg.dimsize)
        list_dat4d[ii] = dat

    return dim0, dim1, dim2, dim3, list_dat4d, list_dimnms, list_valnms, list_dimunits, list_valunits


def choose_ch(sn):

    # information
    devices = {0: 'BS (3-O)', 1: 'DBS (3-O)', 2: 'DBS (9-O)'}

    if sn >= 179617:
        diagch_highK = {
            1: ['MWRM-COMB2', [17, 18]],
            2: ['MWRM-PXI', [11, 12]],
            3: ['MWRM-COMB2', [19, 20]]
        }
    else:
        diagch_highK = {
            1: ['MWRM-COMB2', [17, 18]],
            2: ['MWRM-PXI', [11, 12]],
            3: ['MWRM-PXI', [1, 2]]
        }

    fsig_comb = {0: '27.7G', 1: '29.1G', 2: '30.5G', 3: '32.0G',
                 4: '33.4G', 5: '34.8G', 6: '36.9G', 7: '38.3G'}
    diagch_comb = {
        '27.7G': ['MWRM-COMB', [7, 8]],
        '29.1G': ['MWRM-COMB', [5, 6]],
        '30.5G': ['MWRM-COMB', [3, 4]],
        '32.0G': ['MWRM-COMB', [1, 2]],
        '33.4G': ['MWRM-COMB', [9, 10]],
        '34.8G': ['MWRM-PXI', [7, 8]],
        '36.9G': ['MWRM-COMB', [11, 12]],
        '38.3G': ['MWRM-COMB', [13, 14]]
    }

    diagch_comb2 = {
        '27.7G': ['MWRM-COMB2', [1, 2]],
        '29.1G': ['MWRM-COMB2', [3, 4]],
        '30.5G': ['MWRM-COMB2', [5, 6]],
        '32.0G': ['MWRM-COMB2', [7, 8]],
        '33.4G': ['MWRM-COMB2', [9, 10]],
        '34.8G': ['MWRM-COMB2', [11, 12]],
        '36.9G': ['MWRM-COMB2', [13, 14]],
        '38.3G': ['MWRM-COMB2', [15, 16]]
    }

    # input
    idx_dev = int(input(f'Which device use ?\n'
                        f'{devices}\n'
                        f'>>> '))
    if idx_dev == 0:
        idx_line = int(input(f'Which line use ?\n'
                             f'{diagch_highK}\n'
                             f'>>> '))
        ch = idx_line
        diag = diagch_highK[ch][0]
        chIQ = diagch_highK[ch][1]
    elif idx_dev == 1:
        idx_line = int(input(f'Which line use ?\n'
                             f'{fsig_comb}\n'
                             f'>>> '))
        ch = fsig_comb[idx_line]
        diag = diagch_comb[ch][0]
        chIQ = diagch_comb[ch][1]
    elif idx_dev == 2:
        idx_line = int(input(f'Which line use ?\n'
                             f'{fsig_comb}\n'
                             f'>>> '))
        ch = fsig_comb[idx_line]
        diag = diagch_comb2[ch][0]
        chIQ = diagch_comb2[ch][1]
    else:
        print(f'It does not exist.\n')
        exit()

    print('\n')

    return idx_dev, ch, diag, chIQ


def choose_ch_2(sn, idx_dev, ch):

    # information
    devices = {0: 'BS (3-O)', 1: 'DBS (3-O)', 2: 'DBS (9-O)'}

    if sn >= 179617:
        diagch_highK = {
            1: ['MWRM-COMB2', [17, 18]],
            2: ['MWRM-PXI', [11, 12]],
            3: ['MWRM-COMB2', [19, 20]]
        }
    else:
        diagch_highK = {
            1: ['MWRM-COMB2', [17, 18]],
            2: ['MWRM-PXI', [11, 12]],
            3: ['MWRM-PXI', [1, 2]]
        }

    fsig_comb = {0: '27.7G', 1: '29.1G', 2: '30.5G', 3: '32.0G',
                 4: '33.4G', 5: '34.8G', 6: '36.9G', 7: '38.3G'}
    diagch_comb = {
        '27.7G': ['MWRM-COMB', [7, 8]],
        '29.1G': ['MWRM-COMB', [5, 6]],
        '30.5G': ['MWRM-COMB', [3, 4]],
        '32.0G': ['MWRM-COMB', [1, 2]],
        '33.4G': ['MWRM-COMB', [9, 10]],
        '34.8G': ['MWRM-PXI', [7, 8]],
        '36.9G': ['MWRM-COMB', [11, 12]],
        '38.3G': ['MWRM-COMB', [13, 14]]
    }

    diagch_comb2 = {
        '27.7G': ['MWRM-COMB2', [1, 2]],
        '29.1G': ['MWRM-COMB2', [3, 4]],
        '30.5G': ['MWRM-COMB2', [5, 6]],
        '32.0G': ['MWRM-COMB2', [7, 8]],
        '33.4G': ['MWRM-COMB2', [9, 10]],
        '34.8G': ['MWRM-COMB2', [11, 12]],
        '36.9G': ['MWRM-COMB2', [13, 14]],
        '38.3G': ['MWRM-COMB2', [15, 16]]
    }

    # input
    if idx_dev == 0:
        diag = diagch_highK[ch][0]
        chIQ = diagch_highK[ch][1]
    elif idx_dev == 1:
        diag = diagch_comb[ch][0]
        chIQ = diagch_comb[ch][1]
    elif idx_dev == 2:
        diag = diagch_comb2[ch][0]
        chIQ = diagch_comb2[ch][1]
    else:
        print(f'It does not exist.\n')
        exit()

    print('\n')

    return diag, chIQ


def choose_sn(inputfile):

    critsn = int(input(f'Choose shot# and sub-shot# by ...\n'
                       f'0: Manual, 1: Input file\n'
                       f'>>> '))
    if critsn == 0:
        sn = int(input('Shot # >>> '))
        subsn = int(input('Sub Shot # >>> '))
    elif critsn == 1:
        input_df = pd.read_csv(inputfile, header=None, index_col=0)
        sn = int(input_df.at['sn', 1])
        subsn = int(input_df.at['subsn', 1])
    else:
        print('Oups!')
        exit()

    return sn, subsn


def input_FFT(inputFFTfile):

    inputFFT_df = pd.read_csv(inputFFTfile, header=None, index_col=0)
    Nfft_pw = int(inputFFT_df.at['Nfftpwr', 1])
    window = str(inputFFT_df.at['window', 1])
    Nens = int(inputFFT_df.at['Nens', 1])
    Nfft = 2 ** Nfft_pw
    OVR = float(inputFFT_df.at['OVR', 1])
    NOV = int(Nfft * OVR + 0.5)

    return Nfft_pw, Nfft, window, Nens, OVR, NOV


def input_fDSk(inputFFTfile):

    proc.suggestNewVer(2, 'input_fDSk')

    inputFFT_df = pd.read_csv(inputFFTfile, header=None, index_col=0)
    fdelDopp_k_L = int(inputFFT_df.at['fdelk_L', 1])
    fdelDopp_k_H = int(inputFFT_df.at['fdelk_H', 1])
    fForSk_k_L = int(inputFFT_df.at['fForSk_k_L', 1])
    fForSk_k_H = int(inputFFT_df.at['fForSk_k_H', 1])
    frangefD_k = (fdelDopp_k_L, fdelDopp_k_H)
    frangeSk_k = (fForSk_k_L, fForSk_k_H)

    return frangefD_k, frangeSk_k


def input_fDSk_v2(inputFFTfile):

    inputFFT_df = pd.read_csv(inputFFTfile, header=None, index_col=0)
    fdelDopp_k_L = int(inputFFT_df.at['fdelk_L', 1])
    fdelDopp_k_H = int(inputFFT_df.at['fdelk_H', 1])
    fForSk_k_L = int(inputFFT_df.at['fForSk_k_L', 1])
    fForSk_k_H = int(inputFFT_df.at['fForSk_k_H', 1])
    SkCoefPow = int(inputFFT_df.at['SkCoefPow', 1])
    SkCoef = 10**SkCoefPow
    frangefD_k = (fdelDopp_k_L, fdelDopp_k_H)
    frangeSk_k = (fForSk_k_L, fForSk_k_H)

    return frangefD_k, frangeSk_k, SkCoef


def input_specrange(inputfile):

    input_df = pd.read_csv(inputfile, header=None, index_col=0)

    flim_k = int(input_df.at['flimk', 1])
    bottom = int(input_df.at['bottomdB', 1])
    top = int(input_df.at['topdB', 1])

    return flim_k, bottom, top


def LHD_IQ_et(sn, subsn, diagname, chs, et):

    Idat, Iprms = LHDR.RetrieveData_et(diagname, sn, subsn, chs[0], et)
    Qdat, Qprms = LHDR.RetrieveData_et(diagname, sn, subsn, chs[1], et)
    tdat, tprms = LHDR.RetrieveTime(diagname, sn, subsn, chs[0])
    tdat = tdat[(tdat >= et[0]) & (tdat <= et[1])]

    dT = parse('{:f}{:S}', tprms['ClockCycle'][0])[0]
    tsize = len(tdat)
    Fs = int(Iprms['SamplingClock'][0])

    print('\n')

    return tdat, Idat, Qdat, dT, Fs, tsize


def LHD_IQ_et_v2(sn, subsn, diagname, chs, et):

    Idat, Iprms = LHDR.RetrieveData_et(diagname, sn, subsn, chs[0], et)
    Qdat, Qprms = LHDR.RetrieveData_et(diagname, sn, subsn, chs[1], et)
    tdat, tprms = LHDR.RetrieveTime(diagname, sn, subsn, chs[0])
    tdat = tdat[(tdat >= et[0]) & (tdat <= et[1])]

    dT = parse('{:f}{:S}', tprms['ClockCycle'][0])[0]
    tsize = len(tdat)
    Fs = int(Iprms['SamplingClock'][0])

    print('\n')

    return tdat, Idat, Qdat, dT, Fs, tsize, tprms, Iprms, Qprms


def LHD_IQ_et_v3(sn, subsn, diagname, chs, et):

    Idat, Iprms = LHDR.RetrieveData_et(diagname, sn, subsn, chs[0], et)
    Qdat, Qprms = LHDR.RetrieveData_et(diagname, sn, subsn, chs[1], et)
    tdat, tprms = LHDR.RetrieveTime(diagname, sn, subsn, chs[0])
    tdat = tdat[(tdat >= et[0]) & (tdat <= et[1])]

    dT = parse('{:f}{:S}', tprms['ClockCycle'][0])[0]
    tsize = len(tdat)

    print('\n')

    return tdat, Idat, Qdat, dT, tsize, tprms, Iprms, Qprms


def LHD_IQ_ss(sn, subsn, diagname, chs, ss):

    Idat, Iprms = LHDR.RetrieveData_ss(diagname, sn, subsn, chs[0], ss)
    Qdat, Qprms = LHDR.RetrieveData_ss(diagname, sn, subsn, chs[1], ss)

    print('\n')

    return Idat, Qdat


def LHD_time(sn, subsn, diagname, ch):

    print(type(ch))
    tdat, tprms = LHDR.RetrieveTime(diagname, sn, subsn, ch)
    tsize = len(tdat)
    dT = parse('{:f}{:S}', tprms['ClockCycle'][0])[0]

    return tdat, dT, tsize


def LHD_et(sn, subsn, diagname, ch, et):
    dat, prms = LHDR.RetrieveData_et(diagname, sn, subsn, ch, et)
    tdat, tprms = LHDR.RetrieveTime(diagname, sn, subsn, ch)
    tdat = tdat[(tdat >= et[0]) & (tdat <= et[1])]

    tsize = len(tdat)
    dT = parse('{:f}{:S}', tprms['ClockCycle'][0])[0]
    Fs = int(prms['SamplingClock'][0])

    print('\n')

    return tdat, dat, dT, Fs, tsize


def LHD_et_v2(sn, subsn, diagname, ch, et):
    dat, prms = LHDR.RetrieveData_et(diagname, sn, subsn, ch, et)
    tdat, tprms = LHDR.RetrieveTime(diagname, sn, subsn, ch)
    tdat = tdat[(tdat >= et[0]) & (tdat <= et[1])]

    tsize = len(tdat)
    dT = parse('{:f}{:S}', tprms['ClockCycle'][0])[0]
    Fs = int(prms['SamplingClock'][0])

    print('\n')

    return tdat, dat, dT, Fs, tsize, tprms, prms


def LHD_ss(sn, subsn, diagname, ch, ss):

    dat, prms = LHDR.RetrieveData_ss(diagname, sn, subsn, ch, ss)
    tdat, tprms = LHDR.RetrieveTime(diagname, sn, subsn, ch)
    tdat = tdat[np.arange(ss[0], ss[1] + 1)]

    tsize = len(tdat)
    dT = parse('{:f}{:S}', tprms['ClockCycle'][0])[0]
    Fs = int(prms['SamplingClock'][0])

    print('\n')

    return tdat, dat, dT, Fs, tsize


def LHD_tat(sn, subsn, diag, chIQ, tat, NSamp):

    tiI, tprmsI = LHDR.RetrieveTime(diag, sn, subsn, chIQ[0])
    print('\n')
    tiQ, tprmsQ = LHDR.RetrieveTime(diag, sn, subsn, chIQ[1])
    print('\n')

    dTI = parse('{:f}{:S}', tprmsI['ClockCycle'][0])[0]
    dTQ = parse('{:f}{:S}', tprmsQ['ClockCycle'][0])[0]
    print(f'dT= I:{dTI:g}s, Q:{dTQ:g}s\n')
    if dTI != dTQ:
        print('dT are different between I and Q.')
        exit()
    else:
        dT = dTI

    if False in tiI == tiQ:
        print('ti are different between I and Q.')
        exit()
    else:
        ti = tiI
    idx_at = int(tat / dT + 0.5)
    N_half = int(NSamp / 2 + 0.5)
    ss = (idx_at - N_half, idx_at + N_half - 1)
    tmp = np.arange(ss[0], ss[1] + 1)
    ti = ti[tmp]

    Idat, prmsI = LHDR.RetrieveData_ss(diag, sn, subsn, chIQ[0], ss)
    print('\n')
    Qdat, prmsQ = LHDR.RetrieveData_ss(diag, sn, subsn, chIQ[1], ss)
    print('\n')
    if ti.size != Idat.size or Idat.size != Qdat.size:
        print('size are different among ti, I and Q.')
        exit()

    print(f'FFT time: {dT * NSamp * 1e6:.1f} us')
    print('\n')

    return ti, Idat, Qdat, dT


def LHD_IQsignal_atTime(sn, subsn, diag, chIQ, tat, NFFT, NEns, NOV):
    tiI0, dTI0, sizeI0 = LHD_time(sn, subsn, diag, chIQ[0])

    NSamp = NEns * NFFT - (NEns - 1) * NOV
    idx_tat = int((tat - 10 * (subsn - 1)) / dTI0 + 0.5)
    ss = (idx_tat - int(0.5 * NSamp + 0.5), idx_tat + int(0.5 * NSamp + 0.5) - 1)

    tiI, vI, dtI, FsI, sizeI = LHD_ss(sn, subsn, diag, chIQ[0], ss)
    tiQ, vQ, dtQ, FsQ, sizeQ = LHD_ss(sn, subsn, diag, chIQ[1], ss)

    if dtI != dtQ:
        print('Time data set is different between I and Q.')
        exit()

    dt = dtI

    return vI, vQ, dt


def getIdxsFromLongTime(tdat, tat, NSamp):
    idx_tat = np.argmin(np.abs(tdat - tat))
    ss = (idx_tat - NSamp // 2, idx_tat + NSamp // 2 - 1)
    return ss


def call_crossspec(dirin, sn, tstart, tend, diag1, chIQ1, diag2, chIQ2,
                   dT, Fsp, Nfft_pw, window, Nens):

    inname_base = f'#{sn:d}_{tstart:g}-{tend:g}s_' \
                  f'{diag1:s}_{chIQ1[0]:d}_{chIQ1[1]:d}_vs_' \
                  f'{diag2:s}_{chIQ2[0]:d}_{chIQ2[1]:d}_' \
                  f'{dT:s}_{Fsp:g}Hz_2e{Nfft_pw:d}_{window:s}_{Nens:g}_'

    inname_cs = inname_base + 'cs.csv'
    inname_cserr = inname_base + 'cserr.csv'
    inname_coh = inname_base + 'coh.csv'
    inname_coherr = inname_base + 'coherr.csv'
    inname_phs = inname_base + 'phs.csv'
    inname_phserr = inname_base + 'phserr.csv'

    csd = np.loadtxt(os.path.join(dirin, 'csvfile', inname_cs), delimiter=',')
    csd_err = np.loadtxt(os.path.join(dirin, 'csvfile', inname_cserr), delimiter=',')
    coh = np.loadtxt(os.path.join(dirin, 'csvfile', inname_coh), delimiter=',')
    coh_err = np.loadtxt(os.path.join(dirin, 'csvfile', inname_coherr), delimiter=',')
    phs = np.loadtxt(os.path.join(dirin, 'csvfile', inname_phs), delimiter=',')
    phs_err = np.loadtxt(os.path.join(dirin, 'csvfile', inname_phserr), delimiter=',')

    return csd, csd_err, coh, coh_err, phs, phs_err


def ech_local(sn, tstart_out, tend_out): ## dat_ech == Total ech power
    dir_heat = os.path.join(dirs()[1], '04-heat')
    fnm_ech = f'#{sn:d}.csv'
    path_ech = os.path.join(dir_heat, 'ech', fnm_ech)
    if os.path.isfile(path_ech):
        ech = np.loadtxt(path_ech, delimiter=',').T
        time_ech = ech[0]
        idxs_use_ech = np.where((time_ech >= tstart_out) & (time_ech <= tend_out))
        dat_ech = ech[6]
        time_ech = time_ech[idxs_use_ech]
        dat_ech = dat_ech[idxs_use_ech]
    else:
        print(f'{path_ech} is not exist. \n')
        time_ech = np.nan
        dat_ech = np.nan
    return time_ech, dat_ech


def nb_local(sn, tstart_out, tend_out):
    dir_heat = os.path.join(dirs()[1], '04-heat')
    fnm_nb = f'#{sn:d}_nb.csv'
    path_nb = os.path.join(dir_heat, 'nb', fnm_nb)
    if os.path.isfile(path_nb):
        nb = np.loadtxt(path_nb, delimiter=',').T
        time_nb = nb[0]
        idxs_use_nb = np.where((time_nb >= tstart_out) & (time_nb <= tend_out))
        dat_nb4a = nb[7]
        dat_nb5a = nb[11]
        dat_nb1 = nb[1]
        dat_nb2 = nb[3]
        dat_nb3 = nb[5]
        time_nb = time_nb[idxs_use_nb]
        dat_nb4a = dat_nb4a[idxs_use_nb]
        dat_nb5a = dat_nb5a[idxs_use_nb]
        dat_nb1 = dat_nb1[idxs_use_nb]
        dat_nb2 = dat_nb2[idxs_use_nb]
        dat_nb3 = dat_nb3[idxs_use_nb]
    else:
        print(f'{path_nb} is not exist. \n')
        time_nb = np.nan
        dat_nb1 = np.nan
        dat_nb2 = np.nan
        dat_nb3 = np.nan
        dat_nb4a = np.nan
        dat_nb5a = np.nan
    return time_nb, dat_nb1, dat_nb2, dat_nb3, dat_nb4a, dat_nb5a


def fDSk_local(sn, tstart, tend, diag, chIQ, dT, Nfft_pw, window, Nens, OVR, frangefD_k, frangeSk_k,
               sw_BSmod, sw_nb5mod, tstart_out=False, tend_out=False):

    proc.suggestNewVer(2, 'fDSk_local')

    if tstart_out == False:
        tstart_out = tstart
    if tend_out == False:
        tend_out = tend

    dir_fDSk = os.path.join(dirs()[1], '02-PowerSpecfDSk')
    fnm_fDSk = f'#{sn:d}_{tstart:g}-{tend:g}s_' \
               f'{diag:s}_{chIQ[0]:d}_{chIQ[1]:d}_' \
               f'{dT:s}_2^{Nfft_pw:d}_{window:s}_{Nens:g}_{OVR:g}_' \
               f'fD={frangefD_k[0]:d}-{frangefD_k[1]:d}kHz_Sk={frangeSk_k[0]:d}-{frangeSk_k[1]:d}kHz.csv'
    path_fDSk = os.path.join(dir_fDSk, 'csvfDSk', fnm_fDSk)

    if os.path.isfile(path_fDSk):
        fDSk = np.loadtxt(path_fDSk, delimiter=',').T
        time_fDSk = fDSk[0]
        idxs_use_fDSk = np.where((time_fDSk >= tstart_out) & (time_fDSk <= tend_out))
        dat_Sk, err_Sk = fDSk[5:7]
        dat_Ia = np.sqrt(dat_Sk)
        err_Ia = 0.5 / dat_Ia * err_Sk
        dat_Ia *= 1e4
        err_Ia *= 1e4
        dat_fD, err_fD = fDSk[3:5]

        if sw_BSmod == 1:
            dir_mod = os.path.join(dirs()[1], '03-BSmod')
            fnm_mod = f'#{sn:d}_{tstart:g}-{tend:g}s_2^{Nfft_pw:d}_{Nens:d}_{dT:s}.csv'
            path_mod = os.path.join(dir_mod, 'csv', fnm_mod)
            BSmod = np.loadtxt(path_mod, delimiter=',').T
            time_mod, dat_mod = BSmod
            idx_del = np.where(dat_mod < 0.75)[0]
            dat_Ia[idx_del] = np.nan
            err_Ia[idx_del] = np.nan
        elif sw_nb5mod == 1:
            time_nb, dat_nb1, dat_nb2, dat_nb3, dat_nb4a, dat_nb5a = nb_local(sn, tstart, tend)
            dat_nb5a_fDSk = interp1d(time_nb, dat_nb5a)(time_fDSk)
            dat_diff_nb5a = np.diff(dat_nb5a_fDSk)
            idx_del = np.where(dat_diff_nb5a < -0.5)[0]
            dat_Ia[idx_del] = np.nan
            err_Ia[idx_del] = np.nan

        time_fDSk = time_fDSk[idxs_use_fDSk]
        dat_Ia = dat_Ia[idxs_use_fDSk]
        err_Ia = err_Ia[idxs_use_fDSk]
        dat_fD = dat_fD[idxs_use_fDSk]
        err_fD = err_fD[idxs_use_fDSk]
    else:
        print(f'{path_fDSk} is not exist. \n')
        time_fDSk = np.nan
        dat_Ia = np.nan
        err_Ia = np.nan
        dat_fD = np.nan
        err_fD = np.nan

    return time_fDSk, dat_Ia, err_Ia, dat_fD, err_fD

def fDSk_local_v2(sn, tstart, tend, diag, chIQ, dT, Nfft_pw, window, Nens, OVR, frangefD_k, frangeSk_k,
                  sw_BSmod, sw_nb5mod, tstart_out=False, tend_out=False):
    if tstart_out == False:
        tstart_out = tstart
    if tend_out == False:
        tend_out = tend

    dir_fDSk = os.path.join(dirs()[1], '02-PowerSpecfDSk')
    fnm_fDSk = f'#{sn:d}_{tstart:g}-{tend:g}s_' \
               f'{diag:s}_{chIQ[0]:d}_{chIQ[1]:d}_' \
               f'{dT:s}_2^{Nfft_pw:d}_{window:s}_{Nens:g}_{OVR:g}_' \
               f'fD={frangefD_k[0]:d}-{frangefD_k[1]:d}kHz_Sk={frangeSk_k[0]:d}-{frangeSk_k[1]:d}kHz.csv'
    path_fDSk = os.path.join(dir_fDSk, 'csvfDSk', fnm_fDSk)

    if os.path.isfile(path_fDSk):
        fDSk = np.loadtxt(path_fDSk, delimiter=',').T
        time_fDSk = fDSk[0]
        idxs_use_fDSk = np.where((time_fDSk >= tstart_out) & (time_fDSk <= tend_out))
        dat_Sk, err_Sk = fDSk[5:7]
        dat_Ia, err_Ia = fDSk[11:13]
        dat_fD, err_fD = fDSk[3:5]

        if sw_BSmod == 1:
            dir_mod = os.path.join(dirs()[1], '03-BSmod')
            fnm_mod = f'#{sn:d}_{tstart:g}-{tend:g}s_2^{Nfft_pw:d}_{Nens:d}_{dT:s}.csv'
            path_mod = os.path.join(dir_mod, 'csv', fnm_mod)
            BSmod = np.loadtxt(path_mod, delimiter=',').T
            time_mod, dat_mod = BSmod
            idx_del = np.where(dat_mod < 0.75)[0]
            dat_Sk[idx_del] = np.nan
            err_Sk[idx_del] = np.nan
            dat_Ia[idx_del] = np.nan
            err_Ia[idx_del] = np.nan
        elif sw_nb5mod == 1:
            time_nb, dat_nb1, dat_nb2, dat_nb3, dat_nb4a, dat_nb5a = nb_local(sn, tstart, tend)
            dat_nb5a_fDSk = interp1d(time_nb, dat_nb5a)(time_fDSk)
            dat_diff_nb5a = np.diff(dat_nb5a_fDSk)
            idx_del = np.where(dat_diff_nb5a < -0.5)[0]
            dat_Sk[idx_del] = np.nan
            err_Sk[idx_del] = np.nan
            dat_Ia[idx_del] = np.nan
            err_Ia[idx_del] = np.nan

        time_fDSk = time_fDSk[idxs_use_fDSk]
        dat_Sk = dat_Sk[idxs_use_fDSk]
        err_Sk = err_Sk[idxs_use_fDSk]
        dat_Ia = dat_Ia[idxs_use_fDSk]
        err_Ia = err_Ia[idxs_use_fDSk]
        dat_fD = dat_fD[idxs_use_fDSk]
        err_fD = err_fD[idxs_use_fDSk]
    else:
        print(f'{path_fDSk} is not exist. \n')
        exit()

    return time_fDSk, dat_Sk, err_Sk, dat_Ia, err_Ia, dat_fD, err_fD


def dat1_woerr_local(path, col, tstart_out, tend_out):
    if os.path.isfile(path):
        array = np.loadtxt(path, delimiter=',').T
        time = array[0]
        idxs_use = np.where((time >= tstart_out) & (time <= tend_out))
        dat1 = array[col]
        time = time[idxs_use]
        dat1 = dat1[idxs_use]
    else:
        print(f'{path} is not exist. \n')
        time = np.nan
        dat1 = np.nan
    return time, dat1


def dat1_local(path, cols, tstart_out, tend_out):
    if os.path.isfile(path):
        array = np.loadtxt(path, delimiter=',').T
        time = array[0]
        idxs_use = np.where((time >= tstart_out) & (time <= tend_out))
        dat1, err1 = array[cols]
        time = time[idxs_use]
        dat1 = dat1[idxs_use]
        err1 = err1[idxs_use]
    else:
        print(f'{path} is not exist. \n')
        time = np.nan
        dat1 = np.nan
        err1 = np.nan
    return time, dat1, err1


def dat2_local(path, cols, tstart_out, tend_out):
    if os.path.isfile(path):
        array = np.loadtxt(path, delimiter=',').T
        time = array[0]
        idxs_use = np.where((time >= tstart_out) & (time <= tend_out))
        dat1, err1, dat2, err2 = array[cols]
        time = time[idxs_use]
        dat1 = dat1[idxs_use]
        err1 = err1[idxs_use]
        dat2 = dat2[idxs_use]
        err2 = err2[idxs_use]
    else:
        print(f'{path} is not exist. \n')
        time = np.nan
        dat1 = np.nan
        err1 = np.nan
        dat2 = np.nan
        err2 = np.nan
    return time, dat1, err1, dat2, err2


def fDSk_color_label(idx_dev, ch):

    devices = {0: 'BS (3-O)', 1: 'DBS (3-O)', 2: 'DBS (9-O)'}

    if idx_dev == 0:
        color_fDSk = 'red'
    elif idx_dev == 1:
        color_fDSk = 'blue'
    elif idx_dev == 2:
        color_fDSk = 'green'
    else:
        print('Device does not exist. ')
        exit()
    label_Sk = f'$I_{{pow,{devices[idx_dev]},ch{ch:d}}}$\n[a.u.]'
    label_fD_k = f'$f_{{D,{devices[idx_dev]},ch{ch:d}}}$\n[kHz]'
    label_Ia = f'$I_{{amp,{devices[idx_dev]},ch{ch:d}}}$\n[a.u.]'

    return color_fDSk, label_Sk, label_fD_k, label_Ia
