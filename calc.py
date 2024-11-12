import numpy as np  # type: ignore
from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
from scipy import signal, fft, interpolate, optimize  # type: ignore
from scipy.signal import welch  # type: ignore
import gc
import matplotlib.pyplot as plt  # type: ignore
from nasu import proc, plot, getShotInfo, myEgdb, read, const
import os
import sys
import inspect
import traceback
import time


class struct:
    pass


def samplingrate_from_timedat(times_s, atol=1e-30, rtol=2e-2):

    # calculate sampling freuency Fs
    dts = np.diff(times_s)
    if np.allclose(dts[:-1], dts[1:], atol=atol, rtol=rtol):
        Fs_Hz = 1. / (dts.mean())
    else:
        raise ValueError("Time data not equally spaced")
    
    return Fs_Hz


def expand_by1dim(array, Nexp=1, axis=-1):
    if axis < 0:
        axis = array.ndim + 1 + axis
    idx_trans = [i + 1 for i in range(array.ndim)]
    idx_trans.insert(axis, 0)

    idx_tile = [1]*array.ndim
    idx_tile.insert(0, Nexp)

    return np.transpose(np.tile(array, idx_tile), idx_trans)


def average(dat, err=None, axis=None, skipnan=False):

    if axis is None:
        datsize = dat.size
    else:
        datsize = dat.shape[axis]

    if err is None:
        if skipnan:
            avg = np.nanmean(dat, axis=axis)
            std = np.nanstd(dat, axis=axis, ddof=1)
            ste = std / np.sqrt(datsize)
        else:
            avg = np.average(dat, axis=axis)
            std = np.std(dat, axis=axis, ddof=1)
            ste = std / np.sqrt(datsize)
    else:
        if skipnan:
            err_sq = err ** 2
            w = 1. / err_sq
            avg = np.nansum(dat * w, axis=axis) / np.nansum(w, axis=axis)
            std = np.sqrt(np.nanvar(dat, axis=axis) + np.nanmean(err_sq, axis=axis))
            if axis is None:
                ste = np.sqrt(np.nansum((dat - avg) ** 2 * w, axis=axis) \
                              / ((datsize - 1) * np.nansum(w, axis=axis)) + 1. / np.nansum(w, axis=axis))
            else:
                ste = np.sqrt(np.nansum((dat - expand_by1dim(avg, datsize, axis=axis)) ** 2 * w, axis=axis) \
                              / ((datsize - 1) * np.nansum(w, axis=axis)) + 1. / np.nansum(w, axis=axis))
        else:
            err_sq = err ** 2
            w = 1. / err_sq
            avg = np.average(dat, axis=axis, weights=1./err_sq)
            std = np.sqrt(np.var(dat, axis=axis) + np.average(err_sq, axis=axis))
            if axis is None:
                ste = np.sqrt(np.sum((dat - avg) ** 2 * w, axis=axis) \
                              / ((datsize - 1) * np.sum(w, axis=axis)) + 1. / np.sum(w, axis=axis))
            else:
                ste = np.sqrt(np.sum((dat - expand_by1dim(avg, datsize, axis=axis))**2 * w, axis=axis) \
                / ((datsize - 1) * np.sum(w, axis=axis)) + 1./np.sum(w, axis=axis))

    return avg, std, ste


def average_dat_withinRhoRange(rho, list_dat, rho_in, rho_out, list_err=None, include_outerside=False):
    idxs, list_dat = proc.getXIdxsAndYs(rho, rho_in, rho_out, list_dat, include_outerside=include_outerside)
    array_dat = np.array(list_dat)

    if list_err is None:
        array_avg, array_std, array_ste = average(array_dat, err=None, axis=-1)
    else:
        idxs, list_err = proc.getXIdxsAndYs(rho, rho_in, rho_out, list_err)
        array_err = np.array(list_err)
        array_avg, array_std, array_ste = average(array_dat, err=array_err, axis=-1)

    return array_avg, array_std, array_ste


def avgByWeightHavingError(xxs, weights, weights_err):

    areNotNansInX = (~np.isnan(xxs)).astype(np.int8)
    areNotNansInWgt = (~np.isnan(weights)).astype(np.int8)
    xxs = np.nan_to_num(xxs) * areNotNansInWgt
    weights = np.nan_to_num(weights) * areNotNansInX
    weights_err = np.nan_to_num(weights_err) * areNotNansInX

    Sws = np.sum(weights, axis=-1)
    Swxs = np.sum(weights * xxs, axis=-1)
    xWAvgs = Swxs / Sws
    xWAvgs_ex = proc.repeat_and_add_lastdim(xWAvgs, weights.shape[-1])
    xWAvgErrs = np.sqrt(np.sum((xxs - xWAvgs_ex) ** 2 * weights_err ** 2, axis=-1)) / Sws

    return xWAvgs, xWAvgErrs


def sumErr(errList):
    err = np.sqrt(np.sum(np.array(errList)**2, axis=0))
    return err


def multiRer(datList, errList):
    rerArray = np.array(errList) / np.array(datList)
    rer = np.sqrt(np.sum(rerArray**2, axis=0))
    return rer


def inverseDat_andErr(dat, err):
    inv = 1 / dat
    rer = err / np.abs(dat)
    inv_err = inv * rer
    return inv, inv_err


def sqrt_AndErr(x, x_err):
    y = np.sqrt(x)
    y_err = 0.5 / y * x_err
    return y, y_err


"""
# def timeAverageProfiles(dat2d, err=np.array([False])):
#     if err.all():
#         idxs_isnanInDat2d = np.isnan(dat2d)
#         idxs_isnanInErr = np.isnan(err)
#         idxs_isnan = idxs_isnanInErr + idxs_isnanInDat2d
#         dat2d[idxs_isnan] = np.nan
#         err[idxs_isnan] = np.nan
#
#         avg = np.nanmean(dat2d, axis=0)
#         std = np.sqrt(np.nanvar(dat2d, axis=0) + np.nanmean(err ** 2, axis=0))
#     else:
#         avg = np.nanmean(dat2d, axis=0)
#         std = np.nanstd(dat2d, axis=0, ddof=1)
#
#     return avg, std
#
#
# def timeAverageDatByRefs(timeDat, dat, err, timeRef):
#     proc.suggestNewVer(2, 'timeAverageDatByRefs')
#
#     dtDat = timeDat[1] - timeDat[0]
#     dtRef = timeRef[1] - timeRef[0]
#     dNDatRef = int(dtRef / dtDat + 0.5)
#     timeRef_ext = repeat_and_add_lastdim(timeRef, len(timeDat))
#     idxDatAtRef = np.argsort(np.abs(timeDat - timeRef_ext))[:, :dNDatRef]
#
#     datAtRef = dat[idxDatAtRef]
#     errAtRef = err[idxDatAtRef]
#     dat_Ref = np.nanmean(datAtRef, axis=1)
#     err_Ref = np.sqrt(np.nanvar(datAtRef, axis=1) + np.nanmean(errAtRef ** 2, axis=1))
#
#     return dat_Ref, err_Ref
"""


def timeAverageDatByRefs_v2(timeDat, dat, timeRef, err=None, skipnan=False):
    dtDat = timeDat[1] - timeDat[0]
    dtRef = timeRef[1] - timeRef[0]
    dNDatRef = int(dtRef / dtDat + 0.5)
    timeRef_ext = repeat_and_add_lastdim(timeRef, len(timeDat))
    idxDatAtRef = np.argsort(np.abs(timeDat - timeRef_ext))[:, :dNDatRef]

    datAtRef = dat[idxDatAtRef]
    if err is not None:
        errAtRef = err[idxDatAtRef]
    else:
        errAtRef = None

    dat_Ref, std_Ref, err_Ref = average(datAtRef, err=errAtRef, axis=-1, skipnan=skipnan)

    return dat_Ref, std_Ref, err_Ref


def timeAverageDatListByRefs(timeDat, datList, timeRef, errList=None, skipnan=False):
    datRefList = [0] * len(datList)
    stdRefList = [0] * len(datList)
    errRefList = [0] * len(datList)

    for ii, dat in enumerate(datList):
        if errList is None:
            datRef, stdRef, errRef = timeAverageDatByRefs_v2(timeDat, dat, timeRef, skipnan=skipnan)
        else:
            err = errList[ii]
            datRef, stdRef, errRef = timeAverageDatByRefs_v2(timeDat, dat, timeRef, err, skipnan=skipnan)
        datRefList[ii] = datRef
        stdRefList[ii] = stdRef
        errRefList[ii] = errRef

    return datRefList, stdRefList, errRefList


def shifted_gauss(x, a, b, x0, C):
    return C + a * np.exp(-(x - x0) ** 2 / b ** 2)


def fit_xyshifted_gauss_1d(x, y, y_err):

    Ndat = len(y)

    window_size = 20
    y_smooth = np.convolve(y, np.ones(window_size) / window_size, mode='same')
    top_values = np.sort(y_smooth)[-Ndat // 50:]
    bottom_values = np.sort(y_smooth)[:Ndat // 10]

    y_max = np.nanmean(top_values)
    y_min = np.nanmean(bottom_values)

    # y_1_3rd = y_min + 0.33 * (y_max - y_min)
    # y_2_3rd = y_min + 0.66 * (y_max - y_min)
    # idx_at_y_max = np.nanargmin(np.abs(y_smooth - y_max))
    # idx_at_y_min = np.nanargmin(np.abs(y_smooth - y_min))
    # idx_at_y_1_3rd = np.nanargmin(np.abs(y_smooth - y_1_3rd))
    # idx_at_y_2_3rd = np.nanargmin(np.abs(y_smooth - y_2_3rd))
    # x_at_ymin = x[idx_at_y_min]
    # x_at_ymax = x[idx_at_y_max]
    # x_at_y_1_3rd = x[idx_at_y_1_3rd]
    # x_at_y_2_3rd = x[idx_at_y_2_3rd]

    iniC = y_min
    weight_power = 10
    inix0 = np.average(x, weights= np.abs(y_smooth) ** weight_power)

    inia = y_max - iniC
    n = 1
    idx = np.nanargmin(np.abs(y_smooth - (iniC + inia / (np.e ** (n ** 2)))))
    inib = np.abs(x[idx] - inix0) / n

    # ini_ys = np.array([y_max, y_1_3rd, y_2_3rd])
    # ini_xs = np.array([x_at_ymax, x_at_y_1_3rd, x_at_y_2_3rd])
    # ini_lnys = np.log(ini_ys - iniC)
    # coefs = np.polyfit(ini_xs, ini_lnys, 2)
    # alpha, beta, gamma = coefs
    # iniA = gamma - 0.25 * beta**2 / alpha
    # inia = np.exp(iniA)
    # inib = np.sqrt(-1. / alpha)
    # inix0 = - beta / (2. * alpha)

    iniP1 = [inia, inib]
    print(iniP1)
    bounds1 = ([1e-16, 1e-16], [np.inf, np.inf])
    popt1, pcov1 = optimize.curve_fit(lambda x_tmp, a, b: shifted_gauss(x_tmp, a, b, inix0, iniC),
                                      x, y, p0=iniP1, sigma=y_err, bounds=bounds1)

    iniP2 = [popt1[0], popt1[1], inix0, iniC]
    print(iniP2)
    bounds2 = ([1e-16, 1e-16, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf])
    popt2, pcov2 = optimize.curve_fit(shifted_gauss, x, y, p0=iniP2, sigma=y_err, bounds=bounds2)

    perr = np.sqrt(np.diagonal(pcov2))
    y_hut = shifted_gauss(x, popt2[0], popt2[1], popt2[2], popt2[3])
    sigma_y = np.sqrt(np.average((y - y_hut) ** 2) + np.average(y_err ** 2))
    print(popt2)

    print('\n')

    return popt2, perr, sigma_y, y_hut


def IQsignal(Idat, Qdat, idx_dev, chDiag):
    if idx_dev == 1 and chDiag in ['33.4G', '34.8G', '36.9G', '38.3G']:
        signal = Idat + 1.j * Qdat
    elif idx_dev == 0:
        signal = Idat + 1.j * Qdat
    else:
        signal = Idat - 1.j * Qdat

    return signal


def MakeLLSMFitProfilesFromTS(sn, startTime, endTime, Nfit, poly):

    info = getShotInfo.info(sn)
    Bt, Rax = info[0:2]

    egts = myEgdb.LoadEG('tsmap_calib', sn)
    egnel = myEgdb.LoadEG('tsmap_nel', sn)
    egtsreff = myEgdb.LoadEG('tsmap_reff', sn)
    print('\n')

    if egts == None:
        print('egts == None')
        sys.exit()

    timeref, Rref, reffref, \
    B, Br, Bz, Bphi = read.field_from_tsmap_reff(egtsreff, sn)
    time, R, reff, rho, \
    dat_Te, err_Te, dat_ne, err_ne, \
    dat_Te_fit, err_Te_fit, dat_ne_fit, err_ne_fit = read.tsmap_calib(egts)
    a99 = egnel.trace_of('a99', 0, [0])

    datList = [time, reff, rho, a99, dat_Te, err_Te, dat_ne, err_ne, B, Br, Bz, Bphi]
    idxs_tRange, datList = proc.getTimeIdxsAndDats(time, startTime, endTime, datList)
    time, reff, rho, a99, dat_Te, err_Te, dat_ne, err_ne, B, Br, Bz, Bphi = datList

    dat2dList = [reff, rho, dat_Te, err_Te, dat_ne, err_ne, Bz, Bphi, B]
    isNotNan, dat2dList = proc.notNanInDat2d(dat2dList, 0)
    reff, rho, dat_Te, err_Te, dat_ne, err_ne, Bz, Bphi, B = dat2dList
    R = R[isNotNan]

    R_f, reff_f, rho_f, dat_Te_grad, err_Te_grad, \
    dat_Te_reg, err_Te_reg = gradient_reg_v2(R, reff, a99, dat_Te, err_Te, Nfit, poly)
    R_f, reff_f, rho_f, dat_ne_grad, err_ne_grad, \
    dat_ne_reg, err_ne_reg = gradient_reg_v2(R, reff, a99, dat_ne, err_ne, Nfit, poly)

    dat_Lne, err_Lne, dat_RpLne, err_RpLne = Lscale(dat_ne_reg, err_ne_reg,
                                                         dat_ne_grad, err_ne_grad, Rax)
    dat_LTe, err_LTe, dat_RpLTe, err_RpLTe = Lscale(dat_Te_reg, err_Te_reg,
                                                         dat_Te_grad, err_Te_grad, Rax)
    dat_etae, err_etae = eta(dat_LTe, err_LTe, dat_Lne, err_Lne)

    omega_ce = const.ee * B / const.me
    tmp = 1e19 * const.ee ** 2 / (const.eps0 * const.me)
    omega_pe, omega_pe_err = sqrt_AndErr(dat_ne * tmp, err_ne * tmp)
    omega_L = 0.5 * (-omega_ce + np.sqrt(omega_ce ** 2 + 4 * omega_pe ** 2))
    omega_L_err = np.abs(0.5 * 0.5 * 1 / np.sqrt(omega_ce ** 2 + 4 * omega_pe ** 2) * 4 * 2 * omega_pe * omega_pe_err)
    omega_R = omega_L + omega_ce
    omega_R_err = omega_L_err

    datList = [omega_ce, omega_pe, omega_pe_err, omega_L, omega_L_err, omega_R, omega_R_err]
    dat2List = [0] * len(datList)
    for ii, dat in enumerate(datList):
        dat2List[ii] = dat / (2 * np.pi)
    fce, fpe, fpe_err, fL, fL_err, fR, fR_err = dat2List

    datList = [fce, fpe, fpe_err, fL, fL_err, fR, fR_err]
    dat2List = [0] * len(datList)
    for ii, dat in enumerate(datList):
        dat2List[ii] = dat * 1e-9
    fce_G, fpe_G, fpe_G_err, fL_G, fL_G_err, fR_G, fR_G_err = dat2List

    raw_list = [R, reff, rho, a99, dat_Te, err_Te, dat_ne, err_ne, B, Br, Bz, Bphi,
                fce_G, fpe_G, fpe_G_err, fL_G, fL_G_err, fR_G, fR_G_err]
    reg_list = [R_f, reff_f, rho_f, dat_Te_reg, err_Te_reg, dat_ne_reg, err_ne_reg, dat_Te_grad, err_Te_grad,
                dat_ne_grad, err_ne_grad, dat_LTe, err_LTe, dat_Lne, err_Lne,
                dat_RpLTe, err_RpLTe, dat_RpLne, err_RpLne, dat_etae, err_etae]

    return time, raw_list, reg_list


def MakeLLSMFitProfilesFromCXS7(sn, startTime, endTime, Nfit, poly):

    info = getShotInfo.info(sn)
    Rax = info[1]

    egcx7 = myEgdb.LoadEG('cxsmap7', sn)
    print('\n')

    if egcx7 == None:
        print('egcx7 == None')
        sys.exit()

    time, R_pol, R_tor, \
    reff_pol, reff_tor, rho_pol, rho_tor, a99, \
    dat_Tipol, err_Tipol, dat_Titor, err_Titor, \
    dat_Vcpol, err_Vcpol, dat_Vctor, err_Vctor = read.cxsmap7_v1(egcx7)

    datList = [time, reff_pol, reff_tor, rho_pol, rho_tor, a99,
               dat_Tipol, err_Tipol, dat_Titor, err_Titor, dat_Vcpol, err_Vcpol, dat_Vctor, err_Vctor]
    idxs_tRange, datList = proc.getTimeIdxsAndDats(time, startTime, endTime, datList)

    time, reff_pol, reff_tor, rho_pol, rho_tor, a99, \
    dat_Tipol, err_Tipol, dat_Titor, err_Titor, dat_Vcpol, err_Vcpol, dat_Vctor, err_Vctor = datList

    dat2dList = [reff_pol, rho_pol, dat_Tipol, err_Tipol, dat_Vcpol, err_Vcpol]
    isNotNan, dat2dList = proc.notNanInDat2d(dat2dList, 1)
    reff_pol, rho_pol, dat_Tipol, err_Tipol, dat_Vcpol, err_Vcpol = dat2dList

    time = time[isNotNan]
    a99 = a99[isNotNan]

    dat2dList = [reff_pol, rho_pol, dat_Tipol, err_Tipol, dat_Vcpol, err_Vcpol]
    isNotNan, dat2dList = proc.notNanInDat2d(dat2dList, 0)
    reff_pol, rho_pol, dat_Tipol, err_Tipol, dat_Vcpol, err_Vcpol = dat2dList

    R_pol = R_pol[isNotNan]

    dat2dList = [reff_tor, rho_tor, dat_Titor, err_Titor, dat_Vctor, err_Vctor]
    isNotNan, dat2dList = proc.notNanInDat2d(dat2dList, 1)
    reff_tor, rho_tor, dat_Titor, err_Titor, dat_Vctor, err_Vctor = dat2dList

    dat2dList = [reff_tor, rho_tor, dat_Titor, err_Titor, dat_Vctor, err_Vctor]
    isNotNan, dat2dList = proc.notNanInDat2d(dat2dList, 0)
    reff_tor, rho_tor, dat_Titor, err_Titor, dat_Vctor, err_Vctor = dat2dList

    R_tor = R_tor[isNotNan]

    R_pol_f, reff_pol_f, rho_pol_f, dat_Tipol_grad, err_Tipol_grad, \
    dat_Tipol_reg, err_Tipol_reg = gradient_reg_v2(R_pol, reff_pol, a99, dat_Tipol, err_Tipol, Nfit, poly)
    R_tor_f, reff_tor_f, rho_tor_f, dat_Titor_grad, err_Titor_grad, \
    dat_Titor_reg, err_Titor_reg = gradient_reg_v2(R_tor, reff_tor, a99, dat_Titor, err_Titor, Nfit, poly)
    R_pol_f, reff_pol_f, rho_pol_f, dat_Vcpol_grad, err_Vcpol_grad, \
    dat_Vcpol_reg, err_Vcpol_reg = gradient_reg_v2(R_pol, reff_pol, a99, dat_Vcpol, err_Vcpol, Nfit, poly)
    R_tor_f, reff_tor_f, rho_tor_f, dat_Vctor_grad, err_Vctor_grad, \
    dat_Vctor_reg, err_Vctor_reg = gradient_reg_v2(R_tor, reff_tor, a99, dat_Vctor, err_Vctor, Nfit, poly)

    dat_LTipol, err_LTipol, dat_RpLTipol, err_RpLTipol = \
        Lscale(dat_Tipol_reg, err_Tipol_reg, dat_Tipol_grad, err_Tipol_grad, Rax)
    dat_LTitor, err_LTitor, dat_RpLTitor, err_RpLTitor = \
        Lscale(dat_Titor_reg, err_Titor_reg, dat_Titor_grad, err_Titor_grad, Rax)
    dat_LVcpol, err_LVcpol, dat_RpLVcpol, err_RpLVcpol = \
        Lscale(dat_Vcpol_reg, err_Vcpol_reg, dat_Vcpol_grad, err_Vcpol_grad, Rax)
    dat_LVctor, err_LVctor, dat_RpLVctor, err_RpLVctor = \
        Lscale(dat_Vctor_reg, err_Vctor_reg, dat_Vctor_grad, err_Vctor_grad, Rax)

    raw_list = [R_pol, R_tor, reff_pol, reff_tor, rho_pol, rho_tor, a99,
                dat_Tipol, err_Tipol, dat_Titor, err_Titor, dat_Vcpol, err_Vcpol, dat_Vctor, err_Vctor]
    reg_list = [R_pol_f, reff_pol_f, rho_pol_f, R_tor_f, reff_tor_f, rho_tor_f,
                dat_Tipol_reg, err_Tipol_reg, dat_Titor_reg, err_Titor_reg,
                dat_Vcpol_reg, err_Vcpol_reg, dat_Vctor_reg, err_Vctor_reg,
                dat_Tipol_grad, err_Tipol_grad, dat_Titor_grad, err_Titor_grad,
                dat_Vcpol_grad, err_Vcpol_grad, dat_Vctor_grad, err_Vctor_grad,
                dat_LTipol, err_LTipol, dat_LTitor, err_LTitor,
                dat_LVcpol, err_LVcpol, dat_LVctor, err_LVctor,
                dat_RpLTipol, err_RpLTipol, dat_RpLTitor, err_RpLTitor,
                dat_RpLVcpol, err_RpLVcpol, dat_RpLVctor, err_RpLVctor ]

    return time, raw_list, reg_list


def Er_vExB_1ion(Ti, LTi, Lne, Vtor, Vpol, Btor, Bpol, Zi,
                 Ti_er, LTi_er, Lne_er, Vtor_er, Vpol_er, Btor_er, Bpol_er):
    # [keV, keV/m, e19m^-3, e19m^-4, km/s, km/s, T, T, -]

    gradTi, gradTi_er = inverseDat_andErr(LTi, LTi_er)
    gradne, gradne_er = inverseDat_andErr(Lne, Lne_er)
    sumgrad = gradTi + gradne
    sumgrad_er = sumErr([gradTi_er, gradne_er])
    Er_gradp = Ti * sumgrad / Zi
    Er_gradp_err = multiRer([Ti, sumgrad], [Ti_er, sumgrad_er]) * np.abs(Er_gradp)
    VtBp = Vtor * Bpol
    VtBp_err = multiRer([Vtor, Bpol], [Vtor_er, Bpol_er]) * np.abs(VtBp)
    VpBt = Vpol * Btor
    VpBt_err = multiRer([Vpol, Btor], [Vpol_er, Btor_er]) * np.abs(VpBt)
    Er_lorenz = - (VtBp - VpBt)
    Er_lorenz_err = sumErr([VtBp_err, VpBt_err])
    print(f'grad p term: {Er_gradp} pm {Er_gradp_err}')
    print(f'lorenz term: {Er_lorenz} pm {Er_lorenz_err}')

    Er = Er_gradp + Er_lorenz   # [k V/m = k N/C]
    Er_err = sumErr([Er_gradp_err, Er_lorenz_err])

    vExB = Er / Btor  # [km/s]
    vExB_err = multiRer([Er, Btor], [Er_err, Btor_er]) * np.abs(vExB)

    return Er, Er_err, vExB, vExB_err   # [kV/m, km/s]


"""
# def center_of_gravity_of_complex_spectrum(freq, psd, psd_err, power=2):

#     idx_fD = np.where(np.abs(freq) > 1000)[0]
#     freq_fD = freq[idx_fD]
#     psd_fD = psd[:, idx_fD]
#     psd_err_fD = psd_err[:, idx_fD]

#     freq_ex2d = np.full(psd_fD.shape, freq_fD)
#     weight = psd_fD**power
#     weight_err = power * (psd_fD**(power - 1)) * psd_err_fD
#     cog, cog_err = avgByWeightHavingError(freq_ex2d, weight, weight_err)

#     return cog, cog_err
"""

def center_of_gravity(t, x, NFFT=2**6, OVR=0.5, window="hann"):

    NOV = int(NFFT * OVR)
    Nsp = NEnsFromNSample(NFFT, NOV, len(t))
    Ndat = NSampleForFFT(NFFT, Nsp, NOV)

    t = t[:Ndat]
    x = x[:Ndat]

    idxs = make_idxs_for_spectrogram_wo_EnsembleAVG(NFFT, NOV, Nsp)
    tisp = time_for_spectrogram_wo_ensembleAVG(t, idxs)
    xens = x[idxs]

    dt = t[1] - t[0]
    win, enbw, CG = get_window(NFFT, window)
    freq, fft_x = fourier_components_2s(xens, dt, NFFT, win)
    fft_x /=  np.sqrt(NFFT / 2)

    ind=np.where(np.abs(freq)>1000)[0]
    S = fft_x[:, ind] * np.conj(fft_x[:, ind])
    SF = np.real(S * np.conj(S))
    cog = np.sum(SF*freq[ind], axis=-1)/np.sum(SF, axis=-1)

    return tisp, cog


def gradRegAtInterestRhoWithErr(ReffAvg, RhoAvg, Dat, DatEr, InterestingRho, NFit, polyGrad):
    print('Newest ver. -> timeSeriesRegGradAtRhoOfInterest')
    reffIdxsForLSMAtInterest, shiftedReffsForLSMAtInterest = proc.makeReffArraysForLSMAtInterestRho(ReffAvg, RhoAvg, InterestingRho)

    reffsForLSMAtInterest = np.tile(shiftedReffsForLSMAtInterest[:NFit], (Dat.shape[0], 1))
    DatAtInterest = Dat[:, reffIdxsForLSMAtInterest][:, :NFit]
    DatErAtInterest = DatEr[:, reffIdxsForLSMAtInterest][:, :NFit]

    prms, errs = polyN_LSM(reffsForLSMAtInterest, DatAtInterest, DatErAtInterest, polyGrad)
    RegDat = prms[-1]
    DatGrad = prms[-2]
    RegDatEr = errs[-1]
    DatGradEr = errs[-2]

    return RegDat, RegDatEr, DatGrad, DatGradEr


def timeSeriesRegGradAtRhoOfInterest1d(reff1d, rho1d, dat, err, rhoOfInterest, NFit, polyGrad, fname_base=None, dir_name=None):

    reffOfInterest, idxsForLSMAtInterest, shiftedReffsForLSMAtInterest = \
        proc.makeReffArraysForLSMAtInterestRho(reff1d, rho1d, rhoOfInterest)

    newShape = (len(dat), len(reff1d))
    shiftedReffsForLSMAtInterest = np.full(newShape, shiftedReffsForLSMAtInterest)

    reffsForLSMAtInterest = shiftedReffsForLSMAtInterest[:, :NFit]
    DatAtInterest = dat[:, idxsForLSMAtInterest][:, :NFit]
    DatErAtInterest = err[:, idxsForLSMAtInterest][:, :NFit]

    idxs_sort = proc.argsort(reffsForLSMAtInterest)
    reffsForLSMAtInterest = reffsForLSMAtInterest[idxs_sort]
    DatAtInterest = DatAtInterest[idxs_sort]
    DatErAtInterest = DatErAtInterest[idxs_sort]

    prms, errs, fitErr, datFit, fitCurveErrs = \
        polyN_LSM_v2(reffsForLSMAtInterest, DatAtInterest, polyGrad, DatErAtInterest)
    RegDat = prms[-1]
    RegDatEr = fitErr
    DatGrad = prms[-2]
    DatGradEr = errs[-2]

    if fname_base is not None:

        figOutDir = os.path.join(dir_name, 'fig')
        proc.ifNotMake(figOutDir)

        for i in range(len(reffsForLSMAtInterest)):
            Nrepeat = 1000
            yNoise_set = np.random.normal(datFit[i], fitErr[i], (Nrepeat, len(datFit[i])))
            prms, errs, sigma_y, y_hut, y_hut_errs = \
                polyN_LSM_v2(np.full((Nrepeat, len(datFit[i])), reffsForLSMAtInterest[i]), yNoise_set, polyGrad)
            prms_avg = np.average(prms, axis=1)
            prms_std = np.std(prms, axis=1, ddof=1)
            errs_avg = np.average(errs, axis=1)

            print(f'dat = {prms[-1][i]:.3f} pm {errs[-1][i]:.3f}\n'
                  f'grd = {prms[-2][i]:.3f} pm {errs[-2][i]:.3f}\n')
            print(f'fitErr = {fitErr[i]:.3f}\n')
            print(f'dat = {prms_avg[-1]:.3f} pm {prms_std[-1]:.3f}\n'
                  f'grd = {prms_avg[-2]:.3f} pm {prms_std[-2]:.3f}\n')
            print(f'daterr = {errs_avg[-1]:.3f}, grderr = {errs_avg[-2]:.3f}\n')
            fig, ax = plt.subplots()
            ax.errorbar(reff1d, dat[i], err[i], color='black', fmt='.', capsize=3)
            ax.errorbar(reffsForLSMAtInterest[i] + reffOfInterest, DatAtInterest[i], DatErAtInterest[i],
                         color='blue', fmt='.', capsize=3)
            ax.plot(reffsForLSMAtInterest[i] + reffOfInterest, datFit[i], color='red')
            ax.errorbar(reffOfInterest, RegDat[i], RegDatEr[i], color='red', capsize=5)
            ax.fill_between(reffsForLSMAtInterest[i] + reffOfInterest, datFit[i] - fitCurveErrs[i],
                             datFit[i] + fitCurveErrs[i],
                             color='green', alpha=0.3)
            idxs_center = np.nanargmin(np.abs(reff1d))
            max = dat[i][idxs_center] + 0.5
            ax.set_ylim(0, max)
            ax.set_xlim(0, np.nanmax(reff1d))
            fnm = f'{fname_base}_{i}.png'
            plot.capsave(fig, '', fnm, os.path.join(figOutDir, fnm))
            plot.check(0.1)

    return reffOfInterest, RegDat, RegDatEr, DatGrad, DatGradEr


def timeSeriesRegGradAtRhoOfInterest(reff2d, rho2d, dat, err, rhoOfInterest, NFit, polyGrad,
                                     fname_base=None, dir_name=None, showfig=True):

    reffsOfInterest, idxsForLSMAtInterest, shiftedReffsForLSMAtInterest = \
        proc.makeReffsForLSMAtRhoOfInterest2d(reff2d, rho2d, rhoOfInterest)

    reffsForLSMAtInterest = shiftedReffsForLSMAtInterest[:, :NFit]
    DatAtInterest = dat[idxsForLSMAtInterest][:, :NFit]
    DatErAtInterest = err[idxsForLSMAtInterest][:, :NFit]

    idxs_sort = proc.argsort(reffsForLSMAtInterest)
    reffsForLSMAtInterest = reffsForLSMAtInterest[idxs_sort]
    DatAtInterest = DatAtInterest[idxs_sort]
    DatErAtInterest = DatErAtInterest[idxs_sort]

    prms, errs, fitErr, datFit, fitCurveErrs = \
        polyN_LSM_v2(reffsForLSMAtInterest, DatAtInterest, polyGrad, DatErAtInterest)
    RegDat = prms[-1]
    RegDatEr = fitErr
    DatGrad = prms[-2]
    DatGradEr = errs[-2]

    if fname_base is not None:

        figOutDir = os.path.join(dir_name, 'fig')
        proc.ifNotMake(figOutDir)

        for i in range(len(reffsForLSMAtInterest)):
            Nrepeat = 1000
            yNoise_set = np.random.normal(datFit[i], fitErr[i], (Nrepeat, len(datFit[i])))
            prmsSamp, errsSamp, sigma_y, y_hut, y_hut_errs = \
                polyN_LSM_v2(np.full((Nrepeat, len(datFit[i])), reffsForLSMAtInterest[i]), yNoise_set, polyGrad)
            prms_avg = np.average(prmsSamp, axis=-1)
            prms_std = np.std(prmsSamp, axis=-1, ddof=1)
            errs_avg = np.average(errsSamp, axis=-1)

            print(i)
            print(f'dat = {prms[-1][i]:.3f} pm {errs[-1][i]:.3f}\n'
                  f'grd = {prms[-2][i]:.3f} pm {errs[-2][i]:.3f}\n')
            print(f'fitErr = {fitErr[i]:.3f}\n')
            print(f'dat = {prms_avg[-1]:.3f} pm {prms_std[-1]:.3f}\n'
                  f'grd = {prms_avg[-2]:.3f} pm {prms_std[-2]:.3f}\n')
            print(f'daterr = {errs_avg[-1]:.3f}, grderr = {errs_avg[-2]:.3f}\n')
            fig, ax = plt.subplots()
            ax.errorbar(reff2d[i], dat[i], err[i], color='black', fmt='.', capsize=3)
            ax.errorbar(reffsForLSMAtInterest[i] + reffsOfInterest[i], DatAtInterest[i], DatErAtInterest[i],
                         color='blue', fmt='.', capsize=3)
            ax.plot(reffsForLSMAtInterest[i] + reffsOfInterest[i], datFit[i], color='red')
            ax.errorbar(reffsOfInterest[i], RegDat[i], RegDatEr[i], color='red', capsize=5)
            ax.fill_between(reffsForLSMAtInterest[i] + reffsOfInterest[i], datFit[i] - fitCurveErrs[i],
                             datFit[i] + fitCurveErrs[i],
                             color='green', alpha=0.3)
            # idxs_center = np.nanargmin(np.abs(reff2d[i]))
            # max = dat[i][idxs_center] + 0.5
            max = np.nanmax(dat[i]) * 1.2
            min = np.nanmin(dat[i]) * 1.2
            ax.set_ylim(np.min([0, min]), np.max([0, max]))
            ax.set_xlim(0, np.nanmax(reff2d))
            fnm = f'{fname_base}_{i}.png'
            plot.capsave(fig, '', fnm, os.path.join(figOutDir, fnm))
            if showfig:
                plot.check(0.001)
            else:
                plot.close(fig)

    return reffsOfInterest, RegDat, RegDatEr, DatGrad, DatGradEr


"""
# def nanWeightedAvg(dat, err):
#
#     areNotNansInDat = (~np.isnan(dat)).astype(np.int8)
#     dat = np.nan_to_num(dat)
#     Wg = 1. / (err ** 2)
#     err = np.nan_to_num(err)
#     Wg = np.nan_to_num(Wg)
#
#     Avg = np.sum(Wg * dat, axis=0) / np.sum(Wg * areNotNansInDat, axis=0)
#     AvgErr = np.sum(Wg * err, axis=0) / np.sum(Wg, axis=0)
#
#     return Avg, AvgErr
"""

def dB(spec, spec_err):
    spec_db = 10 * np.log10(spec)
    spec_err_db = 10 / np.log(10) / spec * spec_err
    return spec_db, spec_err_db


def toZeroMeanTimeSliceEnsemble(xx, NFFT, NEns, NOV, time=None, tout=None):
    xens = toTimeSliceEnsemble(xx, NFFT, NEns, NOV, time=time, tout=tout)
    xensAvg = np.average(xens, axis=-1)
    xensAvg = repeat_and_add_lastdim(xensAvg, NFFT)
    xens -= xensAvg

    return xens

"""
# def makeidxsalongtime(tdat, tout, NSamp):
#
#     idxs_tout = np.argmin(np.abs(np.tile(tdat, (len(tout), 1)) - np.reshape(tout, (len(tout), 1))), axis=-1)
#     ss_base = (- int(0.5 * NSamp + 0.5), int(0.5 * NSamp + 0.5))
#     idxs_samp = np.tile(np.arange(ss_base[0], ss_base[1]), (len(tout), 1)) + np.reshape(idxs_tout, (len(tout), 1))
#
#     return idxs_samp
"""

def toTimeSliceEnsemble(xx, NFFT, NEns, NOV, time=None, tout=None):  # time, tout: 1darray

    print(f'Overlap ratio: {NOV / NFFT * 100:.0f}%\n')

    if time is not None:
        if len(xx) != len(time):
            print('The number of samples is improper. \n')
            sys.exit()
        idxs = proc.makefftsampleidxs(time, tout, NFFT, NEns, NOV)

    else:

        Nsamp = NEns * NFFT - (NEns - 1) * NOV

        if len(xx) != Nsamp:
            print('The number of samples is improper. \n')
            sys.exit()
        else:
            print(f'The number of samples: {Nsamp:d}')

        if NOV != 0:
            idxs = (np.reshape(np.arange(NEns * NFFT), (NEns, NFFT)).T - np.arange(0, NEns * NOV, NOV)).T
        else:
            idxs = np.reshape(np.arange(NEns * NFFT), (NEns, NFFT))

    xens = xx[idxs]

    return xens


def CV(dat):
    datAvg = np.mean(dat, axis=0)
    datExp = np.mean(datAvg)
    datExpErr = np.std(datAvg, ddof=1)
    CV_dat = datExpErr / np.abs(datExp)
    print(f'C.V.={CV_dat * 100:.0f}%')

    return CV_dat


def CV_overlap(NFFT, NEns, NOV):

    NSamp = NEns * NFFT - (NEns - 1) * NOV
    randND = np.random.normal(size = NSamp)
    randND = toTimeSliceEnsemble(randND, NFFT, NEns, NOV)

    rfft_randND = fft.rfft(randND)
    p_randND = np.real(rfft_randND * rfft_randND.conj())
    p_randND[:, 1:-1] *= 2
    CV_randND = CV(p_randND)

    return CV_randND


def CVForBiSpecAna(NFFTs, NEns, NOVs):
    NFFT1, NFFT2, NFFT3 = NFFTs
    NOV1, NOV2, NOV3 = NOVs

    idxMx3, idxNan = makeIdxsForCrossBiSpectrum(NFFTs)

    NSamp1 = NEns * NFFT1 - (NEns - 1) * NOV1
    NSamp2 = NEns * NFFT2 - (NEns - 1) * NOV2
    NSamp3 = NEns * NFFT3 - (NEns - 1) * NOV3
    randND1 = np.random.normal(size=NSamp1)
    randND1 = toTimeSliceEnsemble(randND1, NFFT1, NEns, NOV1)
    randND2 = np.random.normal(size=NSamp2)
    randND2 = toTimeSliceEnsemble(randND2, NFFT2, NEns, NOV2)
    randND3 = np.random.normal(size=NSamp3)
    randND3 = toTimeSliceEnsemble(randND3, NFFT3, NEns, NOV3)
    ND1 = fft.fft(randND1)
    ND2 = fft.fft(randND2)
    ND3 = fft.fft(randND3)

    ND1 = np.reshape(ND1, (NEns, 1, NFFT1))
    ND2 = np.reshape(ND2, (NEns, NFFT2, 1))
    ND3 = ND3[:, idxMx3]

    ND2ND1 = np.matmul(ND2, ND1)
    ND2ND1AbsSq = np.abs(ND2ND1) ** 2
    ND3AbsSq = np.abs(ND3) ** 2

    CV_ND2ND1 = CV(ND2ND1AbsSq)
    CV_ND3 = CV(ND3AbsSq)

    return CV_ND2ND1, CV_ND3


def getWindowAndCoefs(NFFT, window, NEns):

    win, enbw, CG = get_window(NFFT, window)
    CV = get_CV(NEns)

    return win, enbw, CG, CV


def get_window(NFFT, window):

    win = signal.get_window(window, NFFT)
    enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    CG = np.abs(np.sum(win)) / NFFT

    return win, enbw, CG


def get_CV(NEns):

    # CV = CV_overlap(NFFT, NEns, NOV)
    CV = 1./np.sqrt(NEns)

    return CV


def powerTodB(power, powerErr):
    power_db = 10 * np.log10(power)
    powerErr_db = 10 / np.log(10) / power * powerErr

    return power_db, powerErr_db


def fourier_components_1s(xens, dt, NFFT, win):

    rfreq = fft.rfftfreq(NFFT, dt)
    rfft_x = fft.rfft(xens * win)

    return rfreq, rfft_x


def fourier_components_2s(xens, dt, NFFT, win):

    freq = fft.fftfreq(NFFT, dt)
    freq = fft.fftshift(freq)
    fft_x = fft.fft(xens * win)
    fft_x = fft.fftshift(fft_x, axes=-1)

    return freq, fft_x


def NSampleForFFT(NFFT, NEns, NOV):
    return NEns * NFFT - (NEns - 1) * NOV


def NdatForFFT(Nsample, Nsp, NOV):
    return Nsp * Nsample - (Nsp - 1) * NOV


def NEnsFromNSample(NFFT, NOV, Nsamp):
    return (Nsamp - NOV) // (NFFT - NOV)


def NspFromNdat(Nsample, NOV, Ndat):
    return (Ndat - NOV) // (Nsample - NOV)


def bestNfftFromNdat(Nens, OVR, Ndat):
    return 2**int(np.log2(Ndat / (Nens - (Nens - 1) * OVR)))


"""
# def power_spectre_1s(xx, dt, NFFT, window, NEns, NOV):

#     xens = toTimeSliceEnsemble(xx, NFFT, NEns, NOV)
#     win, enbw, CG, CV = getWindowAndCoefs(NFFT, window, NEns, NOV)

#     rfreq, rfft_x = fourier_components_1s(xens, dt, NFFT, win)
#     p_xx = np.real(rfft_x * rfft_x.conj())
#     p_xx[:, 1:-1] *= 2
#     p_xx_ave = np.mean(p_xx, axis=0)
#     p_xx_std = np.std(p_xx, axis=0, ddof=1)
#     p_xx_rerr = p_xx_std / np.abs(p_xx_ave) * CV

#     Fs = 1. / dt
#     psd = p_xx_ave / (Fs * NFFT * enbw * (CG ** 2))
#     psd_err = psd * p_xx_rerr

#     dfreq = 1. / (NFFT * dt)
#     print(f'Power x^2_bar            ={np.sum(xx**2) / len(xx)}')
#     print(f'Power integral of P(f)*df={np.sum(psd * dfreq)}\n')

#     return rfreq, psd, psd_err
"""


def spectrum(t_s, d, Fs_Hz, tstart, tend, NFFT=2**10, ovr=0.5, window="hann", detrend="constant"):

    sp = struct()
    sp.tstart = tstart
    sp.tend = tend

    sp.NFFT = NFFT
    sp.ovr = ovr
    sp.NOV = int(sp.NFFT * sp.ovr)
    sp.window = window

    _, datlist = proc.getTimeIdxsAndDats(t_s, sp.tstart, sp.tend, [t_s, d])
    sp.traw, sp.draw = datlist
    sp.dF = Fs_Hz / sp.NFFT

    return_onesided = not np.iscomplexobj(d)

    sp.t = (sp.tstart + sp.tend) / 2
    sp.f, sp.psd = welch(x=sp.draw, fs=Fs_Hz, window=window,
                        nperseg=sp.NFFT, noverlap=sp.NOV,
                        detrend=detrend, scaling="density",
                        average="mean", return_onesided=return_onesided)
    if not return_onesided:
        sp.f = fft.fftshift(sp.f)
        sp.psd = fft.fftshift(sp.psd)
    sp.psddB = 10 * np.log10(sp.psd)

    return sp


def specgram(t_s, d, Fs_Hz, NFFT=2**10, ovr=0., window="hann", NEns=1, detrend="constant"):

    spg = struct()
    spg.NFFT = NFFT
    spg.ovr = ovr
    spg.NOV = int(spg.NFFT * spg.ovr)
    spg.window = window
    spg.NEns = NEns
    spg.size = len(t_s)
    spg.NSamp = NSampleForFFT(NFFT=spg.NFFT, NEns=spg.NEns, NOV=spg.NOV)
    spg.Nsp = spg.size // spg.NSamp

    return_onesided = ~ np.iscomplexobj(spg.draw)

    spg.tarray = t_s[:spg.Nsp * spg.NSamp].reshape((spg.Nsp, spg.NSamp))
    spg.t = spg.tarray.mean(axis=-1)
    spg.darray = d[:spg.Nsp * spg.NSamp].reshape((spg.Nsp, spg.NSamp))
    spg.f, spg.psd = welch(x=spg.darray, fs= Fs_Hz, window="hann",
                            nperseg=spg.NFFT, noverlap=spg.NOV,
                            return_onesided=return_onesided,
                            detrend=detrend, scaling="density",
                            axis=-1, average="mean")

    if not return_onesided:
        spg.f = fft.fftshift(spg.f)
        spg.psd = fft.fftshift(spg.psd, axes=-1)
    spg.psdamp = np.sqrt(spg.psd)
    spg.psddB = 10 * np.log10(spg.psd)

    spg.dF = Fs_Hz / spg.NFFT
    spg.fmax = Fs_Hz / 2

    return spg


def cross_spectrum(t_s, d1, d2, Fs_Hz, tstart, tend, NFFT=2**10, ovr=0.5, window="hann", detrend="constant", unwrap_phase=False):

    cs = struct()
    cs.tstart = tstart
    cs.tend = tend

    cs.NFFT = NFFT
    cs.ovr = ovr
    cs.NOV = int(cs.NFFT * cs.ovr)
    cs.window = window

    _, datlist = proc.getTimeIdxsAndDats(t_s, cs.tstart, cs.tend, [t_s, d1, d2])
    _, cs.d1raw, cs.d2raw = datlist
    cs.dF = Fs_Hz / cs.NFFT
    cs.NEns = NEnsFromNSample(cs.NFFT, cs.NOV, cs.d1raw.size)

    return_onesided = not (np.iscomplexobj(d1) or np.iscomplexobj(d2))

    cs.t = (cs.tstart + cs.tend) / 2
    cs.f, cs.csd = signal.csd(x=cs.draw1, y=cs.draw2, fs=Fs_Hz, window=window,
                                nperseg=cs.NFFT, noverlap=cs.NOV, nfft=None, 
                                detrend=detrend, scaling="density",
                                average="mean", return_onesided=return_onesided)
    _, cs.cohsq = signal.coherence(x=cs.draw1, y=cs.draw2, fs=Fs_Hz, window=window,
                                    nperseg=cs.NFFT, noverlap=cs.NOV, nfft=None, 
                                    detrend=detrend, scaling="density",
                                    average="mean", return_onesided=return_onesided)
    if not return_onesided:
        cs.f = fft.fftshift(cs.f)
        cs.csd = fft.fftshift(cs.csd)
        cs.cohsq = fft.fftshift(cs.cohsq)
    cs.psd = np.amp(cs.csd)
    cs.phase = phase(cs.csd, unwrap=unwrap_phase)
    cs.coh = np.sqrt(cs.cohsq)

    return cs


# def power_spectre_2s(xx, dt, NFFT, window, NEns, NOV):

    # print(f'Overlap ratio: {NOV/NFFT*100:.0f}%\n')

    # Nsamp = NEns * NFFT - (NEns - 1) * NOV
    # if len(xx) != Nsamp:
    #     print('The number of samples is improper. \n')
    #     sys.exit()
    # else:
    #     print(f'The number of samples: {Nsamp:d}')

    # if NOV != 0:
    #     idxs = (np.reshape(np.arange(NEns * NFFT), (NEns, NFFT)).T - np.arange(0, NEns * NOV, NOV)).T
    # else:
    #     idxs = np.reshape(np.arange(NEns * NFFT), (NEns, NFFT))
    # xens = xx[idxs]

    # win = signal.get_window(window, NFFT)
    # enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    # CG = np.abs(np.sum(win)) / NFFT

    # CV = CV_overlap(NFFT, NEns, NOV)

    # freq = fft.fftshift(fft.fftfreq(NFFT, dt))

    # fft_x = fft.fftshift(fft.fft(xens * win), axes=1)

    # p_xx = np.real(fft_x * fft_x.conj())
    # p_xx_ave = np.mean(p_xx, axis=0)
    # p_xx_err = np.std(p_xx, axis=0, ddof=1)
    # p_xx_rerr = p_xx_err / np.abs(p_xx_ave) * CV

    # Fs = 1. / dt
    # psd = p_xx_ave / (Fs * NFFT * enbw * (CG ** 2))
    # psd_err = np.abs(psd) * p_xx_rerr

    # dfreq = 1. / (NFFT * dt)
    # print(f'Power x^2_bar            ={np.sum(np.real(xx*np.conj(xx))) / Nsamp}')
    # print(f'Power integral of P(f)*df={np.sum(psd * dfreq)}')

    # return freq, psd, psd_err


def get_intermediate(xx, Nsample):

    Nremain = len(xx) - Nsample
    idx_s = Nremain // 2

    return xx[idx_s: idx_s + Nsample]


def power_spectre_2s_v2(tt, xx, NFFT, window, OVR):

    NOV = int(NFFT * OVR)
    NEns = NEnsFromNSample(NFFT, NOV, len(tt))
    Nsample = NSampleForFFT(NFFT, NEns, NOV)

    tt = get_intermediate(tt, Nsample)
    xx = get_intermediate(xx, Nsample)
    dt = tt[1] - tt[0]

    tisp, freq, psd, psd_std, psd_err = power_spectrogram_2s_v4(tt, xx, dt, NFFT, window, NEns, NOV)

    return tisp[0], freq, psd[0], psd_std[0], psd_err[0]


def lowpass(x, samplerate, fp, fs, gpass=3, gstop=16):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y                                      #フィルタ後の信号を返す


def bandPass(xx, sampleRate, fp, fs, gpass=3, gstop=16, cut=False):
    fn = sampleRate / 2                           # ナイキスト周波数
    wp = fp / fn                                  # ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  # ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)        # オーダーとバターワースの正規化周波数を計算
    if cut:
        b, a = signal.butter(N, Wn, "bandstop")  # フィルタ伝達関数の分子と分母を計算
    else:
        b, a = signal.butter(N, Wn, "band")           # フィルタ伝達関数の分子と分母を計算
    yy = signal.filtfilt(b, a, xx)                 # 信号に対してフィルタをかける
    return yy


def notch(xx, samplerate, f0, Q):  # Q = f0 / df, where df = frequency range with
    b, a = signal.iirnotch(f0, Q, samplerate)
    yy = signal.filtfilt(b, a, xx)
    return yy


def filter_butterworth(xx, samplingFreq, cutoffFreq, filtertype, order):
    # filtertype: "low", "high", "band", "bandstop"
    # cutoffFreq should be list [fl, fh] in the case of "band" or "bandstop"

    fnyq = samplingFreq / 2
    Wn = cutoffFreq / fnyq
    b, a = signal.butter(order, Wn, filtertype)
    yy = signal.filtfilt(b, a, xx)

    return yy


def highpass(x, samplerate, fp, fs, gpass=3, gstop=16):
    fn = samplerate / 2  # ナイキスト周波数
    wp = fp / fn  # ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "high")  # フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
    return y  # フィルタ後の信号を返す


def THDF(rfreq, rpsd, FF, maxH):

    func_psd = interpolate.interp1d(rfreq, rpsd)
    Po1 = func_psd(FF)
    PoH = 0
    for i in range(2, maxH+1):
        PoH += func_psd(i * FF)
    THDF = np.sqrt(PoH / Po1)
    print(f'THD_F={THDF*100:.2f}%\n'
          f'up to {maxH:d} order harmonics\n')

    return THDF


def phase_delay(dat, ref, dt, wavfreq):

    # Auto-correlation
    # reference https://qiita.com/inoory/items/3ea2d447f6f1e8c40ffa
    ref_eq = ref - ref.mean()
    dat_eq = dat - dat.mean()
    corr = np.correlate(dat_eq, ref_eq, 'full')
    idx_delay = corr.argmax() - (len(ref_eq) - 1)  # delay of dat from ref
    t_delay = idx_delay * dt
    phdeg_delay = 360*wavfreq*t_delay  # degree

    if phdeg_delay > 180:
        phdeg_delay -= 360
    elif phdeg_delay < -180:
        phdeg_delay += 360

    return phdeg_delay


def calibIQComp2(datI, datQ, VAR, VOS_I, VOS_Q, phDif):
    phDifErr = phDif - 90

    datICalib = datI - VOS_I
    datQCalib = (VAR * (datQ - VOS_Q) + datICalib * np.sin(np.radians(phDifErr))) / np.cos(np.radians(phDifErr))

    return datICalib, datQCalib


"""
# def power_spectrogram_1s(ti, xx, dt, NFFT, window, Ndiv, Ntisp):
#
#     idxs = np.arange(0, NFFT * Ndiv * Ntisp)
#     idxs = idxs.reshape((Ntisp, Ndiv, NFFT))
#
#     dtisp = Ndiv * NFFT * dt
#     tisp = ti[idxs.T[0][0]] + 0.5 * dtisp
#     xens = xx[idxs]
#
#     win = signal.get_window(window, NFFT)
#     enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
#     CG = np.abs(np.sum(win)) / NFFT
#
#     div_CV = np.sqrt(1. / Ndiv)  # 分割平均平滑化による相対誤差の変化率
#     sp_CV = div_CV
#
#     rfreq = fft.rfftfreq(NFFT, dt)
#
#     rfft_x = fft.rfft(xens * win)
#
#     p_xx = np.real(rfft_x * rfft_x.conj())
#     p_xx[:, :, 1:-1] *= 2
#     p_xx_ave = np.mean(p_xx, axis=1)
#     p_xx_err = np.std(p_xx, axis=1, ddof=1)
#     p_xx_rerr = p_xx_err / np.abs(p_xx_ave) * sp_CV
#
#     Fs = 1. / dt
#     psd = p_xx_ave / (Fs * NFFT * enbw * (CG ** 2))
#     psd_err = np.abs(psd) * p_xx_rerr
#
#     dfreq = 1. / (NFFT * dt)
#     print(f'Power x^2_bar             = {np.sum(xx[idxs][0] ** 2) / (NFFT * Ndiv):.3f}V^2 '
#           f'@{tisp[0]:.3f}+-{0.5 * dtisp:.3f}s')
#     print(f'Power integral of P(f)*df = {np.sum(psd[0] * dfreq):.3f}V^2'
#           f' @{tisp[0]:.3f}+-{0.5 * dtisp:.3f}s')
#
#     return tisp, rfreq, psd, psd_err
"""


def power_spectrogram_1s(ti, xx, dt, NFFT, window, NEns, NOV):

    print(f'Overlap ratio: {NOV/NFFT*100:.0f}%\n')

    Nsamp = NEns * NFFT - (NEns - 1) * NOV
    Nsp = int(len(ti) / Nsamp + 0.5)

    if len(xx) % Nsamp != 0 or len(ti) % Nsamp != 0 or len(xx) != len(ti):
        print('The number of data points is improper. \n')
        sys.exit()
    else:
        print(f'The number of samples a spectrum: {Nsamp:d}')
        print(f'The number of spectra: {Nsp:d}\n')

    if NOV != 0:
        tmp = (np.reshape(np.arange(NEns * NFFT), (NEns, NFFT)).T - np.arange(0, NEns * NOV, NOV)).T
    else:
        tmp = np.reshape(np.arange(NEns * NFFT), (NEns, NFFT))
    idxs = np.transpose(np.tile(tmp, (Nsp, 1, 1)).T + np.arange(0, Nsp*Nsamp, Nsamp))

    dtisp = Nsamp * dt
    tisp = ti[idxs.T[0][0]] + 0.5 * dtisp
    xens = xx[idxs]

    win = signal.get_window(window, NFFT)
    enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    CG = np.abs(np.sum(win)) / NFFT

    CV = CV_overlap(NFFT, NEns, NOV)

    rfreq = fft.rfftfreq(NFFT, dt)

    fft_x = fft.rfft(xens * win)

    p_xx = np.real(fft_x * fft_x.conj())
    p_xx_ave = np.mean(p_xx, axis=1)
    p_xx_err = np.std(p_xx, axis=1, ddof=1)
    p_xx_rerr = p_xx_err / np.abs(p_xx_ave) * CV

    Fs = 1. / dt
    psd = p_xx_ave / (Fs * NFFT * enbw * (CG ** 2))
    psd_err = np.abs(psd) * p_xx_rerr

    dfreq = 1. / (NFFT * dt)
    print(f'Power: Time average of x(t)^2 = {np.sum(np.abs(xx[0:Nsamp])**2) / Nsamp:.6f} V^2 '
          f'@{tisp[0]:.3f}+-{0.5 * dtisp:.3f} s')
    print(f'Power: Integral of P(f)       = {np.sum(psd[0] * dfreq):.6f} V^2 '
          f'@{tisp[0]:.3f}+-{0.5 * dtisp:.3f} s')

    return tisp, rfreq, psd, psd_err


def power_spectrogram_2s(ti, xx, dt, NFFT, window, NEns, NOV):

    proc.suggestNewVer(2, 'power_spectrogram_2s')

    print(f'Overlap ratio: {NOV/NFFT*100:.0f}%\n')

    Nsamp = NEns * NFFT - (NEns - 1) * NOV
    Nsp = int(len(ti) / Nsamp + 0.5)

    if len(xx) % Nsamp != 0 or len(ti) % Nsamp != 0 or len(xx) != len(ti):
        print('The number of data points is improper. \n')
        sys.exit()
    else:
        print(f'The number of samples a spectrum: {Nsamp:d}')
        print(f'The number of spectra: {Nsp:d}\n')

    if NOV != 0:
        tmp = (np.reshape(np.arange(NEns * NFFT), (NEns, NFFT)).T - np.arange(0, NEns * NOV, NOV)).T
    else:
        tmp = np.reshape(np.arange(NEns * NFFT), (NEns, NFFT))
    idxs = np.transpose(np.tile(tmp, (Nsp, 1, 1)).T + np.arange(0, Nsp*Nsamp, Nsamp))

    dtisp = Nsamp * dt
    tisp = ti[idxs.T[0][0]] + 0.5 * dtisp
    xens = xx[idxs]

    win = signal.get_window(window, NFFT)
    enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    CG = np.abs(np.sum(win)) / NFFT

    CV = CV_overlap(NFFT, NEns, NOV)

    freq = fft.fftshift(fft.fftfreq(NFFT, dt))

    fft_x = fft.fftshift(fft.fft(xens * win), axes=2)

    p_xx = np.real(fft_x * fft_x.conj())
    p_xx_ave = np.mean(p_xx, axis=1)
    p_xx_err = np.std(p_xx, axis=1, ddof=1)
    p_xx_rerr = p_xx_err / np.abs(p_xx_ave) * CV

    Fs = 1. / dt
    psd = p_xx_ave / (Fs * NFFT * enbw * (CG ** 2))
    psd_err = np.abs(psd) * p_xx_rerr

    dfreq = 1. / (NFFT * dt)
    print(f'Power: Time average of x(t)^2 = {np.sum(np.abs(xx[0:Nsamp])**2) / Nsamp:.6f} V^2 '
          f'@{tisp[0]:.3f}+-{0.5 * dtisp:.3f} s')
    print(f'Power: Integral of P(f)       = {np.sum(psd[0] * dfreq):.6f} V^2 '
          f'@{tisp[0]:.3f}+-{0.5 * dtisp:.3f} s')

    return tisp, freq, psd, psd_err


def Nspectra(time, NFFT, NEns, NOV):

    Nsample = NEns * NFFT - (NEns - 1) * NOV
    Nspectra = len(time) // Nsample
    Ndat = Nsample * Nspectra

    print(f'The number of samples per a spectrum: {Nsample:d}')
    print(f'The number of spectra: {Nspectra:d}\n')

    return Nspectra, Nsample, Ndat


def Nspectra_v2(time, NFFT, NEns, NOV):

    Nsample = NSampleForFFT(NFFT, NEns, NOV)
    Nsp = NspFromNdat(Nsample, NOV, len(time))
    Ndat = NdatForFFT(Nsample, Nsp, NOV)

    print(f'The number of samples per a spectrum: {Nsample:d}')
    print(f'The number of spectra: {Nsp:d}\n')

    return Nsp, Nsample, Ndat


def power_spectrogram_2s_v2(ti, xx, dt, NFFT, window, NEns, NOV):

    proc.suggestNewVer(3, 'power_spectrogram_2s')

    print(f'Overlap ratio: {NOV / NFFT * 100:.0f}%\n')

    Nsp, Nsamp, Ndat = Nspectra(ti, NFFT, NEns, NOV)

    if len(xx) % Nsamp != 0 or len(ti) % Nsamp != 0 or len(xx) != len(ti):
        print('The number of data points is improper. \n')
        sys.exit()
    else:
        print(f'The number of samples a spectrum: {Nsamp:d}')
        print(f'The number of spectra: {Nsp:d}\n')

    if NOV != 0:
        tmp = (np.reshape(np.arange(NEns * NFFT), (NEns, NFFT)).T - np.arange(0, NEns * NOV, NOV)).T
    else:
        tmp = np.reshape(np.arange(NEns * NFFT), (NEns, NFFT))
    idxs = np.transpose(np.tile(tmp, (Nsp, 1, 1)).T + np.arange(0, Nsp * Nsamp, Nsamp))

    dtisp = Nsamp * dt
    tisp = ti[idxs.T[0][0]] + 0.5 * dtisp
    xens = xx[idxs]

    win = signal.get_window(window, NFFT)
    enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    CG = np.abs(np.sum(win)) / NFFT

    # CV = CV_overlap(NFFT, NEns, NOV)
    CV = 1./np.sqrt(NEns)

    freq = fft.fftshift(fft.fftfreq(NFFT, dt))

    fft_x = fft.fftshift(fft.fft(xens * win), axes=2)

    p_xx = np.real(fft_x * fft_x.conj())
    p_xx_ave = np.mean(p_xx, axis=1)
    p_xx_std = np.std(p_xx, axis=1, ddof=1)
    p_xx_rerr = p_xx_std / np.abs(p_xx_ave) * CV

    Fs = 1. / dt
    psd = p_xx_ave / (Fs * NFFT * enbw * (CG ** 2))
    psd_std = p_xx_std / (Fs * NFFT * enbw * (CG ** 2))
    psd_err = np.abs(psd) * p_xx_rerr

    dfreq = 1. / (NFFT * dt)
    print(f'Power: Time average of x(t)^2 = {np.sum(np.abs(xx[0:Nsamp]) ** 2) / Nsamp:.6f} V^2 '
          f'@{tisp[0]:.3f}+-{0.5 * dtisp:.3f} s')
    print(f'Power: Integral of P(f)       = {np.sum(psd[0] * dfreq):.6f} V^2 '
          f'@{tisp[0]:.3f}+-{0.5 * dtisp:.3f} s')

    return tisp, freq, psd, psd_std, psd_err


def make_idxs_for_spectrogram(NFFT, NEns, NOV, Nsp):

    if NOV != 0:
        tmp = (np.reshape(np.arange(NEns * NFFT), (NEns, NFFT)).T - np.arange(0, NEns * NOV, NOV)).T
    else:
        tmp = np.reshape(np.arange(NEns * NFFT), (NEns, NFFT))
    Nsamp = NSampleForFFT(NFFT, NEns, NOV)
    idxs = np.transpose(np.tile(tmp, (Nsp, 1, 1)).T + np.arange(0, Nsp * Nsamp, Nsamp))

    return idxs


def make_idxs_for_spectrogram_v2(Nfft, Nens, Nov, Nsp):

    tmp1 = np.reshape(np.arange(Nsp*Nens*Nfft), (Nsp, Nens, Nfft))
    tmp2 = np.reshape(repeat_and_add_lastdim(Nov * np.arange(Nens * Nsp), Nfft), (Nsp, Nens, Nfft))
    idxs = tmp1 - tmp2

    return idxs


def make_idxs_for_spectrogram_wo_EnsembleAVG(NFFT, NOV, Nsp):

    if NOV != 0:
        tmp = (np.reshape(np.arange(Nsp * NFFT), (Nsp, NFFT)).T - np.arange(0, Nsp * NOV, NOV)).T
    else:
        tmp = np.reshape(np.arange(Nsp * NFFT), (Nsp, NFFT))
    Ndat = NSampleForFFT(NFFT, Nsp, NOV)
    idxs = np.transpose(np.tile(tmp, (1, 1)).T + np.arange(0, Ndat, Ndat))

    return idxs


def time_for_spectrogram(time, idxs_for_spectrogram):

    tisp = np.average(np.array([time[idxs_for_spectrogram][:, 0, 0], time[idxs_for_spectrogram][:, -1, -1]]), axis=0)

    return tisp


def time_for_spectrogram_wo_ensembleAVG(time, idxs_for_spectrogram):

    tisp = np.average(time[idxs_for_spectrogram], axis=-1)

    return tisp


def get_Sxx(fft_x, CV):

    Sxx = np.real(fft_x * fft_x.conj())
    Sxx_avg = np.mean(Sxx, axis=-2)
    Sxx_std = np.std(Sxx, axis=-2, ddof=1)
    Sxx_err = Sxx_std * CV

    return Sxx_avg, Sxx_std, Sxx_err


def get_psd(fft_x, CV, dt, NFFT, enbw, CG):

    Sxx_ave, Sxx_std, Sxx_err = get_Sxx(fft_x, CV)
    Sxx_rerr = Sxx_err / np.abs(Sxx_ave)

    Fs = 1. / dt
    psd = Sxx_ave / (Fs * NFFT * enbw * (CG ** 2))
    psd_std = Sxx_std / (Fs * NFFT * enbw * (CG ** 2))
    psd_err = np.abs(psd) * Sxx_rerr

    return psd, psd_std, psd_err


def check_power(NFFT, dt, xx, Nsamp, tisp, dtisp, psd):

    dfreq = 1. / (NFFT * dt)
    print(f'Power: Time average of x(t)^2 = {np.sum(np.abs(xx[0:Nsamp]) ** 2) / Nsamp:.6f} V^2 '
          f'@{tisp[0]:.3f}+-{0.5 * dtisp:.3f} s')
    print(f'Power: Integral of P(f)       = {np.sum(psd[0] * dfreq):.6f} V^2 '
          f'@{tisp[0]:.3f}+-{0.5 * dtisp:.3f} s')

    return


def power_spectrogram_2s_v3(ti, xx, dt, NFFT, window, NEns, NOV):

    print(f'Overlap ratio: {NOV / NFFT * 100:.0f}%\n')

    Nsp, Nsamp, Ndat = Nspectra(ti, NFFT, NEns, NOV)
    ti = ti[:Ndat]
    xx = xx[:Ndat]

    idxs = make_idxs_for_spectrogram(NFFT, NEns, NOV, Nsp)
    tisp = time_for_spectrogram(ti, idxs)
    xens = xx[idxs]

    win, enbw, CG, CV = getWindowAndCoefs(NFFT, window, NEns, NOV)

    freq, fft_x = fourier_components_2s(xens, dt, NFFT, win)
    psd, psd_std, psd_err = get_psd(fft_x, CV, dt, NFFT, enbw, CG)

    dtisp = Nsamp * dt
    check_power(NFFT, dt, xx, Nsamp, tisp, dtisp, psd)

    return tisp, freq, psd, psd_std, psd_err


def power_spectrogram_2s_v4(ti, xx, dt, NFFT=2**10, window="hann", NEns=20, NOV=2**9):

    print(f'Overlap ratio: {NOV / NFFT * 100:.0f}%\n')

    Nsp, Nsamp, Ndat = Nspectra_v2(ti, NFFT, NEns, NOV)
    ti = ti[:Ndat]
    xx = xx[:Ndat]

    idxs = make_idxs_for_spectrogram_v2(NFFT, NEns, NOV, Nsp)
    tisp = time_for_spectrogram(ti, idxs)
    xens = xx[idxs]

    xavg = repeat_and_add_lastdim(np.average(xens, axis=-1), NFFT)
    xens -= xavg

    win, enbw, CG, CV = getWindowAndCoefs(NFFT, window, NEns, NOV)

    freq, fft_x = fourier_components_2s(xens, dt, NFFT, win)
    psd, psd_std, psd_err = get_psd(fft_x, CV, dt, NFFT, enbw, CG)

    dtisp = Nsamp * dt
    check_power(NFFT, dt, xx, Nsamp, tisp, dtisp, psd)

    return tisp, freq, psd, psd_std, psd_err


def power_spectrogram_2s_v5(ti, xx, NFFT=2**10, ovr=0.5, window="hann", NEns=20):

    o = struct()

    o.NFFT = NFFT
    o.ovr = ovr
    o.window = window
    o.NEns = NEns

    o.NOV = int(NFFT * ovr)
    o.dt = ti[1] - ti[0]

    print(f'Overlap ratio: {o.NOV / NFFT * 100:.0f}%\n')

    o.Nsp, o.Nsamp, o.Ndat = Nspectra_v2(ti, NFFT, NEns, o.NOV)
    ti = ti[:o.Ndat]
    xx = xx[:o.Ndat]

    o.ti = ti
    o.xx = xx

    idxs = make_idxs_for_spectrogram_v2(NFFT, NEns, o.NOV, o.Nsp)
    o.tsp = time_for_spectrogram(ti, idxs)
    xens = xx[idxs]

    xavg = repeat_and_add_lastdim(np.average(xens, axis=-1), NFFT)
    xens -= xavg

    o.win, o.enbw, o.CG, o.CV = getWindowAndCoefs(NFFT, window, NEns, o.NOV)

    o.freq, o.fft_x = fourier_components_2s(xens, o.dt, NFFT, o.win)
    o.psd, o.psd_std, o.psd_err = get_psd(o.fft_x, o.CV, o.dt, NFFT, o.enbw, o.CG)

    o.dtsp = o.tsp[1] - o.tsp[0]
    o.dfreq = 1. / (o.Nsamp * o.dt)
    check_power(NFFT, o.dt, xx, o.Nsamp, o.tsp, o.dtsp, o.psd)

    return o


def calib_dbs9O(chDiag, pathCalib, Idat, Qdat):
    dictIF = {'27.7G': 40, '29.1G': 80, '30.5G': 120, '32.0G': 160,
              '33.4G': 200, '34.8G': 240, '36.9G': 300, '38.3G': 340}
    frLO = dictIF[chDiag]
    lvLO = float(input('LO Power [dBm] >>> '))
    vgaLO = float(input('LO VGA [V] >>> '))
    vgaRF = float(input('RF VGA [V] >>> '))
    calibPrms_df = read.calibPrms_df(pathCalib)
    VAR, VOS_I, VOS_Q, phDif = calibPrms_df.loc[(frLO, lvLO, vgaLO, vgaRF)]

    Idat, Qdat = calibIQComp2(Idat, Qdat, VAR, VOS_I, VOS_Q, phDif)

    return Idat, Qdat


def interpolate_nan(array):
    nans, x = np.isnan(array), lambda z: z.nonzero()[0]
    array[nans] = np.interp(x(nans), x(~nans), array[~nans])
    return array


def cross_correlation_analysis(sig, ref, dt, window_len=1, mode="ccc"):
    # mode = "envelope", "ccc"
    # cc.delay (or cc.lags): sig's delay (or lag) time to ref

    cc = cross_correlation(sig, ref, dt, mode="full")
    idx0 = np.argmin(np.abs(cc.lags - 0))
    cc.ccc0 = cc.ccc[idx0]

    if np.iscomplexobj(cc.ccc):
        cc.ccc_env = np.abs(cc.ccc)
    else:
        cc.ccc_env = envelope(cc.ccc)

    if mode == "envelope":
        cc.idx_max, cc.idx_min, cc.ccc_max, cc.ccc_min, cc.delay_max, cc.delay_min \
            = delay_by_ccc(cc.lags, cc.ccc_env, window_len=window_len)
    elif mode == "ccc":
        cc.idx_max, cc.idx_min, cc.ccc_max, cc.ccc_min, cc.delay_max, cc.delay_min \
            = delay_by_ccc(cc.lags, cc.ccc, window_len=window_len)

    return cc


def cross_correlation(sig, ref, dt, mode="same"):

    sig_norm = (sig - sig.mean()) / sig.std()
    ref_norm = (ref - ref.mean()) / ref.std()

    length = len(ref_norm)
    ccc = signal.correlate(sig_norm, ref_norm, mode=mode, method="fft") / length
    idxs_lags = signal.correlation_lags(len(sig_norm), len(ref_norm), mode=mode)
    lags = idxs_lags * dt

    o = struct()
    o.sig = sig
    o.ref = ref
    o.dt = dt
    o.sig_norm = sig_norm
    o.ref_norm = ref_norm
    o.lags = lags   # sig's lag time to ref
    o.ccc = ccc

    return o


def delay_by_ccc(lags, ccc, window_len=1):

    ccc_ma = moving_average(ccc, window_len, mode="same")

    idx_max = np.nanargmax(ccc_ma)
    idx_min = np.nanargmin(ccc_ma)

    ccc_max = ccc[idx_max]
    ccc_min = ccc[idx_min]
    delay_max = lags[idx_max]
    delay_min = lags[idx_min]

    return idx_max, idx_min, ccc_max, ccc_min, delay_max, delay_min



def cross_correlation_analysis_temporal(sig, ref, tt, Nsample=10000, mode="full", delay_by_ccc_amp=False):

    if len(ref) != len(sig):
        print("ref and sig size should be identical")
        sys.exit()

    Ndat = len(tt)
    Nout = (Ndat - 1) // Nsample - 1

    ccf_list = [0]*Nout
    ccc_list = [0]*Nout
    ccc_amp_list = [0] * Nout
    tout = [0]*Nout
    lag_list = [0]*Nout

    for i in range(Nout):
        tat = tt[1 + Nsample // 2 + i * Nsample: 1 + Nsample // 2 + (i + 1) * Nsample].mean()

        tsig_tlim, sig_tlim, tref_tlim, ref_tlim = get_dat_for_cross_correlation_around_tat(tt, sig, ref, tat, Nsample=Nsample)
        lags, ccf, ccc, ccc_amp, delay = cross_correlation_analysis(sig_tlim, ref_tlim, tsig_tlim, tref_tlim, mode=mode, delay_by_ccc_amp=delay_by_ccc_amp)

        ccf_list[i] = ccf
        ccc_list[i] = ccc
        ccc_amp_list[i] = ccc_amp
        tout[i] = tat
        lag_list[i] = delay

    return np.array(tout), lags, np.array(ccf_list), np.array(ccc_list), np.array(ccc_amp_list), np.array(lag_list)


def get_dat_for_cross_correlation_around_tat(tt, sig, ref, tat, Nsample=10000):

    idx_at = np.argmin(np.abs(tt - tat))
    sig_tlim = sig[idx_at - Nsample: idx_at + Nsample]
    tsig_tlim = tt[idx_at - Nsample: idx_at + Nsample]
    ref_tlim = ref[idx_at - Nsample // 2: idx_at + Nsample // 2]
    tref_tlim = tt[idx_at - Nsample // 2: idx_at + Nsample // 2]

    return tsig_tlim, sig_tlim, tref_tlim, ref_tlim


def corrcoef_series(time, sig1, sig2, Nsamp):

    Ncorr = len(time) - Nsamp + 1

    idxs_corr = np.full((Ncorr, Nsamp), np.arange(Nsamp)).T
    idxs_corr = idxs_corr + np.arange(Ncorr)
    idxs_corr = idxs_corr.T
    time_cor = np.average(time[idxs_corr], axis=-1)
    sig1_cor = sig1[idxs_corr]
    sig2_cor = sig2[idxs_corr]

    sig1_AVG = np.repeat(np.reshape(np.nanmean(sig1_cor, axis=-1), (Ncorr, 1)), Nsamp, axis=-1)
    sig2_AVG = np.repeat(np.reshape(np.nanmean(sig2_cor, axis=-1), (Ncorr, 1)), Nsamp, axis=-1)
    sig12_xy = (sig1_cor - sig1_AVG) * (sig2_cor - sig2_AVG)
    sig12_cov = np.nansum(sig12_xy, axis=-1) / (np.count_nonzero(~np.isnan(sig12_xy), axis=-1) - 1)
    sig1_var = np.nanvar(sig1_cor, axis=-1, ddof=1)
    sig2_var = np.nanvar(sig2_cor, axis=-1, ddof=1)
    corrcoef = sig12_cov / np.sqrt(sig1_var * sig2_var)

    return time_cor, corrcoef



"""
# try:
#     lags_inc = np.mean(np.diff(cc.lags))
#     _ik = signal.argrelextrema(ccf_amp, np.less)[0]
#     # _ik = np.argmax(ccf_amp)
#     if len(_ik) >= 2:
#         _iu1 = _ik[np.min(np.where(cc.lags[_ik] > 0))]
#         _il1 = _ik[np.min(np.where(cc.lags[_ik] > 0))-1]
#         if cc.lags[_iu1]-cc.lags[_il1] < 7.5:
#             _il1 = int(_il1 - 15/lags_inc)
#             _iu1 = int(_iu1 + 15/lags_inc)
#     else:
#         _iu1 = len(cc.lags)
#         _il1 = 0
#     walag = np.average(cc.lags, weights=ccf_amp)
#     inip = [np.max(ccf_amp), walag, walag, 0.]
#     popt, pcov = optimize.curve_fit(CCFfit, cc.lags[_il1:_iu1],
#                                     cc.ccf[_il1:_iu1], p0=inip)
#     sigma = popt[-1]
# except RuntimeError:
#     peakcorr, delay, sigma, bg = [np.nan]*4
#     print('error in curve fit ... skipping this set')
"""


def cross_spectre_2s(x, y, Fs, NEns, NFFT, window, NOV):

    # proc.suggestNewVer(2, 'cross_spectre_2s')

    dT = 1. / Fs

    x_arr = toZeroMeanTimeSliceEnsemble(x, NFFT, NEns, NOV)
    y_arr = toZeroMeanTimeSliceEnsemble(y, NFFT, NEns, NOV)

    win = signal.get_window(window, NFFT)
    enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    CG = np.abs(np.sum(win)) / NFFT

    # CV = CV_overlap(NFFT, NEns, NOV)
    CV = 1./np.sqrt(NEns)

    freq = fft.fftshift(fft.fftfreq(NFFT, dT))

    # https://watlab-blog.com/2020/07/24/coherence-function/

    fft_x = fft.fft(x_arr * win)
    fft_x = fft.fftshift(fft_x, axes=(1,))
    fft_y = fft.fft(y_arr * win)
    fft_y = fft.fftshift(fft_y, axes=(1,))

    c_xy = fft_y * fft_x.conj()
    p_xx = np.real(fft_x * fft_x.conj())
    p_yy = np.real(fft_y * fft_y.conj())

    c_xy_ave = np.mean(c_xy, axis=0)
    p_xx_ave = np.mean(p_xx, axis=0)
    p_yy_ave = np.mean(p_yy, axis=0)
    p_xx_err = np.std(p_xx, axis=0, ddof=1)
    p_yy_err = np.std(p_yy, axis=0, ddof=1)
    p_xx_rerr = p_xx_err / p_xx_ave * CV
    p_yy_rerr = p_yy_err / p_yy_ave * CV

    Kxy = np.real(c_xy)
    Qxy = - np.imag(c_xy)
    Kxy_ave = np.mean(Kxy, axis=0)
    Qxy_ave = np.mean(Qxy, axis=0)
    Kxy_err = np.std(Kxy, axis=0, ddof=1)
    Qxy_err = np.std(Qxy, axis=0, ddof=1)
    Kxy_rerr = Kxy_err / Kxy_ave * CV
    Qxy_rerr = Qxy_err / Qxy_ave * CV

    CSDxy = np.abs(c_xy_ave) / (Fs * NFFT * enbw * CG**2)
    cs_err = np.sqrt((Kxy_ave * Kxy_err)**2 + (Qxy_ave * Qxy_err)**2) / \
             np.abs(c_xy_ave)
    cs_rerr = cs_err / np.abs(c_xy_ave) * CV
    CSDxy_err = CSDxy * CV

    coh2 = (np.abs(c_xy_ave) ** 2) / (p_xx_ave * p_yy_ave)
    coh2_rerr = np.sqrt(2 * cs_rerr**2 + p_xx_rerr**2 + p_yy_rerr**2)
    coh2_err = coh2 * coh2_rerr
    cohxy = np.sqrt(coh2)
    cohxy_err = 0.5 / cohxy * coh2_err

    tmp = Qxy_ave / Kxy_ave
    tmp_rerr = np.sqrt(Qxy_rerr ** 2 + Kxy_rerr ** 2)
    tmp_err = np.abs(tmp) * tmp_rerr
    phsxy = np.arctan2(Qxy_ave, Kxy_ave)
    phsxy_err = 1. / (1. + tmp ** 2) * tmp_err

    return freq, CSDxy, CSDxy_err, cohxy, cohxy_err, phsxy, phsxy_err


def get_crossspectra(XX, YY):

    YXconj = YY * XX.conj()
    CrossSpec = np.average(YXconj, axis=-2)
    CrossSpecStd = np.std(YXconj, axis=-2, ddof=1)
    CrossSpecReStd = np.std(np.real(YXconj), axis=-2, ddof=1)
    CrossSpecImStd = np.std(np.imag(YXconj), axis=-2, ddof=1)

    return CrossSpec, CrossSpecStd, CrossSpecReStd, CrossSpecImStd


def get_Sxy(XX, YY, CV):

    Sxy = YY * XX.conj()
    Sxy_avg = np.average(Sxy, axis=-2)
    Sxy_std = np.std(Sxy, axis=-2, ddof=1)
    Sxy_Re_std = np.std(np.real(Sxy), axis=-2, ddof=1)
    Sxy_Im_std = np.std(np.imag(Sxy), axis=-2, ddof=1)

    Sxy_err = Sxy_std * CV
    Sxy_Re_err = Sxy_Re_std * CV
    Sxy_Im_err = Sxy_Im_std * CV

    return Sxy_avg, Sxy_std, Sxy_err, Sxy_Re_std, Sxy_Re_err, Sxy_Im_std, Sxy_Im_err


def get_coherenceSq(CrossSpec, CrossSpecStd, XX, YY, CV):

    XAbsSq, XAbsSqAvg, XAbsSqStd, XAbsSqErr = get_Sxx(XX, CV)
    YAbsSq, YAbsSqAvg, YAbsSqStd, YAbsSqErr = get_Sxx(XX, CV)
    XAbsSqRer = XAbsSqErr / XAbsSqAvg
    YAbsSqRer = YAbsSqErr / YAbsSqAvg

    CrossSpecAbsSq = np.real(CrossSpec * CrossSpec.conj())
    CrossSpecAbsSqStd = 2 * np.abs(CrossSpec) * CrossSpecStd
    CrossSpecAbsSqRer = CrossSpecAbsSqStd / CrossSpecAbsSq * CV

    cohSq = CrossSpecAbsSq / (XAbsSqAvg * YAbsSqAvg)
    cohSqRer = np.sqrt(CrossSpecAbsSqRer ** 2 + XAbsSqRer ** 2 + YAbsSqRer ** 2)
    cohSqErr = cohSq * cohSqRer

    return cohSq, cohSqErr


def get_coherenceSq_v2(Sxy_avg, Sxy_err, Sxx_avg, Sxx_err, Syy_avg, Syy_err):

    Sxy_avg_sq = np.real(Sxy_avg * Sxy_avg.conj())
    Sxy_avg_sq_err = 2 * np.abs(Sxy_avg) * Sxy_err

    Sxy_avg_sq_rer = Sxy_avg_sq_err / Sxy_avg_sq
    Sxx_rer = Sxx_err / Sxx_avg
    Syy_rer = Syy_err / Syy_avg

    coh_sq = Sxy_avg_sq / (Sxx_avg * Syy_avg)
    coh_sq_rer = np.sqrt(Sxy_avg_sq_rer ** 2 + Sxx_rer ** 2 + Syy_rer ** 2)
    coh_sq_err = coh_sq * coh_sq_rer

    return coh_sq, coh_sq_err


def get_crossPhase(CrossSpec, CrossSpecReStd, CrossSpecImStd, CV):

    CrossSpecRe = np.real(CrossSpec)
    CrossSpecIm = np.imag(CrossSpec)

    CrossSpecReRer = CrossSpecReStd / CrossSpecRe * CV
    CrossSpecImRer = CrossSpecImStd / CrossSpecIm * CV

    CrossSpecImRe = CrossSpecIm / CrossSpecRe
    CrossSpecImReRer = np.sqrt(CrossSpecImRer ** 2 + CrossSpecReRer ** 2)
    CrossSpecImReErr = np.abs(CrossSpecImRe * CrossSpecImReRer)
    crossPhs = np.unwrap(np.arctan2(CrossSpecIm, CrossSpecRe))
    crossPhsErr = 1. / (1. + CrossSpecImRe ** 2) * CrossSpecImReErr

    return crossPhs, crossPhsErr


def get_crossphase_v2(Sxy_avg, Sxy_Re_err, Sxy_Im_err):

    Sxy_avg_Re = np.real(Sxy_avg)
    Sxy_avg_Im = np.imag(Sxy_avg)

    Sxy_Re_rer = Sxy_Re_err / Sxy_avg_Re
    Sxy_Im_rer = Sxy_Im_err / Sxy_avg_Im

    tmp = Sxy_avg_Im / Sxy_avg_Re
    tmp_rer = np.sqrt(Sxy_Im_rer ** 2 + Sxy_Re_rer ** 2)
    tmp_err = np.abs(tmp * tmp_rer)
    if Sxy_avg.size == 1:
        crossPhs = np.arctan2(Sxy_avg_Im, Sxy_avg_Re)
    else:
        crossPhs = np.unwrap(np.arctan2(Sxy_avg_Im, Sxy_avg_Re))
    crossPhsErr = 1. / (1. + tmp ** 2) * tmp_err

    return crossPhs, crossPhsErr


def crossSpectralAnalysis_2s_v2(xx, yy, dts, NFFTs, NEns, window, NOVs):

    dtx, dty = dts
    NFFTx, NFFTy = NFFTs
    NOVx, NOVy = NOVs

    xens = toZeroMeanTimeSliceEnsemble(xx, NFFTx, NEns, NOVx)
    winx, enbwx, CGx, CV = getWindowAndCoefs(NFFTx, window, NEns, NOVx)
    freqx, XX = fourier_components_2s(xens, dtx, NFFTx, winx)

    yens = toZeroMeanTimeSliceEnsemble(yy, NFFTy, NEns, NOVy)
    winy, enbwy, CGy, CV = getWindowAndCoefs(NFFTy, window, NEns, NOVy)
    freqy, YY = fourier_components_2s(yens, dty, NFFTy, winy)

    # https://watlab-blog.com/2020/07/24/coherence-function/

    if NFFTx > NFFTy:
        freq = freqy
        XX = XX[:, (NFFTx - NFFTy)//2:(NFFTx + NFFTy)//2]
    elif NFFTy > NFFTx:
        freq = freqx
        YY = YY[:, (NFFTy - NFFTx) // 2:(NFFTy + NFFTx) // 2]
    else:
        freq = freqx

    CrossSpec, CrossSpecStd, CrossSpecReStd, CrossSpecImStd = get_crossspectra(XX, YY)

    cohSq, cohSqErr = get_coherenceSq(CrossSpec, CrossSpecStd, XX, YY, CV)
    phs, phsErr = get_crossPhase(CrossSpec, CrossSpecReStd, CrossSpecImStd, CV)

    return freq, cohSq, cohSqErr, phs, phsErr


def get_freq_for_spectrum(NFFT, dt):

    freq = fft.fftfreq(NFFT, dt)
    freq = fft.fftshift(freq)

    return freq


def FFT(xens, win):

    fft_x = fft.fft(xens * win)
    fft_x = fft.fftshift(fft_x, axes=-1)

    return fft_x


def cross_spectrogram_2s(tt, xx, yy, NFFT=2**10, NEns=20, window='hann', OVR=0.5):
    # NOTE: xx and yy must have same dts

    NOV = int(NFFT * OVR)

    Nsp, Nsamp, Ndat = Nspectra_v2(tt, NFFT, NEns, NOV)
    tt = get_intermediate(tt, Ndat)
    xx = get_intermediate(xx, Ndat)
    yy = get_intermediate(yy, Ndat)

    idxs = make_idxs_for_spectrogram_v2(NFFT, NEns, NOV, Nsp)
    tsp = time_for_spectrogram(tt, idxs)
    xens = xx[idxs]
    yens = yy[idxs]
    xens_avg = proc.repeat_and_add_lastdim(np.average(xens, axis=-1), NFFT)
    yens_avg = proc.repeat_and_add_lastdim(np.average(yens, axis=-1), NFFT)
    xens = xens - xens_avg
    yens = yens - yens_avg

    win, enbw, CG, CV = getWindowAndCoefs(NFFT, window, NEns)

    dt = tt[1] - tt[0]
    freq = get_freq_for_spectrum(NFFT, dt)
    fft_x = FFT(xens, win)
    fft_y = FFT(yens, win)

    Sxx_avg, Sxx_std, Sxx_err = get_Sxx(fft_x, CV)
    Syy_avg, Syy_std, Syy_err = get_Sxx(fft_y, CV)

    Sxy_avg, Sxy_std, Sxy_err, Sxy_Re_std, Sxy_Re_err, Sxy_Im_std, Sxy_Im_err = get_Sxy(fft_x, fft_y, CV)

    coh_sq, coh_sq_err = get_coherenceSq_v2(Sxy_avg, Sxy_err, Sxx_avg, Sxx_err, Syy_avg, Syy_err)
    phase, phase_err = get_crossphase_v2(Sxy_avg, Sxy_Re_err, Sxy_Im_err)

    o = struct()
    o.tdat = tt
    o.xdat = xx
    o.ydat = yy
    o.NFFT = NFFT
    o.NEns = NEns
    o.window = window
    o.OVR = OVR
    o.NOV = NOV
    o.win = win
    o.CV = CV
    o.dt = dt
    o.tsp = tsp
    o.dtsp = tsp[1] - tsp[0]
    o.freq = freq
    o.Sxx_avg = Sxx_avg
    o.Sxx_std = Sxx_std
    o.Sxx_err = Sxx_err
    o.Syy_avg = Syy_avg
    o.Syy_std = Syy_std
    o.Syy_err = Syy_err
    o.Sxy_avg = Sxy_avg
    o.Sxy_std = Sxy_std
    o.Sxy_err = Sxy_err
    o.Sxy_Re_std = Sxy_Re_std
    o.Sxy_Re_err = Sxy_Re_err
    o.Sxy_Im_std = Sxy_Im_std
    o.Sxy_Im_err = Sxy_Im_err
    o.coh_sq = coh_sq
    o.coh_sq_err = coh_sq_err
    o.phase = phase
    o.phase_err = phase_err

    return o


def cross_spectre_2s_v2(tt, xx, yy, NFFT=2**10, window="hann", OVR=0.5):

    NOV = int(NFFT * OVR)
    NEns = NEnsFromNSample(NFFT, NOV, len(tt))

    csg = cross_spectrogram_2s(tt, xx, yy, NFFT=NFFT, NEns=NEns, window=window, OVR=OVR)

    o = struct()
    o.tdat = csg.tdat
    o.xdat = csg.xdat
    o.ydat = csg.ydat
    o.NFFT = csg.NFFT
    o.NEns = csg.NEns
    o.window = csg.window
    o.NOV = csg.NOV
    o.win = csg.win
    o.CV = csg.CV
    o.dt = csg.dt
    o.tsp = csg.tsp[0]
    o.freq = csg.freq
    o.Sxx_avg = csg.Sxx_avg[0]
    o.Sxx_std = csg.Sxx_std[0]
    o.Sxx_err = csg.Sxx_err[0]
    o.Syy_avg = csg.Syy_avg[0]
    o.Syy_std = csg.Syy_std[0]
    o.Syy_err = csg.Syy_err[0]
    o.Sxy_avg = csg.Sxy_avg[0]
    o.Sxy_std = csg.Sxy_std[0]
    o.Sxy_err = csg.Sxy_err[0]
    o.Sxy_Re_std = csg.Sxy_Re_std[0]
    o.Sxy_Re_err = csg.Sxy_Re_err[0]
    o.Sxy_Im_std = csg.Sxy_Im_std[0]
    o.Sxy_Im_err = csg.Sxy_Im_err[0]
    o.coh_sq = csg.coh_sq[0]
    o.coh_sq_err = csg.coh_sq_err[0]
    o.phase = csg.phase[0]
    o.phase_err = csg.phase_err[0]

    return o


def ACFfit(x, a, sigma, bck):
    return a*np.exp(-np.log(2)*(x/sigma)**2) + bck


def CCFfit(x, a, delay, sigma, bck):
    return a*np.exp(-np.log(2)*((x-delay)/sigma)**2) + bck


def TauCalc(popt, ccfx, tlags):

    if ccfx > popt[2]:
        afit = ACFfit(tlags, popt[0], popt[1], popt[2])
        xc = np.where(np.diff(np.signbit(afit-ccfx)))
        tau = np.mean(np.abs(tlags[xc]))
    else:
        tau = 0

    return tau



def Nens_from_dtout(dtout, dt, Nfft=2**10, OVR=0.5):

    Nov = int(Nfft * OVR)
    sampledatlen = int(np.round(dtout / dt))
    Nens = NEnsFromNSample(Nfft, Nov, sampledatlen)

    return Nens


def power_spectrogram_2s_by_dtout(tt, xx, dt, dtout=1e-3, Nfft=2**10, OVR=0.5, window="hann"):

    Nens = Nens_from_dtout(dtout=dtout, dt=dt, Nfft=Nfft, OVR=OVR)
    print(f"Nens: {Nens}")
    Nov = int(Nfft * OVR)
    tisp, freq, psd, psd_std, psd_err = power_spectrogram_2s_v4(tt, xx, dt, NFFT=Nfft, window=window, NEns=Nens, NOV=Nov)

    return tisp, freq, psd, psd_std, psd_err


"""
# def matrix_for_1stDerivative_by_5pointsStencil_finiteDiff(Ndat, h):

#     Mstencil = np.zeros((Ndat, Ndat))
#     np.fill_diagonal(Mstencil[1:], -8)
#     np.fill_diagonal(Mstencil[2:], 1)
#     np.fill_diagonal(Mstencil[:, 1:], 8)
#     np.fill_diagonal(Mstencil[:, 2:], -1)

#     Mstencil[0, :] = 0
#     Mstencil[1, :] = 0
#     Mstencil[-1, :] = 0
#     Mstencil[-2, :] = 0

#     Mstencil /= 12 * h

#     return Mstencil


# def function_of_1stDerivative_by_5pointsStencil_finiteDiff(x1d, h):
#     return (1 * x1d[0] - 8 * x1d[1] + 8 * x1d[3] - 1 * x1d[4]) / (12 * h)
"""

def firstDerivative_by_5pointsStencil_finiteDiff(tt, xx):

    dt = tt[1] - tt[0]

    xdot = (np.roll(xx, 2, axis=-1) - 8 * np.roll(xx, 1, axis=-1) + 8 * np.roll(xx, -1, axis=-1) - np.roll(xx, -2, axis=-1)) / (12 * dt)
    xdot = xdot[2: -2]
    t_short = tt[2: -2]

    return t_short, xdot


def lagtime_from_crossphase(freq, crossphase, crossphase_err, Nfit):

    idxs = make_idxs_for_rolling(crossphase.shape[-1], Nfit)
    freq_ext = np.tile(freq, np.append(crossphase.shape[:-1], 1))
    freq_roll = rearange_dat_for_rolling(freq, idxs)
    freq_ext_roll = rearange_dat_for_rolling(freq_ext, idxs)
    freq_avg = np.average(freq_roll, axis=-1)
    crossphase_roll = rearange_dat_for_rolling(crossphase, idxs)
    crossphase_err_roll = rearange_dat_for_rolling(crossphase_err, idxs)

    prms, errs, sigma_y, yHut, yHutErr = polyN_LSM_v2(freq_ext_roll, crossphase_roll, 1, crossphase_err_roll)
    grad = prms[0]
    grad_err = errs[0]

    lagtime = grad / (2*np.pi)
    lagtime_err = grad_err / (2*np.pi)

    return freq_avg, lagtime, lagtime_err


def phasederivative(phase, dt):
    return firstDerivative_by_5pointsStencil_finiteDiff(phase, dt)


def pulsepair(t, iq, Nsample=100, ovr=0.5):

    dt = t[1] - t[0]
    Nov = int(Nsample*ovr)
    Ndat = len(t)
    Nout = NspFromNdat(Nsample, Nov, Ndat)
    idxs = make_idxs_for_spectrogram_wo_EnsembleAVG(Nsample, Nov, Nout)

    tout = time_for_spectrogram_wo_ensembleAVG(t, idxs)
    iqtmp = iq[idxs]
    re = np.real(iqtmp)
    im = np.imag(iqtmp)

    reacf = np.sum(np.delete(re * np.roll(re, -1, axis=-1) + im * np.roll(im, -1, axis=-1), -1, axis=-1), axis=-1)
    imacf = np.sum(np.delete(re * np.roll(im, -1, axis=-1) - np.roll(re, -1, axis=-1) * im, -1, axis=-1), axis=-1)
    iqacf = reacf + 1.j * imacf
    acf0 = np.average(np.real(iqtmp*np.conjugate(iqtmp)), axis=-1)

    fd = 1 / (2*np.pi*dt) * np.angle(iqacf)
    fdstd = 1 / (np.sqrt(2) * np.pi * dt) * np.sqrt(1-(np.abs(iqacf))/acf0)

    o = struct()
    o.t = tout
    o.fd = fd
    o.fdstd = fdstd

    return o


def nonaveragedBiSpectrum(X1, X2, X3):

    X2X1 = X2 * X1
    X3Conj = np.conjugate(X3)
    X2X1X3Conj = X2X1 * X3Conj

    return X2X1X3Conj


def biSpectrum(X1, X2, X3):

    X2X1X3Conj = nonaveragedBiSpectrum(X1, X2, X3)

    BSpec = np.average(X2X1X3Conj, axis=0)
    BSpecStd = np.std(X2X1X3Conj, axis=0, ddof=1)
    BSpecReStd = np.std(np.real(X2X1X3Conj), axis=0, ddof=1)
    BSpecImStd = np.std(np.imag(X2X1X3Conj), axis=0, ddof=1)

    return BSpec, BSpecStd, BSpecReStd, BSpecImStd


def biCoherenceSq(BSpec, BSpecStd, X1, X2, X3, NFFTs, NEns, NOVs):

    proc.suggestNewVer(2, 'biCoherenceSq')

    CV_X2X1, CV_X3 = CVForBiSpecAna(NFFTs, NEns, NOVs)

    X2X1 = X2 * X1
    X2X1AbsSq = np.abs(X2X1) ** 2
    X2X1AbsSqAvg = np.average(X2X1AbsSq, axis=0)
    X2X1AbsSqStd = np.std(X2X1AbsSq, axis=0, ddof=1)
    X2X1AbsSqRer = X2X1AbsSqStd / X2X1AbsSqAvg * CV_X2X1

    X3AbsSq = np.abs(X3) ** 2
    X3AbsSqAvg = np.average(X3AbsSq, axis=0)
    X3AbsSqStd = np.std(X3AbsSq, axis=0, ddof=1)
    X3AbsSqRer = X3AbsSqStd / X3AbsSqAvg * CV_X3

    BSpecAbsSq = np.abs(BSpec) ** 2
    BSpecAbsSqStd = 2 * np.abs(BSpec) * BSpecStd
    BSpecAbsSqRer = BSpecAbsSqStd / BSpecAbsSq

    biCohSq = BSpecAbsSq / (X2X1AbsSqAvg * X3AbsSqAvg)
    biCohSqRer = np.sqrt(BSpecAbsSqRer ** 2 + X2X1AbsSqRer ** 2 + X3AbsSqRer ** 2)
    biCohSqErr = biCohSq * biCohSqRer

    return biCohSq, biCohSqErr


def biCoherenceSq_v2(BSpec, BSpecStd, X1, X2, X3, NEns):

    CV = 1. / NEns

    X2X1 = X2 * X1
    X2X1AbsSq = np.abs(X2X1) ** 2
    X2X1AbsSqAvg = np.average(X2X1AbsSq, axis=0)
    X2X1AbsSqStd = np.std(X2X1AbsSq, axis=0, ddof=1)
    X2X1AbsSqRer = X2X1AbsSqStd / X2X1AbsSqAvg * CV

    X3AbsSq = np.abs(X3) ** 2
    X3AbsSqAvg = np.average(X3AbsSq, axis=0)
    X3AbsSqStd = np.std(X3AbsSq, axis=0, ddof=1)
    X3AbsSqRer = X3AbsSqStd / X3AbsSqAvg * CV

    BSpecAbsSq = np.abs(BSpec) ** 2
    BSpecAbsSqStd = 2 * np.abs(BSpec) * BSpecStd
    BSpecAbsSqRer = BSpecAbsSqStd / BSpecAbsSq * CV

    biCohSq = BSpecAbsSq / (X2X1AbsSqAvg * X3AbsSqAvg)
    biCohSqRer = np.sqrt(BSpecAbsSqRer ** 2 + X2X1AbsSqRer ** 2 + X3AbsSqRer ** 2)
    biCohSqErr = biCohSq * biCohSqRer

    return biCohSq, biCohSqErr


def biPhase(BSpec, BSpecReStd, BSpecImStd):

    proc.suggestNewVer(2, 'biPhase')

    BSpecRe = np.real(BSpec)
    BSpecIm = np.imag(BSpec)

    BSpecReRer = BSpecReStd / BSpecRe
    BSpecImRer = BSpecImStd / BSpecIm

    BSpecImRe = BSpecIm / BSpecRe
    BSpecImReRer = np.sqrt(BSpecImRer ** 2 + BSpecReRer ** 2)
    BSpecImReErr = np.abs(BSpecImRe * BSpecImReRer)
    biPhs = np.arctan2(BSpecIm, BSpecRe)
    biPhsErr = 1. / (1. + BSpecImRe ** 2) * BSpecImReErr

    return biPhs, biPhsErr


def biPhase_v2(BSpec, BSpecReStd, BSpecImStd, NEns):

    CV = 1./np.sqrt(NEns)

    BSpecRe = np.real(BSpec)
    BSpecIm = np.imag(BSpec)

    BSpecReRer = BSpecReStd / BSpecRe * CV
    BSpecImRer = BSpecImStd / BSpecIm * CV

    BSpecImRe = BSpecIm / BSpecRe
    BSpecImReRer = np.sqrt(BSpecImRer ** 2 + BSpecReRer ** 2)
    BSpecImReErr = np.abs(BSpecImRe * BSpecImReRer)
    biPhs = np.arctan2(BSpecIm, BSpecRe)
    biPhsErr = 1. / (1. + BSpecImRe ** 2) * BSpecImReErr

    return biPhs, biPhsErr


def makeIdxsForAutoBiSpectrum(NFFT):

    idxf0 = int(NFFT / 2 + 0.5)
    idxMx1 = np.tile(np.arange(NFFT), (NFFT, 1))
    idxMx2 = idxMx1.T
    coefMx1 = idxMx1 - idxf0
    coefMx2 = idxMx2 - idxf0
    coefMx3 = coefMx1 + coefMx2
    idxMx3 = coefMx3 + idxf0
    idxNan = np.where(((idxMx3 < 0) | (idxMx3 >= NFFT)))
    idxMx3[idxNan] = False

    return idxMx3, idxNan


def makeIdxsForCrossBiSpectrum(NFFTx, NFFTy, NFFTz):

    idxf0x = NFFTx // 2
    idxf0y = NFFTy // 2
    idxf0z = NFFTz // 2

    idxMx1 = np.tile(np.arange(NFFTx), (NFFTy, 1))
    idxMx2 = np.tile(np.arange(NFFTy), (NFFTx, 1)).T
    coefMx1 = idxMx1 - idxf0x
    coefMx2 = idxMx2 - idxf0y
    coefMx3 = coefMx1 + coefMx2
    idxMx3 = coefMx3 + idxf0z
    idxNan = np.where(((idxMx3 < 0) | (idxMx3 >= NFFTz)))
    idxMx3[idxNan] = 0

    del idxMx1
    del idxMx2
    del coefMx1
    del coefMx2
    del coefMx3
    gc.collect()

    return idxMx3, idxNan


def autoBiSpectralAnalysis(freq, XX, NFFT, NEns, NOV):

    idxMx3, idxNan = makeIdxsForAutoBiSpectrum(NFFT)

    X1 = np.reshape(XX, (NEns, 1, NFFT))
    X2 = np.reshape(XX, (NEns, NFFT, 1))
    X3 = XX[:, idxMx3]
    BSpec, BSpecStd, BSpecReStd, BSpecImStd = biSpectrum(X1, X2, X3)

    NFFTs = (NFFT, NFFT, NFFT)

    biCohSq, biCohSqErr = biCoherenceSq(BSpec, BSpecStd, X1, X2, X3, NFFTs, NEns, NOV)
    biPhs, biPhsErr = biPhase(BSpec, BSpecReStd, BSpecImStd)

    # symmetry
    freq1 = np.tile(freq, (NFFT, 1))
    freq2 = freq1.T
    # idxNan2 = np.where((freq2 >= freq1) | (freq2 < - 0.5 * freq1))
    idxNan2 = np.where(freq2 >= freq1)

    biCohSq[idxNan] = np.nan
    biCohSqErr[idxNan] = np.nan
    biPhs[idxNan] = np.nan
    biPhsErr[idxNan] = np.nan
    biCohSq[idxNan2] = np.nan
    biCohSqErr[idxNan2] = np.nan
    biPhs[idxNan2] = np.nan
    biPhsErr[idxNan2] = np.nan

    return biCohSq, biCohSqErr, biPhs, biPhsErr


def crossBiSpecAna(freqs, XX, YY, ZZ, NFFTs, NEns, NOVs):

    proc.suggestNewVer(2, 'crossBiSpecAna')

    freqx, freqy, freqz = freqs

    NFFTx, NFFTy, NFFTz = NFFTs
    NOVx, NOVy, NOVz = NOVs
    idxMxz, idxNan = makeIdxsForCrossBiSpectrum(NFFTs)

    XX = np.reshape(XX, (NEns, 1, NFFTx))
    YY = np.reshape(YY, (NEns, NFFTy, 1))
    ZZ = ZZ[:, idxMxz]

    BSpec, BSpecStd, BSpecReStd, BSpecImStd = biSpectrum(XX, YY, ZZ)

    biCohSq, biCohSqErr = biCoherenceSq_v2(BSpec, BSpecStd, XX, YY, ZZ, NEns)
    biPhs, biPhsErr = biPhase_v2(BSpec, BSpecReStd, BSpecImStd, NEns)

    # symmetry
    freq1 = np.tile(freqx, (NFFTy, 1))
    freq2 = np.tile(freqy, (NFFTx, 1)).T
    # idxNan2 = np.where((freq2 >= freq1) | (freq2 < - 0.5 * freq1))
    idxNan2 = np.where(freq2 >= freq1)

    biCohSq[idxNan] = np.nan
    biCohSqErr[idxNan] = np.nan
    biPhs[idxNan] = np.nan
    biPhsErr[idxNan] = np.nan
    biCohSq[idxNan2] = np.nan
    biCohSqErr[idxNan2] = np.nan
    biPhs[idxNan2] = np.nan
    biPhsErr[idxNan2] = np.nan

    return biCohSq, biCohSqErr, biPhs, biPhsErr


def crossBiSpecAna_v2(freqs, XX0, YY0, ZZ0, NFFTs, NEns, NOVs, iscomplex=False):

    freqx, freqy, freqz = freqs
    NFFTx, NFFTy, NFFTz = NFFTs
    NOVx, NOVy, NOVz = NOVs
    idxMxz, idxNan = makeIdxsForCrossBiSpectrum(NFFTs)

    XX = np.reshape(XX0, (NEns, 1, NFFTx))
    YY = np.reshape(YY0, (NEns, NFFTy, 1))
    ZZ = ZZ0[:, idxMxz]

    BSpec, BSpecStd, BSpecReStd, BSpecImStd = biSpectrum(XX, YY, ZZ)

    biCohSq, biCohSqErr = biCoherenceSq_v2(BSpec, BSpecStd, XX, YY, ZZ, NEns)
    biPhs, biPhsErr = biPhase_v2(BSpec, BSpecReStd, BSpecImStd, NEns)

    biCohSq[idxNan] = np.nan
    biCohSqErr[idxNan] = np.nan
    biPhs[idxNan] = np.nan
    biPhsErr[idxNan] = np.nan

    # symmetry
    freq1 = np.tile(freqx, (NFFTy, 1))
    freq2 = np.tile(freqy, (NFFTx, 1)).T

    if (XX0 == YY0).all():
        if iscomplex:
            idxNan2 = np.where(freq2 > freq1)
        else:
            idxNan2 = np.where((freq2 > freq1) | (freq1 < 0) | (freq2 < - freq1))
    else:
        if iscomplex:
            idxNan2 = []
        else:
            idxNan2 = np.where((freq1 < 0) | (freq2 < 0))

    biCohSq[idxNan2] = np.nan
    biCohSqErr[idxNan2] = np.nan
    biPhs[idxNan2] = np.nan
    biPhsErr[idxNan2] = np.nan


    return biCohSq, biCohSqErr, biPhs, biPhsErr


def cross_bispectrum(freqx, freqy, XX0, YY0, ZZ0, NFFTx, NFFTy, NFFTz, NEns,
                     Fsx, Fsy, Fsz, flimx=None, flimy=None):

    idxMxz, _ = makeIdxsForCrossBiSpectrum(NFFTx, NFFTy, NFFTz)

    freq1 = np.repeat(a=freqx[np.newaxis, :], repeats=NFFTy, axis=0)
    freq2 = np.repeat(a=freqy[:, np.newaxis], repeats=NFFTx, axis=1)

    XX = np.repeat(a=XX0[:, np.newaxis, :], repeats=NFFTy, axis=1)
    YY = np.repeat(a=YY0[:, :, np.newaxis], repeats=NFFTx, axis=2)
    ZZ = ZZ0[:, idxMxz]

    # limitation
    if flimx is not None:
        fidx_x = np.where(np.abs(freqx) < flimx)[0]
        freqx = freqx[fidx_x]
        freq1 = freq1[:, fidx_x]
        freq2 = freq2[:, fidx_x]
        XX = XX[:, :, fidx_x]
        ZZ = ZZ[:, :, fidx_x]
    if flimy is not None:
        fidx_y = np.where(np.abs(freqy) < flimy)[0]
        freqy = freqy[fidx_y]
        freq1 = freq1[fidx_y, :]
        freq2 = freq2[fidx_y, :]
        YY = YY[:, fidx_y, :]
        ZZ = ZZ[:, fidx_y, :]

    # symmetry
    iscomplex = np.iscomplex(XX).any()

    if (XX0 == YY0).all():
        fidx_x = np.where(freqx >= - 0.5 * 0.5 * Fsx)[0]
        fidx_y = np.where(freqy <= 0.5 * 0.5 * Fsy)[0]
        if not iscomplex:
            fidx_x = np.where(freqx >= 0)[0]
            fidx_y = np.where(freqy <= 0.5 * 0.5 * Fsy)[0]
            if (XX0 == ZZ0).all():
                fidx_y = np.where((freqy <= 0.5 * 0.5 * Fsy)&(freqy >= 0.5 * 0.5 * Fsy))[0]

        freqx = freqx[fidx_x]
        freqy = freqy[fidx_y]
        freq1 = freq1[:, fidx_x]
        freq1 = freq1[fidx_y, :]
        freq2 = freq2[:, fidx_x]
        freq2 = freq2[fidx_y, :]
        XX = XX[:, :, fidx_x]
        YY = YY[:, fidx_y, :]
        ZZ = ZZ[:, :, fidx_x]
        ZZ = ZZ[:, fidx_y, :]

    # assign nan value
    if (XX0 == YY0).all():
        idxNan = np.where((np.abs(freq2 + freq1) > Fsz / 2) | (freq2 > freq1))
        if not iscomplex:
            idxNan = np.where((np.abs(freq2 + freq1) > Fsz / 2) | (freq2 > freq1) | (freq2 < - freq1))
            if (XX0 == ZZ0).all():
                idxNan = np.where((np.abs(freq2 + freq1) > Fsz / 2) | (freq2 > freq1)
                                  | (freq2 < - freq1) | (freq2 < - 0.5 * freq1))
    else:
        idxNan = np.where(np.abs(freq2 + freq1) > Fsz / 2)

    BSpec, BSpecStd, BSpecReStd, BSpecImStd = biSpectrum(XX, YY, ZZ)
    biCohSq, biCohSqErr = biCoherenceSq_v2(BSpec, BSpecStd, XX, YY, ZZ, NEns)
    biPhs, biPhsErr = biPhase_v2(BSpec, BSpecReStd, BSpecImStd, NEns)

    biCohSq[idxNan] = np.nan
    biCohSqErr[idxNan] = np.nan
    biPhs[idxNan] = np.nan
    biPhsErr[idxNan] = np.nan

    return freqx, freqy, biCohSq, biCohSqErr, biPhs, biPhsErr


def cross_bispectrum_at_f3(f3_at, freqx, freqy, freqz, XX0, YY0, ZZ0, NFFTx, NFFTy, NFFTz, NEns,
                           Fsx, Fsy, flimx=None, flimy=None):

    idxMxz, _ = makeIdxsForCrossBiSpectrum(NFFTx, NFFTy, NFFTz)
    freq3 = freqz[idxMxz]

    XX = XX0
    YY = YY0
    ZZ = ZZ0[:, idxMxz]


    # limitation
    if flimx is not None:
        fidx_x = np.where(np.abs(freqx) < flimx)[0]
        freqx = freqx[fidx_x]
        freq3 = freq3[:, fidx_x]
        XX = XX[:, fidx_x]
        ZZ = ZZ[:, :, fidx_x]
    if flimy is not None:
        fidx_y = np.where(np.abs(freqy) < flimy)[0]
        freqy = freqy[fidx_y]
        freq3 = freq3[fidx_y, :]
        YY = YY[:, fidx_y]
        ZZ = ZZ[:, fidx_y, :]

    # symmetry
    iscomplex = np.iscomplex(XX).any()

    if (XX0 == YY0).all():
        fidx_x = np.where(freqx >= - 0.5 * 0.5 * Fsx)[0]
        fidx_y = np.where(freqy <= 0.5 * 0.5 * Fsy)[0]
        if not iscomplex:
            fidx_x = np.where(freqx >= 0)[0]
            fidx_y = np.where(freqy <= 0.5 * 0.5 * Fsy)[0]
            if (XX0 == ZZ0).all():
                fidx_y = np.where((freqy <= 0.5 * 0.5 * Fsy)&(freqy >= 0.5 * 0.5 * Fsy))[0]

        freqx = freqx[fidx_x]
        freqy = freqy[fidx_y]
        freq3 = freq3[:, fidx_x]
        freq3 = freq3[fidx_y, :]
        XX = XX[:, fidx_x]
        YY = YY[:, fidx_y]
        ZZ = ZZ[:, :, fidx_x]
        ZZ = ZZ[:, fidx_y, :]

    f3_at = freqz[np.argmin(np.abs(freqz - f3_at))]
    idxs_f3_at = np.where(freq3 == f3_at)

    XX = XX[:, idxs_f3_at[1]]
    YY = YY[:, idxs_f3_at[0]]
    ZZ = ZZ[:, idxs_f3_at[0], idxs_f3_at[1]]

    BSpec, BSpecStd, BSpecReStd, BSpecImStd = biSpectrum(XX, YY, ZZ)
    biCohSq, biCohSqErr = biCoherenceSq_v2(BSpec, BSpecStd, XX, YY, ZZ, NEns)
    biPhs, biPhsErr = biPhase_v2(BSpec, BSpecReStd, BSpecImStd, NEns)

    return f3_at, biCohSq, biCohSqErr, biPhs, biPhsErr


def cross_bispectrum_in_f_range(fmin, fmax, freqx, freqy, freqz, XX0, YY0, ZZ0, NFFTx, NFFTy, NFFTz, NEns,
                                Fsx, Fsy, flimx=None, flimy=None):
    # flimx, flimy: int, float as fmax or tuple, list as (fmin, fmax), [fmin, fmax]

    idxMxz, _ = makeIdxsForCrossBiSpectrum(NFFTx, NFFTy, NFFTz)
    freq3 = freqz[idxMxz]

    XX = XX0
    YY = YY0
    ZZ = ZZ0[:, idxMxz]

    # limitation
    if flimx is not None:
        if isinstance(flimx, tuple) or isinstance(flimx, list):
            fidx_x = np.where((np.abs(freqx) < flimx[1]) & (np.abs(freqx) > flimx[0]))[0]
        else:
            fidx_x = np.where(np.abs(freqx) < flimx)[0]
        freqx = freqx[fidx_x]
        freq3 = freq3[:, fidx_x]
        XX = XX[:, fidx_x]
        ZZ = ZZ[:, :, fidx_x]
    if flimy is not None:
        if isinstance(flimy, tuple) or isinstance(flimy, list):
            fidx_y = np.where((np.abs(freqy) < flimy[1]) & (np.abs(freqy) > flimy[0]))[0]
        else:
            fidx_y = np.where(np.abs(freqy) < flimy)[0]
        freqy = freqy[fidx_y]
        freq3 = freq3[fidx_y, :]
        YY = YY[:, fidx_y]
        ZZ = ZZ[:, fidx_y, :]

    # symmetry
    iscomplex = np.iscomplex(XX).any()

    if (XX0 == YY0).all():
        fidx_x = np.where(freqx >= - 0.5 * 0.5 * Fsx)[0]
        fidx_y = np.where(freqy <= 0.5 * 0.5 * Fsy)[0]
        if not iscomplex:
            fidx_x = np.where(freqx >= 0)[0]
            fidx_y = np.where(freqy <= 0.5 * 0.5 * Fsy)[0]
            if (XX0 == ZZ0).all():
                fidx_y = np.where((freqy <= 0.5 * 0.5 * Fsy)&(freqy >= 0.5 * 0.5 * Fsy))[0]

        freqx = freqx[fidx_x]
        freqy = freqy[fidx_y]
        freq3 = freq3[:, fidx_x]
        freq3 = freq3[fidx_y, :]
        XX = XX[:, fidx_x]
        YY = YY[:, fidx_y]
        ZZ = ZZ[:, :, fidx_x]
        ZZ = ZZ[:, fidx_y, :]

    idxs_f3_at = np.where((np.abs(freq3) > fmin) & (np.abs(freq3) < fmax))

    XX = XX[:, idxs_f3_at[1]]
    YY = YY[:, idxs_f3_at[0]]
    ZZ = ZZ[:, idxs_f3_at[0], idxs_f3_at[1]]

    BSpec, BSpecStd, BSpecReStd, BSpecImStd = biSpectrum(XX, YY, ZZ)
    biCohSq, biCohSqErr = biCoherenceSq_v2(BSpec, BSpecStd, XX, YY, ZZ, NEns)
    biPhs, biPhsErr = biPhase_v2(BSpec, BSpecReStd, BSpecImStd, NEns)

    return biCohSq, biCohSqErr, biPhs, biPhsErr


def cross_bispectral_analysis(xx, yy, zz, dtx, dty, dtz,
                              NFFTx, NFFTy, NFFTz, flimx=None, flimy=None,
                              OVR=0.5, window="hann"):

    o = struct()

    o.NFFTx = NFFTx
    o.NFFTy = NFFTy
    o.NFFTz = NFFTz
    o.OVR = OVR
    o.window = window

    o.NOVx = int(NFFTx * OVR)
    o.NOVy = int(NFFTy * OVR)
    o.NOVz = int(NFFTz * OVR)

    o.Tx = NFFTx * dtx  # Analysis time
    o.Ty = NFFTy * dty  # Analysis time
    o.Tz = NFFTz * dtz  # Analysis time
    if o.Tx != o.Ty or o.Tx != o.Tz or o.Tz != o.Tx:
        filename, lineno = proc.get_current_file_and_line()
        print(f"file: {filename}, line: {lineno}")
        print('Frequency bin widths are different. \n')
        exit()

    # Bi-Spectral Analysis
    xidxs = sliding_window_view(np.arange(xx.size), NFFTx)[::NFFTx - o.NOVx]
    o.NEns = xidxs.shape[-2]
    o.xens = xx[xidxs]
    o.xens = o.xens - o.xens.mean(axis=-1, keepdims=True)
    o.winx, o.enbwx, o.CGx, o.CVx = getWindowAndCoefs(NFFTx, window, o.NEns)
    o.freqx, o.XX = fourier_components_2s(o.xens, dtx, NFFTx, o.winx)

    yidxs = sliding_window_view(np.arange(yy.size), NFFTy)[::NFFTy - o.NOVy]
    o.yens = yy[yidxs]
    o.yens = o.yens - o.yens.mean(axis=-1, keepdims=True)
    o.winy, o.enbwy, o.CGy, o.CVy = getWindowAndCoefs(NFFTy, window, o.NEns)
    o.freqy, o.YY = fourier_components_2s(o.yens, dty, NFFTy, o.winy)

    zidxs = sliding_window_view(np.arange(zz.size), NFFTz)[::NFFTz - o.NOVz]
    o.zens = zz[zidxs]
    o.zens = o.zens - o.zens.mean(axis=-1, keepdims=True)
    o.winz, o.enbwz, o.CGz, o.CVz = getWindowAndCoefs(NFFTz, window, o.NEns)
    _, o.ZZ = fourier_components_2s(o.zens, dtz, NFFTz, o.winz)

    o.freqx = o.freqx.astype(np.float32)
    o.freqy = o.freqy.astype(np.float32)
    o.XX = o.XX.astype(np.complex64)
    o.YY = o.YY.astype(np.complex64)
    o.ZZ = o.ZZ.astype(np.complex64)
    o.freqx, o.freqy, o.biCohSq, o.biCohSqErr, o.biPhs, o.biPhsErr \
        = cross_bispectrum(o.freqx, o.freqy, o.XX, o.YY, o.ZZ,
                           NFFTx, NFFTy, NFFTz, o.NEns,
                           1./dtx, 1./dty, 1./dtz, flimx=flimx, flimy=flimy)
    o.biCohSqRer = o.biCohSqErr / o.biCohSq

    return o


def cross_bispectral_analysis_at_f3(f3_at, xx, yy, zz, dtx, dty, dtz,
                                    NFFTx, NFFTy, NFFTz, flimx=None, flimy=None,
                                    OVR=0.5, window="hann"):

    o = struct()

    o.f3_at = f3_at

    o.NFFTx = NFFTx
    o.NFFTy = NFFTy
    o.NFFTz = NFFTz
    o.OVR = OVR
    o.window = window

    o.NOVx = int(NFFTx * OVR)
    o.NOVy = int(NFFTy * OVR)
    o.NOVz = int(NFFTz * OVR)

    o.Tx = NFFTx * dtx  # Analysis time
    o.Ty = NFFTy * dty  # Analysis time
    o.Tz = NFFTz * dtz  # Analysis time
    if o.Tx != o.Ty or o.Tx != o.Tz or o.Tz != o.Tx:
        filename, lineno = proc.get_current_file_and_line()
        print(f"file: {filename}, line: {lineno}")
        print('Frequency bin widths are different. \n')
        exit()

    # Bi-Spectral Analysis
    xidxs = sliding_window_view(np.arange(xx.size), NFFTx)[::NFFTx - o.NOVx]
    o.NEns = xidxs.shape[-2]
    o.xens = xx[xidxs]
    o.xens = o.xens - o.xens.mean(axis=-1, keepdims=True)
    o.winx, o.enbwx, o.CGx, o.CVx = getWindowAndCoefs(NFFTx, window, o.NEns)
    o.freqx, o.XX = fourier_components_2s(o.xens, dtx, NFFTx, o.winx)

    yidxs = sliding_window_view(np.arange(yy.size), NFFTy)[::NFFTy - o.NOVy]
    o.yens = yy[yidxs]
    o.yens = o.yens - o.yens.mean(axis=-1, keepdims=True)
    o.winy, o.enbwy, o.CGy, o.CVy = getWindowAndCoefs(NFFTy, window, o.NEns)
    o.freqy, o.YY = fourier_components_2s(o.yens, dty, NFFTy, o.winy)

    zidxs = sliding_window_view(np.arange(zz.size), NFFTz)[::NFFTz - o.NOVz]
    o.zens = zz[zidxs]
    o.zens = o.zens - o.zens.mean(axis=-1, keepdims=True)
    o.winz, o.enbwz, o.CGz, o.CVz = getWindowAndCoefs(NFFTz, window, o.NEns)
    o.freqz, o.ZZ = fourier_components_2s(o.zens, dtz, NFFTz, o.winz)

    o.freqx = o.freqx.astype(np.float32)
    o.freqy = o.freqy.astype(np.float32)
    o.freqz = o.freqz.astype(np.float32)
    o.XX = o.XX.astype(np.complex64)
    o.YY = o.YY.astype(np.complex64)
    o.ZZ = o.ZZ.astype(np.complex64)
    o.f3_at, o.biCohSq, o.biCohSqErr, o.biPhs, o.biPhsErr \
        = cross_bispectrum_at_f3(f3_at, o.freqx, o.freqy, o.freqz, o.XX, o.YY, o.ZZ,
                                 NFFTx, NFFTy, NFFTz, o.NEns,
                                 1./dtx, 1./dty, flimx=flimx, flimy=flimy)
    o.biCohSq_total = np.sum(o.biCohSq)
    o.biCohSq_stastd = o.biCohSq.size / o.NEns
    o.biPhs_avg = np.average(o.biPhs)
    o.biPhs_avg_err = np.sqrt(np.average(o.biPhsErr**2))

    return o


def cross_bispectral_analysis_in_f_range(fmin, fmax, xx, yy, zz, dtx, dty, dtz,
                                         NFFTx, NFFTy, NFFTz, flimx=None, flimy=None,
                                         OVR=0.5, window="hann", coef_OV=1.0):

    o = struct()

    o.fmin = fmin
    o.fmax = fmax

    o.NFFTx = NFFTx
    o.NFFTy = NFFTy
    o.NFFTz = NFFTz
    o.OVR = OVR
    o.window = window

    o.NOVx = int(NFFTx * OVR)
    o.NOVy = int(NFFTy * OVR)
    o.NOVz = int(NFFTz * OVR)

    o.Tx = NFFTx * dtx  # Analysis time
    o.Ty = NFFTy * dty  # Analysis time
    o.Tz = NFFTz * dtz  # Analysis time
    if o.Tx != o.Ty or o.Tx != o.Tz or o.Tz != o.Tx:
        filename, lineno = proc.get_current_file_and_line()
        print(f"file: {filename}, line: {lineno}")
        print('Frequency bin widths are different. \n')
        exit()

    # Bi-Spectral Analysis
    xidxs = sliding_window_view(np.arange(xx.size), NFFTx)[::NFFTx - o.NOVx]
    o.NEns = xidxs.shape[-2]
    o.xens = xx[xidxs]
    o.xens = o.xens - o.xens.mean(axis=-1, keepdims=True)
    o.winx, o.enbwx, o.CGx, o.CVx = getWindowAndCoefs(NFFTx, window, o.NEns)
    o.freqx, o.XX = fourier_components_2s(o.xens, dtx, NFFTx, o.winx)

    yidxs = sliding_window_view(np.arange(yy.size), NFFTy)[::NFFTy - o.NOVy]
    o.yens = yy[yidxs]
    o.yens = o.yens - o.yens.mean(axis=-1, keepdims=True)
    o.winy, o.enbwy, o.CGy, o.CVy = getWindowAndCoefs(NFFTy, window, o.NEns)
    o.freqy, o.YY = fourier_components_2s(o.yens, dty, NFFTy, o.winy)

    zidxs = sliding_window_view(np.arange(zz.size), NFFTz)[::NFFTz - o.NOVz]
    o.zens = zz[zidxs]
    o.zens = o.zens - o.zens.mean(axis=-1, keepdims=True)
    o.winz, o.enbwz, o.CGz, o.CVz = getWindowAndCoefs(NFFTz, window, o.NEns)
    o.freqz, o.ZZ = fourier_components_2s(o.zens, dtz, NFFTz, o.winz)

    o.freqx = o.freqx.astype(np.float32)
    o.freqy = o.freqy.astype(np.float32)
    o.freqz = o.freqz.astype(np.float32)
    o.XX = o.XX.astype(np.complex64)
    o.YY = o.YY.astype(np.complex64)
    o.ZZ = o.ZZ.astype(np.complex64)
    o.biCohSq, o.biCohSqErr, o.biPhs, o.biPhsErr \
        = cross_bispectrum_in_f_range(fmin, fmax, o.freqx, o.freqy, o.freqz, o.XX, o.YY, o.ZZ,
                                      NFFTx, NFFTy, NFFTz, o.NEns,
                                      1./dtx, 1./dty, flimx=flimx, flimy=flimy)
    o.biCohSq_total = np.sum(o.biCohSq)
    o.biCohSq_stastd = o.biCohSq.size / o.NEns * coef_OV
    o.biPhs_avg = np.average(o.biPhs)
    o.biPhs_avg_err = np.sqrt(np.average(o.biPhsErr**2))

    return o


def auto_bispectral_analysis(xx, dt, NFFT, OVR=0.5, window="hann", NEns=20):  # not completed

    o = struct()

    o.xx = xx
    o.dt = dt
    o.NFFT = NFFT
    o.OVR = OVR
    o.window = window
    o.NEns = NEns

    o.NOV = int(NFFT * OVR)

    o.T = NFFT * dt  # Analysis time

    # Bi-Spectral Analysis
    o.xens = np.lib.stride_tricks.sliding_window_view(xx, window_shape=NFFT)[::NFFT - o.NOV]
    o.xens = o.xens - o.xens.mean(axis=-1, keepdims=True)
    o.win, o.enbw, o.CG, o.CV = getWindowAndCoefs(NFFT, window, NEns)
    o.freq, o.XX = fourier_components_2s(o.xens, dt, NFFT, o.win)

    o.biCohSq, o.biCohSqErr, o.biPhs, o.biPhsErr = cross_bispectrum(o.freq, o.freq, o.XX, o.XX, o.XX,
                                                                    NFFT, NFFT, NFFT, NEns,
                                                                    iscomplex=np.iscomplex(xx).any())
    o.biCohSqRer = o.biCohSqErr / o.biCohSq

    return o


def autoBiSpecAna_v2(freq, X0, NFFT, NEns, NOV, iscomplex=False):

    idxMxz, idxNan = makeIdxsForCrossBiSpectrum((NFFT, NFFT, NFFT))

    XX = np.reshape(X0, (NEns, 1, NFFT))
    YY = np.reshape(X0, (NEns, NFFT, 1))
    ZZ = X0[:, idxMxz]

    BSpec, BSpecStd, BSpecReStd, BSpecImStd = biSpectrum(XX, YY, ZZ)

    biCohSq, biCohSqErr = biCoherenceSq_v2(BSpec, BSpecStd, XX, YY, ZZ, NEns)
    biPhs, biPhsErr = biPhase_v2(BSpec, BSpecReStd, BSpecImStd, NEns)

    biCohSq[idxNan] = np.nan
    biCohSqErr[idxNan] = np.nan
    biPhs[idxNan] = np.nan
    biPhsErr[idxNan] = np.nan

    # symmetry
    freq1 = np.tile(freq, (NFFT, 1))
    freq2 = np.tile(freq, (NFFT, 1)).T
    if iscomplex:
        idxNan2 = np.where(freq2 > freq1)
    else:
        idxNan2 = np.where((freq2 > freq1) | (freq1 < 0) | (freq2 < - freq1))

    biCohSq[idxNan2] = np.nan
    biCohSqErr[idxNan2] = np.nan
    biPhs[idxNan2] = np.nan
    biPhsErr[idxNan2] = np.nan

    return biCohSq, biCohSqErr, biPhs, biPhsErr


def average_bicoherence_at_f3(freq1, freq2, bicoherence):
    N1 = len(freq1)
    N2 = len(freq2)
    dfreq = freq1[1] - freq1[0]
    idxs = np.arange((N1 + N2) - 1) - (N1 - 1)
    freq3 = dfreq * idxs
    bicoh_f3 = np.array([np.nanmean(np.diagonal(bicoherence, offset=i)) for i in idxs])
    return freq3, bicoh_f3


def total_bicoherence(freq1, freq2, bicoherence):
    N1 = len(freq1)
    N2 = len(freq2)
    dfreq = freq1[1] - freq1[0]
    idxs = np.arange((N1 + N2) - 1) - (N1 - 1)
    freq3 = dfreq * idxs
    bicoh_f3 = np.array([np.nansum(np.diagonal(bicoherence, offset=i)) for i in idxs])
    countarray = bicoherence
    countarray[~np.isnan(countarray)] = 1
    N_components = np.array([np.nansum(np.diagonal(countarray, offset=i)) for i in idxs])
    return freq3, bicoh_f3, N_components


def average_bicoherence_at_f3_withErr(freq1, freq2, bicoherence, bicoherence_Err):
    N1 = len(freq1)
    N2 = len(freq2)
    dfreq = freq1[1] - freq1[0]
    idxs = np.arange((N1 + N2) - 5) - (N1 - 3)
    freq3 = dfreq * idxs
    bicoh_f3 = np.array([np.nanmean(np.diagonal(np.flipud(bicoherence), offset=i)) for i in idxs])
    bicoh_f3_err = np.array([np.sqrt(np.nanvar(np.diagonal(np.flipud(bicoherence), offset=i)) +
                                     np.nanmean(np.diagonal(np.flipud(bicoherence_Err)**2, offset=i))) for i in idxs])
    return freq3, bicoh_f3, bicoh_f3_err


def LSM1(x, y, y_err):

    Nt, Nx = y.shape

    weight = 1./(y_err**2)

    x2 = x**2
    wx02 = np.array([np.ones(weight.shape), x, x2]) * weight
    Swx02 = np.nansum(wx02, axis=-1)
    matrix = np.array([[Swx02[1], Swx02[0]],
                       [Swx02[2], Swx02[1]]])
    matrix = np.transpose(matrix, axes=(2, 0, 1))
    matinv = np.linalg.inv(matrix)

    wx01 = np.array([np.ones(weight.shape), x]) * weight

    wx01 = np.reshape(wx01, (2, Nt, Nx))
    wx01 = np.transpose(wx01, axes=(1, 0, 2))
    Minvwx02 = np.matmul(matinv, wx01)
    y = np.reshape(y, (Nt, Nx, 1))
    y_err = np.reshape(y_err, (Nt, Nx, 1))

    prms = np.matmul(Minvwx02, y)
    errs = np.sqrt(np.matmul(Minvwx02**2, y_err**2))

    prms = np.reshape(prms, (Nt, 2)).T
    errs = np.reshape(errs, (Nt, 2)).T

    return prms, errs


def poly1_LSM(x, y, y_err):

    if x.shape != y.shape:
        print('Improper data shape')
        sys.exit()

    Nfit = x.shape[-1]
    otherNs_array = np.array(x.shape[:-1])
    others_ndim = otherNs_array.size

    weight = 1./(y_err**2)

    x2 = x**2
    wx02 = np.array([np.ones(weight.shape), x, x2]) * weight
    Swx02 = np.nansum(wx02, axis=-1)
    matrix = np.array([[Swx02[1], Swx02[0]],
                       [Swx02[2], Swx02[1]]])
    matrix = np.transpose(matrix, axes=tuple(np.append((np.arange(others_ndim) + 2), [0, 1])))
    matinv = np.linalg.inv(matrix)

    wx01 = np.array([np.ones(weight.shape), x]) * weight

    wx01 = np.transpose(wx01, axes=tuple(np.append((np.arange(others_ndim) + 1), [0, others_ndim + 1])))
    Minvwx02 = np.matmul(matinv, wx01)
    y = np.reshape(y, tuple(np.concatenate([otherNs_array, Nfit, 1], axis=None)))
    y_err = np.reshape(y_err, tuple(np.concatenate([otherNs_array, Nfit, 1], axis=None)))

    prms = np.matmul(Minvwx02, y)
    errs = np.sqrt(np.matmul(Minvwx02**2, y_err**2))

    prms = np.reshape(prms, tuple(np.concatenate([otherNs_array, 2], axis=None)))
    errs = np.reshape(errs, tuple(np.concatenate([otherNs_array, 2], axis=None)))

    prms = np.transpose(prms, axes=tuple(np.concatenate([others_ndim, np.arange(others_ndim)], axis=None)))
    errs = np.transpose(errs, axes=tuple(np.concatenate([others_ndim, np.arange(others_ndim)], axis=None)))

    return prms, errs


def poly2_LSM(x, y, y_err):

    if x.shape != y.shape:
        print('Improper data shape')
        sys.exit()

    Nfit = x.shape[-1]
    otherNs_array = np.array(x.shape[:-1])
    others_ndim = otherNs_array.size

    weight = 1./(y_err**2)
    areNotNansInY = (~np.isnan(y)).astype(np.int8)
    y = np.nan_to_num(y)
    weight = np.nan_to_num(weight)
    y_err = np.nan_to_num(y_err)

    x2 = x**2
    x3 = x**3
    x4 = x**4
    wx04 = np.array([np.ones(weight.shape), x, x2, x3, x4]) * weight
    Swx04 = np.nansum(wx04, axis=-1)
    # Swx04 = np.sum(wx04, axis=-1)
    matrix = np.array([[Swx04[2], Swx04[1], Swx04[0]],
                       [Swx04[3], Swx04[2], Swx04[1]],
                       [Swx04[4], Swx04[3], Swx04[2]]])
    matrix = np.transpose(matrix, axes=tuple(np.append((np.arange(others_ndim) + 2), [0, 1])))
    matinv = np.linalg.inv(matrix)

    wx02 = np.array([np.ones(weight.shape), x, x2]) * weight

    # wx02 = np.reshape(wx02, tuple(np.concatenate([3, otherNs_array, Nfit], axis=None)))
    wx02 = np.transpose(wx02, axes=tuple(np.append((np.arange(others_ndim) + 1), [0, others_ndim + 1])))
    Minvwx02 = np.matmul(matinv, wx02)
    y = np.reshape(y, tuple(np.concatenate([otherNs_array, Nfit, 1], axis=None)))
    y_err = np.reshape(y_err, tuple(np.concatenate([otherNs_array, Nfit, 1], axis=None)))

    prms = np.matmul(Minvwx02, y)
    errs = np.sqrt(np.matmul(Minvwx02**2, y_err**2))

    prms = np.reshape(prms, tuple(np.concatenate([otherNs_array, 3], axis=None)))
    errs = np.reshape(errs, tuple(np.concatenate([otherNs_array, 3], axis=None)))

    prms = np.transpose(prms, axes=tuple(np.concatenate([others_ndim, np.arange(others_ndim)], axis=None)))
    errs = np.transpose(errs, axes=tuple(np.concatenate([others_ndim, np.arange(others_ndim)], axis=None)))

    return prms, errs


def polyN_LSM(x, y, y_err, polyN):

    proc.suggestNewVer(2)

    if x.shape != y.shape:
        print('Improper data shape')
        sys.exit()

    Nfit = x.shape[-1]
    otherNs_array = np.array(x.shape[:-1]).astype(int)
    others_ndim = otherNs_array.size

    weight = 1./(y_err**2)
    areNotNansInY = (~np.isnan(y)).astype(np.int8)
    y = np.nan_to_num(y)
    weight = np.nan_to_num(weight)
    y_err = np.nan_to_num(y_err)

    wx02N = np.array([x**ii for ii in range(polyN * 2 + 1)]) * weight
    Swx02N = np.nansum(wx02N, axis=-1)
    tempArr = np.tile(np.arange(polyN + 1), (polyN + 1, 1))
    matIdxs = np.fliplr(tempArr + np.transpose(tempArr))
    matrix = Swx02N[matIdxs]
    matrix = np.transpose(matrix, axes=tuple(np.append((np.arange(others_ndim) + 2), [0, 1])))
    matinv = np.linalg.inv(matrix)

    wx0N = np.array([x ** ii for ii in range(polyN + 1)]) * weight

    wx0N = np.transpose(wx0N, axes=tuple(np.append((np.arange(others_ndim) + 1), [0, others_ndim + 1])))
    Minvwx0N = np.matmul(matinv, wx0N)
    y = np.reshape(y, tuple(np.concatenate([otherNs_array, Nfit, 1], axis=None)))
    y_err = np.reshape(y_err, tuple(np.concatenate([otherNs_array, Nfit, 1], axis=None)))

    prms = np.matmul(Minvwx0N, y)
    errs = np.sqrt(np.matmul(Minvwx0N**2, y_err**2))

    prms = np.reshape(prms, tuple(np.concatenate([otherNs_array, polyN + 1], axis=None)))
    errs = np.reshape(errs, tuple(np.concatenate([otherNs_array, polyN + 1], axis=None)))

    prms = np.transpose(prms, axes=tuple(np.concatenate([others_ndim, np.arange(others_ndim)], axis=None)))
    errs = np.transpose(errs, axes=tuple(np.concatenate([others_ndim, np.arange(others_ndim)], axis=None)))

    return prms, errs


def polyN_LSM_v2(xx, yy, polyN, yErr=np.array([False])):

    if xx.shape != yy.shape:
        print('Improper data shape')
        sys.exit()

    Nfit = xx.shape[-1]
    otherNs_array = np.array(xx.shape[:-1]).astype(int)
    others_ndim = otherNs_array.size

    xN0 = np.array([xx ** (polyN - ii) for ii in range(polyN + 1)])
    XT = np.transpose(xN0, axes=tuple(np.append((np.arange(others_ndim) + 1), [0, others_ndim + 1])))
    XX = transposeLast2Dims(XT)

    XTX = np.matmul(XT, XX)
    XTXinv = np.linalg.inv(XTX)

    y_vec = turnLastDimToColumnVector(yy)

    if not yErr.all():

        prms_vec = np.matmul(XTXinv, np.matmul(XT, y_vec))
        yHut_vec = np.matmul(XX, prms_vec)
        yHut = turnLastColumnVectorToDim(yHut_vec)
        sigma_y = np.sqrt(np.sum((yy - yHut)**2, axis=-1)/ (Nfit - (polyN + 1)))

    else:

        weight = 1./(yErr**2)

        wX = XX * proc.repeat_and_add_lastdim(weight, polyN + 1)
        wXT = transposeLast2Dims(wX)
        wXTX = np.matmul(XT, wX)
        wXTXinv = np.linalg.inv(wXTX)

        prms_vec = np.matmul(wXTXinv, np.matmul(wXT, y_vec))
        yHut_vec = np.matmul(XX, prms_vec)

        yHut = turnLastColumnVectorToDim(yHut_vec)

        sigma_y = np.sqrt(np.sum(yErr ** 2 + (yy - yHut) ** 2, axis=-1) / Nfit)

    errs = np.sqrt(np.diagonal(XTXinv, axis1=-2, axis2=-1)) * repeat_and_add_lastdim(sigma_y, polyN + 1)
    prms = turnLastColumnVectorToDim(prms_vec)
    yHutErr = np.sqrt(np.diagonal(np.matmul(np.matmul(XX, XTXinv), XT), axis1=-2, axis2=-1)) * repeat_and_add_lastdim(sigma_y, Nfit)

    prms = np.transpose(prms, axes=tuple(np.concatenate([others_ndim, np.arange(others_ndim)], axis=None)))
    errs = np.transpose(errs, axes=tuple(np.concatenate([others_ndim, np.arange(others_ndim)], axis=None)))

    return prms, errs, sigma_y, yHut, yHutErr


"""
# def datMatrix_for_movingLLSM_from_dat1d(dat1d, Nfit):

#     Ndat = len(dat1d)
#     datMat = np.zeros((Ndat, Ndat))
#     for i in np.arange(Nfit):
#         np.fill_diagonal(datMat[i:], dat1d[i:])
#     for i in np.arange(Nfit - 1):
#         datMat[:, -i-1] = 0.

#     return datMat


# def datMatrix_for_movingLLSM_from_dat2d(dat2d, Nfit):

#     ax0size = dat2d.shape[0]
#     ax1size = dat2d.shape[1]
#     datMat = np.zeros(ax0size, ax1size, ax1size))

#     for j in np.arange(ax0size):
#         for i in np.arange(Nfit):
#             np.fill_diagonal(datMat[j, i:], dat2d[j, i:])
#         for i in np.arange(Nfit - 1):
#             datMat[j, :, -i-1] = 0.

#     return datMat


# def datMatrix_for_movingLLSM_from_dat3d(dat3d, Nfit):

#     ax0size = dat3d.shape[0]
#     ax1size = dat3d.shape[1]
#     ax2size = dat3d.shape[2]
#     datMat = np.zeros((ax0size, ax1size, ax2size, ax2size))

#     for k in np.arange(ax0size):
#         for j in np.arange(ax1size):
#             for i in np.arange(Nfit):
#                 np.fill_diagonal(datMat[k, j, i:], dat3d[k, j, i:])
#             for i in np.arange(Nfit - 1):
#                 datMat[k, j, :, -i-1] = 0.

#     return datMat


# def get_unitmatrix_for_movingLLSM(Ndat, Nfit):

#     unitMat = np.zeros((Ndat, Ndat))

#     for i in np.arange(Nfit):
#         np.fill_diagonal(unitMat[i:], 1)
#     for i in np.arange(Nfit - 1):
#         unitMat[:, -i-1] = 0.

#     return unitMat


# def datMatrix_for_movingLLSM(dat, Nfit):

#     Ndat = dat.shape[-1]
#     dat_ext = repeat_and_add_lastdim(dat, Ndat)
#     unitMat = get_unitmatrix_for_movingLLSM(Ndat, Nfit)

#     datMat = unitMat * dat_ext

#     return datMat
"""


def get_Xmatrix_forLLSM(xx, deg):

    tmp = np.array([xx ** i for i in np.flip(np.arange(deg + 1))])
    Xmat = np.transpose(tmp, np.append(np.arange(len(xx.shape)) + 1, 0))

    return Xmat


def get_Xpsinv(WW, XX):

    WX = np.matmul(WW, XX)
    WXt = transposeLast2Dims(WX)
    Xt = transposeLast2Dims(XX)
    XtWX = np.matmul(Xt, WX)
    XtWXinv = np.linalg.inv(XtWX)
    Xpsinv = np.matmul(XtWXinv, WXt)

    return Xpsinv


"""
# def NthPolyfit_by_movingLLSM(xx, yy, Nfit, deg, y_err=np.array([False])):

    # Ndat = xx.shape[-1]

    # if y_err.any():
    #     weights = 1./(y_err**2)
    #     WW = np.matmul(np.identity(Ndat), turnLastDimToColumnVector(weights))
    # else:
    #     WW = np.identity(Ndat)

    # xmat = datMatrix_for_movingLLSM(xx, Nfit)
    # xnew = np.sum(xmat, axis=-2) / Nfit
    # ymat = datMatrix_for_movingLLSM(yy, Nfit)

    # XX = get_Xmatrix_forLLSM(xx, deg)
    # WX = np.matmul(WW, XX)
    # WXt = transposeLast2Dims(WX)
    # WXty = np.matmul(WXt, ymat)

    # Xt = transposeLast2Dims(XX)
    # XtWX = np.matmul(Xt, WX)
    # XtWXinv = np.linalg.inv(XtWX)

    # Xpsinv = np.matmul(XtWXinv, WXt)
    # Xpsinvt = transposeLast2Dims(Xpsinv)
    # XpsinvXpsinvt = np.matmul(Xpsinv, Xpsinvt)

    # theta_mat = np.matmul(XtWXinv, WXty)
    # tmp = np.matmul(XX, theta_mat)
    # yHut = np.where(ymat == 0., 0., tmp)

    # if y_err.any():
    #     y_err_mat = datMatrix_for_movingLLSM(y_err, Nfit)
    #     var_fit = (np.sum(y_err_mat**2, axis=-2) + np.sum((ymat - yHut)**2, axis=-2)) / Nfit
    # else:
    #     var_fit = np.sum((ymat - yHut)**2, axis=-2) / (Nfit - 1)
    # var_fit_mat = repeat_and_add_lastdim(repeat_and_add_lastdim(var_fit, deg), deg)
    # theta_std_mat = np.sqrt(np.diagonal(XpsinvXpsinvt * var_fit_mat, axis1=-2, axis2=-1))

    # return xnew, theta_mat, theta_std_mat, var_fit
"""

def transposeLast2Dims(ndarray):

    N_otherDim = np.array(ndarray.shape[:-2]).astype(int).size
    ndarrayT = np.transpose(ndarray, axes=tuple(np.append(np.arange(N_otherDim), [N_otherDim + 1, N_otherDim])))

    return ndarrayT


def turnLastDimToColumnVector(array):

    temp = list(array.shape)
    temp.append(1)
    newShape = tuple(temp)

    return np.reshape(array, newShape)


def turnLastColumnVectorToDim(array):

    temp = list(array.shape)
    temp = temp[:-1]
    newShape = tuple(temp)

    return np.reshape(array, newShape)


def gauss_LS(x, y, y_err):

    proc.suggestNewVer(2, 'gauss_LS')

    Nt, Nx = y.shape

    print('Y and weight\n')
    Y = np.log(y)
    Y_err = y_err / y
    weight = 1./(Y_err**2)

    del y, y_err
    gc.collect()

    print('Weighted array\n')
    x2 = x**2
    x3 = x**3
    x4 = x**4
    wx04 = np.array([np.ones(weight.shape), x, x2, x3, x4]) * weight
    Swx04 = np.sum(wx04, axis=-1)
    matrix = np.array([[Swx04[2], Swx04[1], Swx04[0]],
                       [Swx04[3], Swx04[2], Swx04[1]],
                       [Swx04[4], Swx04[3], Swx04[2]]])
    matrix = np.transpose(matrix, axes=(2, 0, 1))
    matinv = np.linalg.inv(matrix)

    del wx04, Swx04, x3, x4, matrix
    gc.collect()

    wx02 = np.array([np.ones(weight.shape), x, x2]) * weight

    wx02 = np.reshape(wx02, (3, Nt, Nx))
    wx02 = np.transpose(wx02, axes=(1, 0, 2))
    Minvwx02 = np.matmul(matinv, wx02)
    Y = np.reshape(Y, (Nt, Nx, 1))
    Y_err = np.reshape(Y_err, (Nt, Nx, 1))

    del weight, x, x2, matinv
    gc.collect()

    Lnprm_fit = np.matmul(Minvwx02, Y)
    Lnprm_err = np.sqrt(np.matmul(Minvwx02**2, Y_err**2))

    del Minvwx02, Y, Y_err
    gc.collect()

    print('Fitting parameters\n')
    alpha, beta, gamma = Lnprm_fit[:, :, 0].T
    alpha_err, beta_err, gamma_err = Lnprm_err[:, :, 0].T
    tmp = 0.5 * beta / alpha
    b_fit = np.sqrt(-1./alpha)
    A_fit = gamma - tmp**2 * alpha
    x0_fit = - tmp
    a_fit = np.exp(A_fit)
    A_err = np.sqrt(gamma_err**2 + (tmp * beta_err)**2 + (tmp**2 * alpha_err)**2)
    a_err = a_fit*A_err
    b_err = np.abs(0.5 * b_fit**1.5 * alpha_err)
    x0_err = np.abs(tmp) * np.sqrt((beta_err / beta)**2 + (alpha_err / alpha)**2)

    popt = np.array([a_fit, b_fit, x0_fit])
    perr = np.array([a_err, b_err, x0_err])

    return popt, perr


def gauss_LS_v2(x, y, y_err):

    Y = np.log(y)
    Y_err = y_err / np.abs(y)

    prms, errs, sigma_Y, YHut, YHutErr = polyN_LSM_v2(x, Y, 2, Y_err)

    alpha, beta, gamma = prms
    alpha_err, beta_err, gamma_err = errs
    tmp = 0.5 * beta / alpha
    b_fit = np.sqrt(-1. / alpha)
    A_fit = gamma - tmp ** 2 * alpha
    x0_fit = - tmp
    a_fit = np.exp(A_fit)
    A_err = np.sqrt(gamma_err ** 2 + (tmp * beta_err) ** 2 + (tmp ** 2 * alpha_err) ** 2)
    a_err = a_fit * A_err
    b_err = np.abs(0.5 * b_fit ** 1.5 * alpha_err)
    x0_err = np.abs(tmp) * np.sqrt((beta_err / beta) ** 2 + (alpha_err / alpha) ** 2)

    popt = np.array([a_fit, b_fit, x0_fit])
    perr = np.array([a_err, b_err, x0_err])

    y_hut = np.exp(YHut)
    y_hut_err = y_hut * YHutErr

    return popt, perr, y_hut, y_hut_err


def gradient(R, reff, rho, dat, err, Nfit):

    Nt, NR = reff.shape
    NRf = NR - Nfit + 1

    idxs_calc = np.full((NRf, Nfit), np.arange(Nfit)).T  # (Nfit, NRf)
    idxs_calc = idxs_calc + np.arange(NRf)  # (Nfit, NRf)
    idxs_calc = idxs_T  # (NRf, Nfit)
    Rcal = R[idxs_calc]
    reffcal = reff[:, idxs_calc]
    rhocal = rho[:, idxs_calc]
    datcal = dat[:, idxs_calc]
    errcal = err[:, idxs_calc]

    popt, perr = LSM1_2d(reffcal, datcal, errcal)

    dat_grad = popt[0]
    err_grad = perr[0]

    R_f = np.average(Rcal, axis=-1)  # (NRf,)
    reff_f = np.average(reffcal, axis=-1)  # (Nt, NRf)
    rho_f = np.average(rhocal, axis=-1)  # (Nt, NRf)

    return R_f, reff_f, rho_f, dat_grad, err_grad


def make_idxs_for_MovingLSM(data_len, window_len):

    output_len = data_len - window_len + 1
    output_idxs = np.full((output_len, window_len), np.arange(window_len)).T
    output_idxs = output_idxs + np.arange(output_len)
    output_idxs = output_idxs.T

    return output_idxs


def make_idxs_for_rolling(data_len, window_len):

    output_len = data_len - window_len + 1
    output_idxs = np.full((output_len, window_len), np.arange(window_len)).T
    output_idxs = output_idxs + np.arange(output_len)
    output_idxs = output_idxs.T


    return output_idxs


def rearange_dat_for_rolling(xx, idxs):

    if xx.ndim == 1:
        x_roll = xx[idxs]
    elif xx.ndim == 2:
        x_roll = xx[:, idxs]
    elif xx.ndim == 3:
        x_roll = xx[:, :, idxs]
    else:
        print('please fix function rearange_dat_for_rolling')
        sys.exit()

    return x_roll


def rolling_average(xx, yy, Nwin, y_err=np.array([False])):

    idxs = make_idxs_for_rolling(xx.shape[-1], Nwin)
    x_roll = rearange_dat_for_rolling(xx, idxs)
    y_roll = rearange_dat_for_rolling(yy, idxs)

    x_avg = np.average(x_roll, axis=-1)
    if y_err.any():
        y_err_roll = rearange_dat_for_rolling(y_err, idxs)
        weight_roll = 1./y_err_roll**2
        y_avg = np.average(y_roll, axis=-1, weights=weight_roll)
        y_std = np.sqrt(np.average(y_err_roll**2, axis=-1) + np.var(y_roll, axis=-1))
        sum_weight = np.sum(weight_roll, axis=-1)
        norm_weight = weight_roll / repeat_and_add_lastdim(sum_weight, Nwin)
        y_err = np.sqrt(np.sum(norm_weight**2 * (y_err_roll**2 + (y_roll - repeat_and_add_lastdim(y_avg, Nwin))**2), axis=-1))
    else:
        y_avg = np.average(y_roll, axis=-1)
        y_std = np.std(y_roll, axis=-1, ddof=1)
        y_err = y_std / np.sqrt(Nwin)

    return x_avg, y_avg, y_std, y_err


def gradient_by_roll_avg(xx, yy, Nwin, y_err=np.array([False])):

    if y_err.any():
        x_avg, y_avg, y_std, y_err = rolling_average(xx, yy, Nwin, y_err)
    else:
        x_avg, y_avg, y_std, y_err = rolling_average(xx, yy, Nwin)

    dx = x_avg[1] - x_avg[0]
    grad_y, grad_y_err = firstDerivative_by_5pointsStencil_finiteDiff(y_avg, y_err, dx)

    return x_avg, y_avg, y_std, y_err, grad_y, grad_y_err


def gradient_reg_reff(reff, dat, err, Nfit):

    Nt, NR = reff.shape
    idxs_calc = make_idxs_for_MovingLSM(NR, Nfit)
    # Rcal = R[idxs_calc]
    reffcal = reff[:, idxs_calc]
    # rhocal = rho[:, idxs_calc]
    datcal = dat[:, idxs_calc]
    errcal = err[:, idxs_calc]

    popt, perr = LSM1_2d(reffcal, datcal, errcal)

    reff_f = np.average(reffcal, axis=-1)  # (Nt, NRf)
    dat_reg = popt[0] * reff_f + popt[1]
    popt_cal = np.repeat(popt.reshape((2, Nt, NRf, 1)), Nfit, axis=-1)
    dat_reg_cal = popt_cal[0] * reffcal + popt_cal[1]
    err_reg = np.sqrt(np.sum((datcal - dat_reg_cal) ** 2, axis=-1) / (Nfit - 2))
    # err_reg = np.sqrt((perr[0] * reff_f)**2 + perr[1]**2)
    S = np.sqrt(np.sum((reffcal - np.repeat(np.reshape(reff_f, (Nt, NRf, 1)), Nfit, axis=-1)) ** 2, axis=-1) / Nfit)
    perr[0] = err_reg / S / np.sqrt(Nfit)
    perr[1] = err_reg / Nfit / S * np.sqrt(np.sum(reffcal ** 2, axis=-1))

    dat_grad = popt[0]
    err_grad = perr[0]

    return reff_f, dat_grad, err_grad, dat_reg, err_reg


def repeat_and_add_lastdim(Array, Nrepeat):
    tmp = tuple(np.concatenate([np.array(Array.shape), 1], axis=None).astype(int))
    return np.repeat(np.reshape(Array, tmp), Nrepeat, axis=-1)


def make_fitted_profiles_with_MovingPolyLSM(reff, raw_profiles, profiles_errs, window_len, poly=2):

    print(proc.suggestNewVer(2, 'make_fitted_profiles_with_MovingPolyLSM'))

    if reff.shape != raw_profiles.shape:
        print('Improper data shape')
        sys.exit()
    idxs_for_Moving = make_idxs_for_MovingLSM(reff.shape[-1], window_len)
    output_profiles_count = idxs_for_Moving.shape[0]

    reff_for_Moving = reff[:, idxs_for_Moving]
    reff_avgs = np.nanmean(reff_for_Moving, axis=-1)
    reff_for_fitting = reff_for_Moving - repeat_and_add_lastdim(reff_avgs, window_len)
    profiles_for_fitting = raw_profiles[:, idxs_for_Moving]
    profiles_errs_for_fitting = profiles_errs[:, idxs_for_Moving]

    if poly == 1:
        print('polynomial 1\n')
        popt, perr = poly1_LSM(reff_for_fitting, profiles_for_fitting, profiles_errs_for_fitting)
        fitted_profs_gradients = popt[0]
        fitted_profiles = popt[1]
        aa = repeat_and_add_lastdim(popt[0], window_len)
        bb = repeat_and_add_lastdim(popt[1], window_len)
        # fitted_profiles_wo_average = aa * reff_for_fitting + bb
        # fitted_profiles_errs = np.sqrt(np.sum(profiles_errs_for_fitting**2 + (profiles_for_fitting - fitted_profiles_wo_average)**2, axis=-1)/(window_len - 2))
        S = np.sqrt(np.sum((reff_for_fitting - repeat_and_add_lastdim(reff_avgs, window_len)) ** 2, axis=-1) / window_len)
        fitted_profs_grads_errs = perr[0]
        fitted_profiles_errs = perr[1]

    elif poly == 2:
        print('polynomial 2\n')
        popt, perr = poly2_LSM(reff_for_fitting, profiles_for_fitting, profiles_errs_for_fitting)
        fitted_profs_gradients = popt[1]
        fitted_profiles = popt[2]

        aa = repeat_and_add_lastdim(popt[0], window_len)
        bb = repeat_and_add_lastdim(popt[1], window_len)
        cc = repeat_and_add_lastdim(popt[2], window_len)
        # fitted_profiles_wo_average = aa * reff_for_fitting**2 + bb * reff_for_fitting + cc
        # fitted_profiles_errs = np.sqrt(np.sum(profiles_errs_for_fitting**2 + (profiles_for_fitting - fitted_profiles_wo_average)**2, axis=-1)/(window_len - 2))
        fitted_profs_grads_errs = perr[1]
        fitted_profiles_errs = perr[2]

    else:
        print('It has not developed yet...\n')
        sys.exit()

    return reff_avgs, fitted_profiles, fitted_profiles_errs, fitted_profs_gradients, fitted_profs_grads_errs


def make_fitted_profiles_with_MovingPolyLSM_v2(reff, raw_profiles, profiles_errs, window_len, poly=1):

    if reff.shape != raw_profiles.shape:
        print('Improper data shape')
        sys.exit()
    idxs_for_Moving = make_idxs_for_MovingLSM(reff.shape[-1], window_len)
    output_profiles_count = idxs_for_Moving.shape[0]

    reff_for_Moving = reff[:, idxs_for_Moving]
    reff_cent = np.nanmean(reff_for_Moving, axis=-1)
    reff_for_fitting = reff_for_Moving - repeat_and_add_lastdim(reff_cent, window_len)
    profiles_for_fitting = raw_profiles[:, idxs_for_Moving]
    profiles_errs_for_fitting = profiles_errs[:, idxs_for_Moving]

    popt, perr, sigma_y, yHut, yHutErr = \
        polyN_LSM_v2(reff_for_fitting, profiles_for_fitting, poly, profiles_errs_for_fitting)
    fitted_profs_gradients = popt[-2]
    fitted_profiles = popt[-1]
    fitted_profs_grads_errs = perr[-2]
    fitted_profiles_errs = sigma_y

    return reff_cent, fitted_profiles, fitted_profiles_errs, fitted_profs_gradients, fitted_profs_grads_errs


def make_radialAxes_for_MovingPolyLSM(reff, window_len):

    # profiles_count, profile_len = reff.shape
    idxs_for_Moving = make_idxs_for_MovingLSM(reff.shape[-1], window_len)

    reff_for_Moving = reff[:, idxs_for_Moving]
    reff_avgs = np.nanmean(reff_for_Moving, axis=-1)
    reff_for_fitting = reff_for_Moving - repeat_and_add_lastdim(reff_avgs, window_len)

    return reff_avgs


def make_fitted_profiles_with_MovingPolyLSM_1d(reff, raw_profiles, profiles_errs, window_len, poly=2):

    if reff.shape != raw_profiles.shape:
        print('Improper data shape')
        sys.exit()
    # profiles_count, profile_len = reff.shape
    idxs_for_Moving = make_idxs_for_MovingLSM(reff.shape[-1], window_len)
    output_profiles_count = idxs_for_Moving.shape[0]

    reff_for_Moving = reff[idxs_for_Moving]
    reff_avgs = np.nanmean(reff_for_Moving, axis=-1)
    reff_for_fitting = reff_for_Moving - repeat_and_add_lastdim(reff_avgs, window_len)
    profiles_for_fitting = raw_profiles[idxs_for_Moving]
    profiles_errs_for_fitting = profiles_errs[idxs_for_Moving]

    if poly == 1:
        print('polynomial 1\n')
        popt, perr = poly1_LSM(reff_for_fitting, profiles_for_fitting, profiles_errs_for_fitting)
        fitted_profs_gradients = popt[0]
        fitted_profiles = popt[1]
        aa = repeat_and_add_lastdim(popt[0], window_len)
        bb = repeat_and_add_lastdim(popt[1], window_len)
        # fitted_profiles_wo_average = aa * reff_for_fitting + bb
        # fitted_profiles_errs = np.sqrt(np.sum(profiles_errs_for_fitting**2 + (profiles_for_fitting - fitted_profiles_wo_average)**2, axis=-1)/(window_len - 2))
        S = np.sqrt(np.sum((reff_for_fitting - repeat_and_add_lastdim(reff_avgs, window_len)) ** 2, axis=-1) / window_len)
        fitted_profs_grads_errs = perr[0]
        fitted_profiles_errs = perr[1]

    elif poly == 2:
        print('polynomial 2\n')
        popt, perr = poly2_LSM(reff_for_fitting, profiles_for_fitting, profiles_errs_for_fitting)
        fitted_profs_gradients = popt[1]
        fitted_profiles = popt[2]

        aa = repeat_and_add_lastdim(popt[0], window_len)
        bb = repeat_and_add_lastdim(popt[1], window_len)
        cc = repeat_and_add_lastdim(popt[2], window_len)
        # fitted_profiles_wo_average = aa * reff_for_fitting**2 + bb * reff_for_fitting + cc
        # fitted_profiles_errs = np.sqrt(np.sum(profiles_errs_for_fitting**2 + (profiles_for_fitting - fitted_profiles_wo_average)**2, axis=-1)/(window_len - 2))
        fitted_profs_grads_errs = perr[1]
        fitted_profiles_errs = perr[2]

    else:
        print('It has not developed yet...\n')
        sys.exit()

    return reff_avgs, fitted_profiles, fitted_profiles_errs, fitted_profs_gradients, fitted_profs_grads_errs


def linInterp1dOf2dDat(x2d, y1d, x2d_ref):

    Nt, Ny = x2d.shape
    Nt_ref, Ny_ref = x2d_ref.shape
    if Nt != Nt_ref:
        print('Error: Nt != Ntf')
        sys.exit()

    x2d_ext = np.repeat(np.reshape(x2d, (Nt, 1, Ny)), Ny_ref, axis=1)  # (Nt, Ny_ref, Ny)
    x2d_ref_ext = np.repeat(np.reshape(x2d_ref, (Nt, Ny_ref, 1)), Ny, axis=2)  # (Nt, Ny_ref, Ny)

    dx2d = x2d_ref_ext - x2d_ext
    idxs1 = np.nanargmin(np.where(dx2d <= 0, np.nan, dx2d), axis=-1)
    idxs2 = np.nanargmax(np.where(dx2d >= 0, np.nan, dx2d), axis=-1)

    y1 = y1d[idxs1]
    y2 = y1d[idxs2]
    idxs_t = np.tile(np.reshape(np.arange(Nt), (Nt, 1)), (1, Ny_ref))
    x1 = x2d[idxs_t, idxs1]
    x2 = x2d[idxs_t, idxs2]

    y2d_ref = (y2 - y1) / (x2 - x1) * (x2d_ref - x1) + y1

    return y2d_ref


# 2022/3/28 define Tratio as Ti / Te ( from tau = Te / Ti )
# 2022/6/14 redefine Tratio as Te / Ti
def Tratio(Te, Ti, Te_err, Ti_err):

    Tratio = Te / Ti

    Te_rerr = Te_err / Te
    Ti_rerr = Ti_err / Ti
    Tratio_rerr = np.sqrt(Te_rerr ** 2 + Ti_rerr ** 2)
    Tratio_err = Tratio * Tratio_rerr

    return Tratio, Tratio_err


"""
# def weighted_average_1D(x1D, weight1D):
#     Sw = np.sum(weight1D)
#     wx = x1D * weight1D
#     Swx = np.sum(wx)
#     xm = Swx / Sw
#     w2 = weight1D ** 2
#     Sw2 = np.sum(w2)
#     errsq = (x1D - np.full(x1D.T.shape, xm).T) ** 2
#     werrsq = weight1D * errsq
#     Swerrsq = np.sum(werrsq)
#     U = Sw / (Sw ** 2 - Sw2) * Swerrsq
#     xerr = np.sqrt(U)
#
#     wwerrsq = weight1D * werrsq
#     Swwerrsq = np.sum(wwerrsq)
#     Um = Sw / (Sw ** 2 - Sw2) * Swwerrsq / Sw
#     xmerr = np.sqrt(Um)
#
#     return xm, xerr, xmerr
#
#
# def weighted_average_2D(x2D, weight2D):
#
#     areNotNansInX = (~np.isnan(x2D)).astype(np.int8)
#     areNotNansInWgt = (~np.isnan(weight2D)).astype(np.int8)
#     x2D = np.nan_to_num(x2D) * areNotNansInWgt
#     weight2D = np.nan_to_num(weight2D) * areNotNansInX
#
#     Sw = np.sum(weight2D, axis=1)
#     wx = x2D * weight2D
#     Swx = np.sum(wx, axis=1)
#     xm = Swx / Sw
#     w2 = weight2D ** 2
#     Sw2 = np.sum(w2, axis=1)
#     errsq = (x2D - np.full(x2D.T.shape, xm).T) ** 2
#     werrsq = weight2D * errsq
#     Swerrsq = np.sum(werrsq, axis=1)
#     U = Sw / (Sw ** 2 - Sw2) * Swerrsq
#     xerr = np.sqrt(U)
#
#     wwerrsq = weight2D * werrsq
#     Swwerrsq = np.sum(wwerrsq, axis=1)
#     Um = Sw / (Sw ** 2 - Sw2) * Swwerrsq / Sw
#     xmerr = np.sqrt(Um)
#
#     return xm, xerr, xmerr
"""


def envelope(sig):
    analytic_signal = signal.hilbert(sig)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def amplitude(signal):

	if np.iscomplexobj(signal):
		amp = np.abs(signal)
	else:
		amp = envelope(signal)

	return amp


def phase(complex_signal, unwrap=True):
    if unwrap:
        phase = np.unwrap(np.angle(complex_signal)) 
    else:
        phase = np.angle(complex_signal)
    return phase


def turnLastDimToDiagMat(array):
    rawdatshape = array.shape
    last_dim_size = rawdatshape[-1]
    result_array = np.zeros(rawdatshape + (last_dim_size,), dtype=array.dtype)
    for i in range(last_dim_size):
        result_array[..., i, i] = array[..., i]
    return result_array


def EMwaveInPlasma(freqin, ne, B):

    e = 1.602176634e-19
    const.me = 9.1093837e-31
    const.eps0 = 8.85418782e-12

    o = struct()
    o.fin = freqin
    o.ne = ne
    o.B = B

    o.omgp = np.sqrt(ne * e ** 2 / (const.eps0 * const.me))
    o.omgc = e * B / const.me
    o.omguh = np.sqrt(o.omgp ** 2 + o.omgc ** 2)
    o.omgL = 0.5 * (-o.omgc + np.sqrt(o.omgc ** 2 + 4 * o.omgp ** 2))
    o.omgR = 0.5 * (o.omgc + np.sqrt(o.omgc ** 2 + 4 * o.omgp ** 2))

    o.omgin = 2 * np.pi * o.fin

    o.NO = np.sqrt(1 - (o.omgp ** 2) / (o.omgin ** 2))
    o.NX = np.sqrt((o.omgin ** 2 - o.omgL ** 2) *
                   (o.omgin ** 2 - o.omgR ** 2) /
                   (o.omgin ** 2 * (o.omgin ** 2 - o.omguh ** 2)))

    return o


def gradient_reg(R, reff, a99, dat, err, Nfit, poly):

    proc.suggestNewVer(2, 'gradient_reg')

    notNanIdxs = np.argwhere(~np.isnan(dat).all(axis=1)).T[0]
    reff_c = reff[notNanIdxs]
    dat_c = dat[notNanIdxs]
    err_c = err[notNanIdxs]

    Nt, NR = reff.shape
    Ntc, NR = reff_c.shape
    NRf = NR - Nfit + 1

    # reff_f, dat_grad, err_grad, dat_reg, err_reg = gradient_reg_reff(reff, dat, err, Nfit)
    reff_f_c, dat_reg_c, err_reg_c, dat_grad_c, err_grad_c = make_fitted_profiles_with_MovingPolyLSM(reff_c, dat_c, err_c, Nfit, poly=poly)
    reff_f = make_radialAxes_for_MovingPolyLSM(reff, Nfit)

    R_f = linInterp1dOf2dDat(reff, R, reff_f)

    dat_reg = np.full((Nt, NRf), np.nan)
    dat_reg[notNanIdxs] = dat_reg_c
    err_reg = np.full((Nt, NRf), np.nan)
    err_reg[notNanIdxs] = err_reg_c
    dat_grad = np.full((Nt, NRf), np.nan)
    dat_grad[notNanIdxs] = dat_grad_c
    err_grad = np.full((Nt, NRf), np.nan)
    err_grad[notNanIdxs] = err_grad_c

    a99 = np.repeat(np.reshape(a99, (Nt, 1)), NRf, axis=-1)
    rho_f = reff_f / a99

    return R_f, reff_f, rho_f, dat_grad, err_grad, dat_reg, err_reg


def gradient_reg_v2(R, reff, a99, dat, err, Nfit, poly):

    Nt, NR = reff.shape
    NRf = NR - Nfit + 1

    reff_f, dat_reg, err_reg, dat_grad, err_grad = \
        make_fitted_profiles_with_MovingPolyLSM_v2(reff, dat, err, Nfit, poly=poly)

    R_f = linInterp1dOf2dDat(reff, R, reff_f)
    a99 = np.repeat(np.reshape(a99, (Nt, 1)), NRf, axis=-1)
    rho_f = reff_f / a99

    return R_f, reff_f, rho_f, dat_grad, err_grad, dat_reg, err_reg


def gradient_reg_1d(R, reff, a99, dat, err, Nfit, poly):

    # reff_f, dat_grad, err_grad, dat_reg, err_reg = gradient_reg_reff(reff, dat, err, Nfit)
    reff_f, dat_reg, err_reg, dat_grad, err_grad = make_fitted_profiles_with_MovingPolyLSM_1d(reff, dat, err, Nfit, poly=poly)

    # Nt, NR = reff.shape
    NR = reff.shape[-1]
    Ndim = reff.ndim
    NRf = NR - Nfit + 1
    # reff_ext = np.repeat(np.reshape(reff, (Nt, 1, NR)), NRf, axis=1)  # (Nt, NRf, NR)
    # reff_f_ext = np.repeat(np.reshape(reff_f, (Nt, NRf, 1)), NR, axis=2)  # (Nt, NRf, NR)

    reff_ext = repeat_and_add_lastdim(reff, NRf).transpose([1, 0])
    reff_f_ext = repeat_and_add_lastdim(reff_f, NR)

    dreff = reff_f_ext - reff_ext
    idxs1 = np.nanargmin(np.where(dreff <= 0, np.nan, dreff), axis=-1)
    idxs2 = np.nanargmax(np.where(dreff >= 0, np.nan, dreff), axis=-1)

    R1 = R[idxs1]
    R2 = R[idxs2]
    reff1 = reff[idxs1]
    reff2 = reff[idxs2]

    R_f = (R2 - R1) / (reff2 - reff1) * (reff_f - reff1) + R1

    a99_ext = np.repeat(a99, NRf)
    # a99_ext = repeat_and_add_lastdim(a99, NRf)
    rho_f = reff_f / a99_ext

    return R_f, reff_f, rho_f, dat_grad, err_grad, dat_reg, err_reg


def polyN_LSM_der(xx, yy, polyN=10, yErr=None, parity=None):  #parity = None, "even", or "odd"

    if xx.shape == yy.shape:
        mode = "multiple_x"
    elif xx.size == yy.shape[-1]:
        mode = "single_x"
    else:
        print('Improper data shape')
        exit()

    if parity == "even" and polyN % 2 != 0:
        polyN -= 1
    if parity == "odd" and polyN % 2 == 0:
        polyN -= 1
    o = struct()
    o.polyN = polyN

    otherNs_array = np.array(yy.shape[:-1]).astype(int)
    others_ndim = otherNs_array.size

    def makeXX(xx, polyN, parity):

        if parity == "even":
            xN0 = np.array([xx ** (polyN - ii) for ii in range(0, polyN + 1, 2)])
            Nparam = xN0.shape[0]
        elif parity == "odd":
            xN0 = np.array([xx ** (polyN - ii) for ii in range(0, polyN + 1, 2)])
            Nparam = xN0.shape[0]
        elif parity == None:
            xN0 = np.array([xx ** (polyN - ii) for ii in range(polyN + 1)])
            Nparam = xN0.shape[0]
        else:
            print("Unavailable input in parity.")
            exit()

        return xN0, Nparam

    if mode == "multiple_x":
        Nfit = xx.shape[-1]
        xN0, Nparam = makeXX(xx, polyN, parity)
        XT = np.transpose(xN0, axes=tuple(np.append((np.arange(others_ndim) + 1), [0, others_ndim + 1])))
        XX = transposeLast2Dims(XT)
    elif mode == "single_x":
        Nfit = xx.size
        XT, Nparam = makeXX(xx, polyN, parity)
        XX = np.transpose(XT)
    else:
        exit()

    if yErr is None:
        WW = np.identity(Nfit)
    else:
        WW = 1. / yErr ** 2
        WW = turnLastDimToDiagMat(WW)  # !!array shape has changed!!

    WX = np.matmul(WW, XX)
    XTWX = np.matmul(XT, WX)
    XTWXinv = np.linalg.inv(XTWX)
    WXT = transposeLast2Dims(WX)
    Xplus = np.matmul(XTWXinv, WXT)  # pseudo inverse matrix
    XplusT = transposeLast2Dims(Xplus)
    XpXpT = np.matmul(Xplus, XplusT)
    XXp = np.matmul(XX, Xplus)
    XXpT = transposeLast2Dims(XXp)
    XXpXXpT = np.matmul(XXp, XXpT)

    y_vec = turnLastDimToColumnVector(yy)
    popt_vec = np.matmul(Xplus, y_vec)
    yHut_vec = np.matmul(XX, popt_vec)
    yHut = turnLastColumnVectorToDim(yHut_vec)
    if yErr is None:
        sigma_y = np.sqrt(np.sum((yy - yHut) ** 2, axis=-1) / (Nfit - Nparam))
    else:
        sigma_y = np.sqrt(np.sum(yErr ** 2 + (yy - yHut) ** 2, axis=-1) / Nfit)
    perr = np.sqrt(np.diagonal(XpXpT, axis1=-2, axis2=-1)) * repeat_and_add_lastdim(sigma_y, Nparam)
    yHutErr = np.sqrt(np.diagonal(XXpXXpT, axis1=-2, axis2=-1)) * repeat_and_add_lastdim(sigma_y, Nfit)

    if parity is None:
        MM = np.diagflat(np.flip(np.arange(1, polyN + 1, 1)), k=-1)
        XM = np.matmul(XX, MM)
        XMXp = np.matmul(XM, Xplus)
        XMXpT = transposeLast2Dims(XMXp)
        XMXpXMXpT = np.matmul(XMXp, XMXpT)
    elif parity == "even":
        if mode == "multiple_x":
            Nfit = xx.shape[-1]
            xN0, Nparam = makeXX(xx, polyN=polyN + 1, parity="odd")
            XoT = np.transpose(xN0, axes=tuple(np.append((np.arange(others_ndim) + 1), [0, others_ndim + 1])))
            Xo = transposeLast2Dims(XoT)
        elif mode == "single_x":
            Nfit = xx.size
            XoT, Nparam = makeXX(xx, polyN=polyN + 1, parity="odd")
            Xo = np.transpose(XoT)
        else:
            exit()
        MM = np.diagflat(np.flip(np.arange(2, polyN + 1, 2)), k=-1)
        XM = np.matmul(Xo, MM)
        XMXp = np.matmul(XM, Xplus)
        XMXpT = transposeLast2Dims(XMXp)
        XMXpXMXpT = np.matmul(XMXp, XMXpT)
    elif parity == "odd":
        if mode == "multiple_x":
            Nfit = xx.shape[-1]
            xN0, Nparam = makeXX(xx, polyN=polyN + 1, parity="even")
            XoT = np.transpose(xN0, axes=tuple(np.append((np.arange(others_ndim) + 1), [0, others_ndim + 1])))
            Xo = transposeLast2Dims(XoT)
        elif mode == "single_x":
            Nfit = xx.size
            XoT, Nparam = makeXX(xx, polyN=polyN + 1, parity="even")
            Xo = np.transpose(XoT)
        else:
            exit()
        MM = np.diagflat(np.flip(np.arange(1, polyN + 1, 2)), k=-1)
        XM = np.matmul(Xo, MM)
        XMXp = np.matmul(XM, Xplus)
        XMXpT = transposeLast2Dims(XMXp)
        XMXpXMXpT = np.matmul(XMXp, XMXpT)
    else:
        exit()

    yHutDer_vec = np.matmul(XM, popt_vec)
    yHutDer = turnLastColumnVectorToDim(yHutDer_vec)
    yHutDerErr = np.sqrt(np.diagonal(XMXpXMXpT, axis1=-2, axis2=-1)) * repeat_and_add_lastdim(sigma_y, Nfit)

    popt = turnLastColumnVectorToDim(popt_vec)
    popt = np.transpose(popt, axes=tuple(np.concatenate([others_ndim, np.arange(others_ndim)], axis=None)))
    perr = np.transpose(perr, axes=tuple(np.concatenate([others_ndim, np.arange(others_ndim)], axis=None)))

    o.yHut = yHut
    o.sigma_y = sigma_y
    o.yHutErr = yHutErr
    o.popt = popt
    o.perr = perr
    o.yHutDer = yHutDer
    o.yHutDerErr = yHutDerErr

    return o


# 2022/3/28 define R as Rax (changed from measured position)
def Lscale(dat, dat_grad, Rax, err=None, err_grad=None):

    dat_L = - dat / dat_grad
    dat_RpL = Rax / dat_L

    if err is None or err_grad is None:
        err_L = None
        err_RpL = None
    else:
        rer = np.abs(err / dat)
        rer_grad = np.abs(err_grad / dat_grad)
        rer_L = np.sqrt(rer ** 2 + rer_grad ** 2)
        err_L = np.abs(dat_L * rer_L)
        err_RpL = np.abs(dat_RpL * rer_L)

    return dat_L, err_L, dat_RpL, err_RpL


def eta(dat_LT, err_LT, dat_Ln, err_Ln):

    rer_LT = np.abs(err_LT / dat_LT)
    rer_Ln = np.abs(err_Ln / dat_Ln)

    dat_eta = dat_Ln / dat_LT
    rer_eta = np.sqrt(rer_Ln**2 + rer_LT**2)
    err_eta = np.abs(dat_eta * rer_eta)

    return dat_eta, err_eta


def dMdreff(dMdR, dreffdR, dMdR_err=None):
    dMdreff = dMdR / dreffdR
    if dMdR_err is not None:
        dMdreff_err = np.abs(dMdR_err / dreffdR)
    else:
        dMdreff_err = None
    return dMdreff, dMdreff_err


def gyroradius(T_keV, B_T, kind="electron", A=1, Z=1, T_err=None):
    # T [keV], B [T]

    o = struct()
    if kind == "electron":
        o.m = const.me
        o.q = const.ee
    elif kind == "ion":
        o.m = const.mp * A
        o.q = Z * const.ee
    else:
        print("Please input correct kind name.")
        exit()

    o.gyrofreq_rads = o.q * B_T / o.m
    o.gyrofreq_Hz = o.gyrofreq_rads / (2 * np.pi)

    o.vT = np.sqrt(o.q * T_keV * 1e3 / o.m)
    o.gyroradius_m = o.vT / o.gyrofreq_rads

    if not T_err is None:
        o.vT_err = 1. / (2. * o.vT) * o.q * T_err * 1e3 / o.m
        o.gyroradius_err = o.vT_err / o.gyrofreq_rads

    return o


def moving_average(data, window_size, mode="same"):
    # window_size must be odd number.
    # mode = "same", "valid"

    if window_size == 1:
        moving_average = data
    else:
        if mode == "same":
            cumsum = np.insert(np.cumsum(np.append(np.insert(data, 0, [0] * ((window_size - 1)//2)),
                                                   [0] * ((window_size - 1)//2))), 0, 0)
        elif mode == "valid":
            cumsum = np.insert(np.cumsum(data), 0, 0)
        else:
            print("please input correct mode. 'same' or 'valid'")
            exit()
        moving_average = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    return moving_average


def divide(nu, de, n_err=None, d_err=None):

    quo = nu / de
    if n_err is None:
        if d_err is None:
            quo_err = None
        else:
            quo_err = np.abs(d_err/de * quo)
    else:
        if d_err is None:
            quo_err = np.abs(n_err/nu * quo)
        else:
            quo_err = np.abs(np.sqrt((n_err/nu)**2 + (d_err/de)**2) * quo)

    return quo, quo_err


def multiple(A, B, A_err=None, B_err=None):
    pro = A * B
    if A_err is None:
        if B_err is None:
            pro_err = None
        else:
            pro_err = np.abs(B_err / B * pro)
    else:
        if B_err is None:
            pro_err = np.abs(A_err / A * pro)
        else:
            pro_err = np.abs(np.sqrt((A_err / A)**2 + (B_err / B)**2) * pro)
    return pro, pro_err


def power(x, e, x_err=None):

    pow = x ** e
    if x_err is None:
        pow_err = None
    else:
        pow_err = np.abs(e * (x ** (e - 1)) * x_err)

    return pow, pow_err


def ion_mass(Zi, Ai):
    return const.mp * Zi + const.mn * (1 - Ai)


def scaled_RI_growthrate(resistivity, dPdr, Zi, Ai, resistivity_err=None, dPdr_err=None):

    A, A_err = power(resistivity, 1./3., resistivity_err)
    B, B_err = power(dPdr, 2./3., dPdr_err)
    mi = ion_mass(Zi, Ai)
    C = mi**(-1./3.)

    A *= C
    if A_err is not None:
        A_err *= C

    gamma, gamma_err = multiple(A, B, A_err, B_err)

    return gamma, gamma_err
