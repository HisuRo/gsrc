import numpy as np
from scipy.interpolate import interp1d
from scipy import signal, fft
import os
import inspect

import sys
import time

import gc
from nasu import calc
import copy


def flatten(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened


def interleave_lists(list1, list2):
    # interleave : to combine different things so that parts of one thing are put between parts of another thing
    interleaved_list = []
    for item1, item2 in zip(list1, list2):
        interleaved_list.append(item1)
        interleaved_list.append(item2)
    # list1 または list2 の長さが異なる場合、残りの要素を追加
    longer_list = list1 if len(list1) > len(list2) else list2
    interleaved_list.extend(longer_list[len(list2):] if len(list1) > len(list2) else longer_list[len(list1):])
    return interleaved_list


def connect_list_to_str(list, by="_", format_str=None):
    # for instance, format_str="{:.0f}"
    if format_str is None:
        connected_str = by.join(map(str, list))
    else:
        connected_str = by.join(map(lambda x: format_str.format(x), list))
    return connected_str


def makefftsampleidxs(tdat, tout, NFFT, NEns, NOV):
    # idxs_tout = ((tout - 10 * (subsn - 1)) / dT + 0.5).astype(int)
    idxs_tout = np.argmin(np.abs(np.tile(tdat, (len(tout), 1)) - np.reshape(tout, (len(tout), 1))), axis=-1)
    # ss_base = (- int(0.5 * NSamp + 0.5), int(0.5 * NSamp + 0.5))
    # idxs_samp = np.tile(np.arange(ss_base[0], ss_base[1]), (len(tout), 1)) + np.reshape(idxs_tout, (len(tout), 1))
    NSamp = NEns * NFFT - (NEns - 1) * NOV

    if NOV != 0:
        tmp = (np.reshape(np.arange(NEns * NFFT), (NEns, NFFT)).T - np.arange(0, NEns * NOV, NOV)).T
    else:
        tmp = np.reshape(np.arange(NEns * NFFT), (NEns, NFFT))
    idxs = np.transpose(np.tile(tmp, (len(tout), 1, 1)).T + idxs_tout - NSamp // 2)

    return idxs


def skip_comments(file):
    for line in file:
        if not line.startswith('#'):
            yield line


def mkdir(dirnm, abovedir=False):
    if abovedir:
        newdir = os.path.join(abovedir, dirnm)
    else:
        newdir = dirnm
    ifNotMake(newdir)
    return newdir


def deepCopy_list_multiply(list_ref, N_list):
    return [copy.deepcopy(list_ref) for i in range(N_list)]


def alternately_arrange_dat_and_err(dat_list, err_list):
    array = np.array([dat_list, err_list]).transpose((2, 0, 1))
    array = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
    return array


def check_shapes(arr_list):
    # 最初のndarrayのshapeを取得します
    shape = arr_list[0].shape

    # リスト内のすべてのndarrayのshapeが等しいかどうかを確認します
    for arr in arr_list:
        if arr.shape != shape:
            return False
    return True


def notNanInDat2d(dat2dList, axis, isany=True):
    isNanList = [0] * len(dat2dList)

    if not check_shapes(dat2dList):
        print('if not check_shapes(dat2dList)')
        exit()

    for ii, dat2d in enumerate(dat2dList):
        if isany:
            isNan = np.isnan(dat2d).any(axis=axis)
        else:
            isNan = np.isnan(dat2d).all(axis=axis)
        isNanList[ii] = isNan
    isNotNan = np.logical_not(np.logical_or.reduce(isNanList))
    dat2dOutList = [0] * len(dat2dList)
    if axis == 0:
        for ii, dat2d in enumerate(dat2dList):
            dat2dOutList[ii] = dat2d[:, isNotNan]
    if axis == 1:
        for ii, dat2d in enumerate(dat2dList):
            dat2dOutList[ii] = dat2d[isNotNan]

    return isNotNan, dat2dOutList


def suggestNewVer(version, fname):

    print(f'There is ver. {version:d} on {fname:s} !!!!\n')
    input('Push enter to continue >>> ')

    return


def makeReffArraysForLSMAtInterestRho(reff1d, rho1d, InterestingRho):

    InterestingReff = interp1d(rho1d, reff1d)(InterestingRho)

    shiftedReff = reff1d - InterestingReff
    reffIdxsForLSMAtInterest = np.argsort(np.abs(shiftedReff))
    shiftedReffsForLSMAtInterest = shiftedReff[reffIdxsForLSMAtInterest]

    return InterestingReff, reffIdxsForLSMAtInterest, shiftedReffsForLSMAtInterest


def makeReffsForLSMAtRhoOfInterest2d(reff2d, rho2d, rhoOfInterest):

    reffsOfInterest = np.zeros(len(rho2d))
    for i in range(len(rho2d)):
        reffOfInterest = interp1d(rho2d[i], reff2d[i])(rhoOfInterest)
        reffsOfInterest[i] = reffOfInterest
    shiftedReffs = reff2d - repeat_and_add_lastdim(reffsOfInterest, len(rho2d[0]))
    idxsForLSMAtInterest = argsort(np.abs(shiftedReffs))
    shiftedReffsForLSMAtInterest = shiftedReffs[idxsForLSMAtInterest[0], idxsForLSMAtInterest[1]]

    return reffsOfInterest, idxsForLSMAtInterest, shiftedReffsForLSMAtInterest


def argsort(array2d):
    idxs_right = np.argsort(array2d)
    idxs_left = repeat_and_add_lastdim(np.arange(len(array2d)).T, len(array2d[0]))
    return (idxs_left, idxs_right)


def repeat_and_add_lastdim(Array, Nrepeat):
    # Add a new axis to the end of the array and repeat each element N times in the direction of that axis.
    tmp = tuple(np.concatenate([np.array(Array.shape), 1], axis=None).astype(int))
    return np.repeat(np.reshape(Array, tmp), Nrepeat, axis=-1)


def ifNotMake(dirPath):
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)
    return dirPath


def getTimeIdxAndDats(time, time_at, datList):
    idx = np.nanargmin(np.abs(time - time_at))
    datList_at = [0]*len(datList)
    for ii, dat in enumerate(datList):
        datList_at[ii] = dat[idx]
    return idx, datList_at


def getTimeIdxsAndDats(time, startTime, endTime, datList, decimate=1):
    idxs = np.argwhere((time >= startTime) & (time <= endTime)).T[0][::decimate]
    if idxs[0] > 0:
        idxs = np.insert(idxs, 0, idxs[0] - 1)
    if idxs[-1] < len(time) - 1:
        idxs = np.append(idxs, idxs[-1] + 1)
    for ii, dat in enumerate(datList):
        datList[ii] = dat[idxs]
    return idxs, datList


def getXIdxsAndYs(xx, x_start, x_end, Ys_list, include_outerside=True):
    idxs = np.argwhere((xx >= x_start) & (xx <= x_end)).T[0]
    if include_outerside:
        if idxs[0] > 0:
            idxs = np.insert(idxs, 0, idxs[0] - 1)
        if idxs[-1] < len(xx) - 1:
            idxs = np.append(idxs, idxs[-1] + 1)
    for ii, dat in enumerate(Ys_list):
        Ys_list[ii] = dat[idxs]
    return idxs, Ys_list


def getXIdxsAndYs_2dalongLastAxis(xx, x_start, x_end, Ys_list, include_outerside=False):
    idxs = np.argwhere((xx >= x_start) & (xx <= x_end)).T[0]
    if include_outerside:
        if idxs[0] > 0:
            idxs = np.insert(idxs, 0, idxs[0] - 1)
        if idxs[-1] < len(xx) - 1:
            idxs = np.append(idxs, idxs[-1] + 1)
    for ii, dat in enumerate(Ys_list):
        Ys_list[ii] = dat[:, idxs]
    return idxs, Ys_list


def getXIdxsAndYs_3dalongLastAxis(xx, x_start, x_end, Ys_list, include_outerside=False):
    idxs = np.argwhere((xx >= x_start) & (xx <= x_end)).T[0]
    if include_outerside:
        if idxs[0] > 0:
            idxs = np.insert(idxs, 0, idxs[0] - 1)
        if idxs[-1] < len(xx) - 1:
            idxs = np.append(idxs, idxs[-1] + 1)
    for ii, dat in enumerate(Ys_list):
        Ys_list[ii] = dat[:, :, idxs]
    return idxs, Ys_list


def makeXidxs(xx, x_start, x_end, include_outerside=False):
    idxs = np.argwhere((xx >= x_start) & (xx <= x_end)).T[0]
    if include_outerside:
        if idxs[0] > 0:
            idxs = np.insert(idxs, 0, idxs[0] - 1)
        if idxs[-1] < len(xx) - 1:
            idxs = np.append(idxs, idxs[-1] + 1)
    return idxs


def findIndex(point, array1d):
    idx = np.nanargmin(np.abs(point - array1d))
    return idx


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

    if dat.shape != rho.shape:
        print('Improper data shape')
        exit()

    idxs_at = (range(time.size), np.nanargmin(np.abs(rho - rho_at), axis=-1))
    dat_at = dat[idxs_at]
    err_at = err[idxs_at]

    idxs_del = np.where((rho_at > np.nanmax(rho, axis=1)) | (rho_at < np.nanmin(rho, axis=1)))[0]
    dat_at[idxs_del] = np.nan
    err_at[idxs_del] = np.nan

    return dat_at, err_at


def takeat_rho_R(R, rho, rho_at):

    idxs_at = np.nanargmin(np.abs(rho - rho_at), axis=-1)
    R_at = R[idxs_at]

    idxs_del = np.where((rho_at > np.nanmax(rho, axis=-1)) | (rho_at < np.nanmin(rho, axis=-1)))[0]
    R_at[idxs_del] = np.nan

    return R_at


def takeat_tlist(dat, err, reff, rho, time, tlist):
    idxs_tlist = [np.nanargmin(np.abs(time - t)) for t in tlist]
    time_tl = time[idxs_tlist]
    reff_tl = reff[idxs_tlist]
    rho_tl = rho[idxs_tlist]
    dat_tl = dat[idxs_tlist]
    err_tl = err[idxs_tlist]
    return time_tl, reff_tl, rho_tl, dat_tl, err_tl


def takeYatX(Yseries, Xseries, Xvalue):
    return Yseries[np.argmin(np.abs(Xseries - Xvalue))]


def interpolate1d(time, val, err, t_ref):

    f = interp1d(time, val, bounds_error=False, fill_value=np.nan)
    fer = interp1d(time, err, bounds_error=False, fill_value=np.nan)
    val_intp = f(t_ref)
    err_intp = fer(t_ref)

    return val_intp, err_intp


def refPrevTime(time, val, err, t_ref):

    ft = interp1d(time, time, kind='previous', bounds_error=False, fill_value=np.nan)
    f = interp1d(time, val, kind='previous', bounds_error=False, fill_value=np.nan)
    fer = interp1d(time, err, kind='previous', bounds_error=False, fill_value=np.nan)
    tim_intp = ft(t_ref)
    val_intp = f(t_ref)
    err_intp = fer(t_ref)

    return tim_intp, val_intp, err_intp


def refNearestTime(time, val, err, t_ref):

    ft = interp1d(time, time, kind='nearest', bounds_error=False, fill_value=np.nan)
    f = interp1d(time, val, kind='nearest', bounds_error=False, fill_value=np.nan)
    fer = interp1d(time, err, kind='nearest', bounds_error=False, fill_value=np.nan)
    tim_intp = ft(t_ref)
    val_intp = f(t_ref)
    err_intp = fer(t_ref)

    return tim_intp, val_intp, err_intp


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


def datAtRhoByTimeVariatingRho(rho2d, dat2d, rho_at):
    datAtRho = [0]*len(rho2d)
    for i in range(len(rho2d)):
        datAtRho[i] = interp1d(rho2d[i], dat2d[i])(rho_at)
    datAtRho = np.array(datAtRho)
    return datAtRho


def pad_lists_to_ndarray(lists):
    max_length = max(len(lst) for lst in lists)
    padded_array = np.array([np.pad(lst, (0, max_length - len(lst)), constant_values=np.nan) for lst in lists])
    return padded_array


def get_current_file_and_line():
    # inspect.stack()は、現在のコールスタックを返します
    stack = inspect.stack()
    # stack[1]にはこの関数を呼び出したコードのフレームが格納されています
    frame = stack[1]
    # frame[1]はファイル名、frame[2]は行番号を表します
    filename = frame.filename
    lineno = frame.lineno
    return filename, lineno


def progress_bar():
    bar_length = 20
    for i in range(101):
        progress = "[" + "=" * (i // (100 // bar_length)) + " " * (bar_length - i // (100 // bar_length)) + "]"
        sys.stdout.write(f"\rProgress: {i}% {progress}")
        sys.stdout.flush()
        time.sleep(0.1)

    print("\nDone!")
