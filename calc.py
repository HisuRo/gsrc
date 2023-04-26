import numpy as np
from scipy import signal, fft, interpolate, optimize
import gc
import matplotlib.pyplot as plt
from nasu import proc, plot
import os


def weightedAverage_alongLastAxis(datArray, errArray):
    datAvgs = np.average(datArray, axis=-1, weights=1. / (errArray ** 2))
    datStds = np.sqrt(np.var(datArray, axis=-1) + np.average(errArray ** 2, axis=-1))
    return datAvgs, datStds


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


def gradRegAtInterestRhoWithErr(ReffAvg, RhoAvg, Dat, DatEr, InterestingRho, NFit, polyGrad):
    print('Newest ver. -> calc.timeSeriesRegGradAtRhoOfInterest')
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


def timeSeriesRegGradAtRhoOfInterest(reff2d, rho2d, dat, err, rhoOfInterest, NFit, polyGrad, fname_base=None, dir_name=None):

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
            plot.check(0.1)

    return reffsOfInterest, RegDat, RegDatEr, DatGrad, DatGradEr


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


def timeAverageProfiles(dat2d, err=np.array([False])):
    if err.all():
        idxs_isnanInDat2d = np.isnan(dat2d)
        idxs_isnanInErr = np.isnan(err)
        idxs_isnan = idxs_isnanInErr + idxs_isnanInDat2d
        dat2d[idxs_isnan] = np.nan
        err[idxs_isnan] = np.nan

        avg = np.nanmean(dat2d, axis=0)
        std = np.sqrt(np.nanvar(dat2d, axis=0) + np.nanmean(err ** 2, axis=0))
    else:
        avg = np.nanmean(dat2d, axis=0)
        std = np.nanstd(dat2d, axis=0, ddof=1)

    return avg, std


def timeAverageDatByRefs(timeDat, dat, err, timeRef):

    proc.suggestNewVer(2, 'timeAverageDatByRefs')

    dtDat = timeDat[1] - timeDat[0]
    dtRef = timeRef[1] - timeRef[0]
    dNDatRef = int(dtRef / dtDat + 0.5)
    timeRef_ext = repeat_and_add_lastdim(timeRef, len(timeDat))
    idxDatAtRef = np.argsort(np.abs(timeDat - timeRef_ext))[:, :dNDatRef]

    datAtRef = dat[idxDatAtRef]
    errAtRef = err[idxDatAtRef]
    dat_Ref = np.nanmean(datAtRef, axis=1)
    err_Ref = np.sqrt(np.nanvar(datAtRef, axis=1) + np.nanmean(errAtRef ** 2, axis=1))

    return dat_Ref, err_Ref


def timeAverageDatByRefs_v2(timeDat, dat, timeRef, err=np.array([False])):
    dtDat = timeDat[1] - timeDat[0]
    dtRef = timeRef[1] - timeRef[0]
    dNDatRef = int(dtRef / dtDat + 0.5)
    timeRef_ext = repeat_and_add_lastdim(timeRef, len(timeDat))
    idxDatAtRef = np.argsort(np.abs(timeDat - timeRef_ext))[:, :dNDatRef]

    datAtRef = dat[idxDatAtRef]
    dat_Ref = np.nanmean(datAtRef, axis=1)
    
    if err.all():
        errAtRef = err[idxDatAtRef]
        err_Ref = np.sqrt(np.nanvar(datAtRef, axis=1) + np.nanmean(errAtRef ** 2, axis=1))
    else:
        err_Ref = np.nanstd(datAtRef, axis=1, ddof=1)

    return dat_Ref, err_Ref


def timeAverageDatListByRefs(timeDat, datList, timeRef, errList=None):

    datRefList = [0] * len(datList)
    errRefList = [0] * len(datList)

    for ii, dat in enumerate(datList):
        if errList is None:
            datRef, errRef = timeAverageDatByRefs_v2(timeDat, dat, timeRef)
        else:
            err = errList[ii]
            datRef, errRef = timeAverageDatByRefs_v2(timeDat, dat, timeRef, err)
        datRefList[ii] = datRef
        errRefList[ii] = errRef

    return datRefList, errRefList


def dB(spec, spec_err):
    spec_db = 10 * np.log10(spec)
    spec_err_db = 10 / np.log(10) / spec * spec_err
    return spec_db, spec_err_db


def toZeroMeanTimeSliceEnsemble(xx, NFFT, NEns, NOV):
    xens = toTimeSliceEnsemble(xx, NFFT, NEns, NOV)
    xensAvg = np.average(xens, axis=-1)
    xensAvg = repeat_and_add_lastdim(xensAvg, NFFT)
    xens -= xensAvg

    return xens


def toTimeSliceEnsemble(xx, NFFT, NEns, NOV):

    print(f'Overlap ratio: {NOV / NFFT * 100:.0f}%\n')

    Nsamp = NEns * NFFT - (NEns - 1) * NOV
    if len(xx) != Nsamp:
        print('The number of samples is improper. \n')
        exit()
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

    ND1 = np.reshape(ND1, (NEns, NFFT1, 1))
    ND2 = np.reshape(ND2, (NEns, 1, NFFT2))
    ND3 = ND3[:, idxMx3]
    ND1ND2 = np.matmul(ND1, ND2)
    ND1ND2AbsSq = np.abs(ND1ND2) ** 2
    ND3AbsSq = np.abs(ND3) ** 2

    CV_ND1ND2 = CV(ND1ND2AbsSq)
    CV_ND3 = CV(ND3AbsSq)

    return CV_ND1ND2, CV_ND3


def getWindowAndCoefs(NFFT, window, NEns, NOV):

    win = signal.get_window(window, NFFT)
    enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    CG = np.abs(np.sum(win)) / NFFT

    CV = CV_overlap(NFFT, NEns, NOV)

    return win, enbw, CG, CV


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
    fft_x = fft.fft(xens * win)
    freq = fft.fftshift(freq)
    fft_x = fft.fftshift(fft_x)

    return freq, fft_x


def NSampleForFFT(NFFT, NEns, NOV):
    return NEns * NFFT - (NEns - 1) * NOV


def power_spectre_1s(xx, dt, NFFT, window, NEns, NOV):

    xens = toTimeSliceEnsemble(xx, NFFT, NEns, NOV)
    win, enbw, CG, CV = getWindowAndCoefs(NFFT, window, NEns, NOV)

    rfreq, rfft_x = fourier_components_1s(xens, dt, NFFT, win)
    p_xx = np.real(rfft_x * rfft_x.conj())
    p_xx[:, 1:-1] *= 2
    p_xx_ave = np.mean(p_xx, axis=0)
    p_xx_std = np.std(p_xx, axis=0, ddof=1)
    p_xx_rerr = p_xx_std / np.abs(p_xx_ave) * CV

    Fs = 1. / dt
    psd = p_xx_ave / (Fs * NFFT * enbw * (CG ** 2))
    psd_err = psd * p_xx_rerr

    dfreq = 1. / (NFFT * dt)
    print(f'Power x^2_bar            ={np.sum(xx**2) / len(xx)}')
    print(f'Power integral of P(f)*df={np.sum(psd * dfreq)}\n')

    return rfreq, psd, psd_err


def power_spectre_2s(xx, dt, NFFT, window, NEns, NOV):

    print(f'Overlap ratio: {NOV/NFFT*100:.0f}%\n')

    Nsamp = NEns * NFFT - (NEns - 1) * NOV
    if len(xx) != Nsamp:
        print('The number of samples is improper. \n')
        exit()
    else:
        print(f'The number of samples: {Nsamp:d}')

    if NOV != 0:
        idxs = (np.reshape(np.arange(NEns * NFFT), (NEns, NFFT)).T - np.arange(0, NEns * NOV, NOV)).T
    else:
        idxs = np.reshape(np.arange(NEns * NFFT), (NEns, NFFT))
    xens = xx[idxs]

    win = signal.get_window(window, NFFT)
    enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    CG = np.abs(np.sum(win)) / NFFT

    CV = CV_overlap(NFFT, NEns, NOV)

    freq = fft.fftshift(fft.fftfreq(NFFT, dt))

    fft_x = fft.fftshift(fft.fft(xens * win), axes=1)

    p_xx = np.real(fft_x * fft_x.conj())
    p_xx_ave = np.mean(p_xx, axis=0)
    p_xx_err = np.std(p_xx, axis=0, ddof=1)
    p_xx_rerr = p_xx_err / np.abs(p_xx_ave) * CV

    Fs = 1. / dt
    psd = p_xx_ave / (Fs * NFFT * enbw * (CG ** 2))
    psd_err = np.abs(psd) * p_xx_rerr

    dfreq = 1. / (NFFT * dt)
    print(f'Power x^2_bar            ={np.sum(np.real(xx*np.conj(xx))) / Nsamp}')
    print(f'Power integral of P(f)*df={np.sum(psd * dfreq)}')

    return freq, psd, psd_err


def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y                                      #フィルタ後の信号を返す


def bandPass(xx, sampleRate, fp, fs, gpass, gstop, cut=False):
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


def highpass(x, samplerate, fp, fs, gpass, gstop):
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


def power_spectrogram_1s(ti, xx, dt, NFFT, window, NEns, NOV):

    print(f'Overlap ratio: {NOV/NFFT*100:.0f}%\n')

    Nsamp = NEns * NFFT - (NEns - 1) * NOV
    Nsp = int(len(ti) / Nsamp + 0.5)

    if len(xx) % Nsamp != 0 or len(ti) % Nsamp != 0 or len(xx) != len(ti):
        print('The number of data points is improper. \n')
        exit()
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

    proc.suggestNewVer(2)

    print(f'Overlap ratio: {NOV/NFFT*100:.0f}%\n')

    Nsamp = NEns * NFFT - (NEns - 1) * NOV
    Nsp = int(len(ti) / Nsamp + 0.5)

    if len(xx) % Nsamp != 0 or len(ti) % Nsamp != 0 or len(xx) != len(ti):
        print('The number of data points is improper. \n')
        exit()
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


def power_spectrogram_2s_v2(ti, xx, dt, NFFT, window, NEns, NOV):

    print(f'Overlap ratio: {NOV / NFFT * 100:.0f}%\n')

    Nsamp = NEns * NFFT - (NEns - 1) * NOV
    Nsp = int(len(ti) / Nsamp + 0.5)

    if len(xx) % Nsamp != 0 or len(ti) % Nsamp != 0 or len(xx) != len(ti):
        print('The number of data points is improper. \n')
        exit()
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

    CV = CV_overlap(NFFT, NEns, NOV)

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


def cross_cor(sig, ref, dt):

    # CCF
    CCF = signal.correlate(sig, ref, mode='full') / len(sig)
    lags = signal.correlation_lags(len(sig), len(ref), mode='full') * dt

    # CCC
    Csig0 = np.sum(sig**2)/len(sig)
    Cref0 = np.sum(ref**2)/len(ref)
    CCC = CCF / np.sqrt(Csig0 * Cref0)

    return lags, CCF, CCC


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


def cross_spectre_2s(x, y, Fs, NEns, NFFT, window, NOV):

    dT = 1. / Fs

    x_arr = toZeroMeanTimeSliceEnsemble(x, NFFT, NEns, NOV)
    y_arr = toZeroMeanTimeSliceEnsemble(y, NFFT, NEns, NOV)

    win = signal.get_window(window, NFFT)
    enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    CG = np.abs(np.sum(win)) / NFFT

    CV = CV_overlap(NFFT, NEns, NOV)

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


def biSpectrum(X1, X2, X3):

    X2X1 = np.matmul(X2, X1)
    X3Conj = np.conjugate(X3)
    X2X1X3Conj = X2X1 * X3Conj
    BSpec = np.average(X2X1X3Conj, axis=0)
    BSpecStd = np.std(X2X1X3Conj, axis=0, ddof=1)
    BSpecReStd = np.std(np.real(X2X1X3Conj), axis=0, ddof=1)
    BSpecImStd = np.std(np.imag(X2X1X3Conj), axis=0, ddof=1)

    return BSpec, BSpecStd, BSpecReStd, BSpecImStd


def biCoherenceSq(BSpec, BSpecStd, X1, X2, X3, NFFTs, NEns, NOVs):

    CV_X2X1, CV_X3 = CVForBiSpecAna(NFFTs, NEns, NOVs)

    X2X1 = np.matmul(X2, X1)
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


def biPhase(BSpec, BSpecReStd, BSpecImStd):

    BSpecRe = np.real(BSpec)
    BSpecIm = np.imag(BSpec)

    BSpecReRer = BSpecReStd / BSpecRe
    BSpecImRer = BSpecImStd / BSpecIm

    BSpecImRe = BSpecIm / BSpecRe
    BSpecImReRer = np.sqrt(BSpecImRer ** 2 + BSpecReRer ** 2)
    BSpecImReErr = np.abs(BSpecImRe * BSpecImReRer)
    biPhs = np.arctan2(BSpecIm, BSpecRe)
    biPhsErr = 1. / (1. + BSpecImRe ** 2) * BSpecImReErr
    biPhs = np.degrees(biPhs)
    biPhsErr = np.degrees(biPhsErr)

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


def makeIdxsForCrossBiSpectrum(NFFTs):
    NFFTx, NFFTy, NFFTz = NFFTs

    idxf0x = int(NFFTx / 2 + 0.5)
    idxf0y = int(NFFTy / 2 + 0.5)
    idxf0z = int(NFFTz / 2 + 0.5)

    idxMx1 = np.tile(np.arange(NFFTx), (NFFTy, 1))
    idxMx2 = np.tile(np.arange(NFFTy), (NFFTx, 1)).T
    coefMx1 = idxMx1 - idxf0x
    coefMx2 = idxMx2 - idxf0y
    coefMx3 = coefMx1 + coefMx2
    idxMx3 = coefMx3 + idxf0z
    idxNan = np.where(((idxMx3 < 0) | (idxMx3 >= NFFTz)))
    idxMx3[idxNan] = False

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
    freqx, freqy, freqz = freqs

    NFFTx, NFFTy, NFFTz = NFFTs
    NOVx, NOVy, NOVz = NOVs
    idxMxz, idxNan = makeIdxsForCrossBiSpectrum(NFFTs)

    XX = np.reshape(XX, (NEns, 1, NFFTx))
    YY = np.reshape(YY, (NEns, NFFTy, 1))
    ZZ = ZZ[:, idxMxz]

    BSpec, BSpecStd, BSpecReStd, BSpecImStd = biSpectrum(XX, YY, ZZ)

    biCohSq, biCohSqErr = biCoherenceSq(BSpec, BSpecStd, XX, YY, ZZ, NFFTs, NEns, NOVs)
    biPhs, biPhsErr = biPhase(BSpec, BSpecReStd, BSpecImStd)

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
        exit()

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
        exit()

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
        exit()

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
        exit()

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


def gradient(R, reff, rho, dat, err, Nfit):

    Nt, NR = reff.shape
    NRf = NR - Nfit + 1

    idxs_calc = np.full((NRf, Nfit), np.arange(Nfit)).T  # (Nfit, NRf)
    idxs_calc = idxs_calc + np.arange(NRf)  # (Nfit, NRf)
    idxs_calc = idxs_calc.T  # (NRf, Nfit)
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
        exit()
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
        exit()

    return reff_avgs, fitted_profiles, fitted_profiles_errs, fitted_profs_gradients, fitted_profs_grads_errs


def make_fitted_profiles_with_MovingPolyLSM_v2(reff, raw_profiles, profiles_errs, window_len, poly=1):

    if reff.shape != raw_profiles.shape:
        print('Improper data shape')
        exit()
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
        exit()
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
        exit()

    return reff_avgs, fitted_profiles, fitted_profiles_errs, fitted_profs_gradients, fitted_profs_grads_errs


def linInterp1dOf2dDat(x2d, y1d, x2d_ref):

    Nt, Ny = x2d.shape
    Nt_ref, Ny_ref = x2d_ref.shape
    if Nt != Nt_ref:
        print('Error: Nt != Ntf')
        exit()

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


# 2022/3/28 define R as Rax (changed from measured position)
def Lscale(dat, err, dat_grad, err_grad, Rax):

    rer = np.abs(err / dat)
    rer_grad = np.abs(err_grad / dat_grad)

    dat_L = - dat / dat_grad
    rer_L = np.sqrt(rer ** 2 + rer_grad ** 2)
    err_L = np.abs(dat_L * rer_L)

    dat_RpL = Rax / dat_L
    err_RpL = np.abs(dat_RpL * rer_L)

    return dat_L, err_L, dat_RpL, err_RpL


def eta(dat_LT, err_LT, dat_Ln, err_Ln):

    rer_LT = np.abs(err_LT / dat_LT)
    rer_Ln = np.abs(err_Ln / dat_Ln)

    dat_eta = dat_Ln / dat_LT
    rer_eta = np.sqrt(rer_Ln**2 + rer_LT**2)
    err_eta = np.abs(dat_eta * rer_eta)

    return dat_eta, err_eta


# 2022/3/28 define Tratio as Ti / Te ( from tau = Te / Ti )
# 2022/6/14 redefine Tratio as Te / Ti
def Tratio(Te, Ti, Te_err, Ti_err):

    Tratio = Te / Ti

    Te_rerr = Te_err / Te
    Ti_rerr = Ti_err / Ti
    Tratio_rerr = np.sqrt(Te_rerr ** 2 + Ti_rerr ** 2)
    Tratio_err = Tratio * Tratio_rerr

    return Tratio, Tratio_err


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
