import numpy as np
from scipy import signal, fft, interpolate, optimize
import gc


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


def CVForBiSpecAna(NFFTs, NEns, NOV):
    NFFT1, NFFT2, NFFT3 = NFFTs

    idxMx3, idxNan = makeIdxsForCrossBiSpectrum(NFFTs)

    NSamp1 = NEns * NFFT1 - (NEns - 1) * NOV
    NSamp2 = NEns * NFFT2 - (NEns - 1) * NOV
    NSamp3 = NEns * NFFT3 - (NEns - 1) * NOV
    randND1 = np.random.normal(size=NSamp1)
    randND1 = toTimeSliceEnsemble(randND1, NFFT1, NEns, NOV)
    randND2 = np.random.normal(size=NSamp2)
    randND2 = toTimeSliceEnsemble(randND2, NFFT2, NEns, NOV)
    randND3 = np.random.normal(size=NSamp3)
    randND3 = toTimeSliceEnsemble(randND3, NFFT3, NEns, NOV)
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


def bandPass(xx, sampleRate, fp, fs, gpass, gstop):
    fn = sampleRate / 2                           # ナイキスト周波数
    wp = fp / fn                                  # ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  # ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)        # オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "band")           # フィルタ伝達関数の分子と分母を計算
    yy = signal.filtfilt(b, a, xx)                 # 信号に対してフィルタをかける
    return yy


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


def power_spectrogram_1s(ti, xx, dt, NFFT, window, Ndiv, Ntisp):

    idxs = np.arange(0, NFFT * Ndiv * Ntisp)
    idxs = idxs.reshape((Ntisp, Ndiv, NFFT))

    dtisp = Ndiv * NFFT * dt
    tisp = ti[idxs.T[0][0]] + 0.5 * dtisp
    xens = xx[idxs]

    win = signal.get_window(window, NFFT)
    enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    CG = np.abs(np.sum(win)) / NFFT

    div_CV = np.sqrt(1. / Ndiv)  # 分割平均平滑化による相対誤差の変化率
    sp_CV = div_CV

    rfreq = fft.rfftfreq(NFFT, dt)

    rfft_x = fft.rfft(xens * win)

    p_xx = np.real(rfft_x * rfft_x.conj())
    p_xx[:, :, 1:-1] *= 2
    p_xx_ave = np.mean(p_xx, axis=1)
    p_xx_err = np.std(p_xx, axis=1, ddof=1)
    p_xx_rerr = p_xx_err / np.abs(p_xx_ave) * sp_CV

    Fs = 1. / dt
    psd = p_xx_ave / (Fs * NFFT * enbw * (CG ** 2))
    psd_err = np.abs(psd) * p_xx_rerr

    dfreq = 1. / (NFFT * dt)
    print(f'Power x^2_bar             = {np.sum(xx[idxs][0] ** 2) / (NFFT * Ndiv):.3f}V^2 '
          f'@{tisp[0]:.3f}+-{0.5 * dtisp:.3f}s')
    print(f'Power integral of P(f)*df = {np.sum(psd[0] * dfreq):.3f}V^2'
          f' @{tisp[0]:.3f}+-{0.5 * dtisp:.3f}s')

    return tisp, rfreq, psd, psd_err


def power_spectrogram_2s(ti, xx, dt, NFFT, window, NEns, NOV):

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


def cross_spectre_2s(x, y, Fs, tana, dtsp, Nsp, dten, NEns, NFFT, window, Ndiv):

    dT = 1. / Fs

    tsp = np.linspace(0, tana - dtsp, Nsp)
    tmp = np.linspace(tsp, tsp + dten * (NEns - 1), num=NEns)
    tmp = (tmp / dT + 0.5).astype(int)
    idx_arr = np.linspace(tmp, tmp + NFFT - 1, num=NFFT, dtype=np.int32)
    idx_arr = np.transpose(idx_arr, (2, 1, 0))

    x_arr = x[idx_arr]
    y_arr = y[idx_arr]

    win = signal.get_window(window, NFFT)
    enbw = NFFT * np.sum(win ** 2) / (np.sum(win) ** 2)
    CG = np.abs(np.sum(win)) / NFFT

    # win_CV = np.sqrt(CG**2 * enbw)  # boxcar以外の窓関数の適用による相対誤差の変化率
    div_CV = np.sqrt(1. / Ndiv)  # 分割平均平滑化による相対誤差の変化率
    # sp_CV = win_CV * div_CV
    sp_CV = div_CV

    freq = fft.fftshift(fft.fftfreq(NFFT, dT))

    # https://watlab-blog.com/2020/07/24/coherence-function/

    fft_x = fft.fft(x_arr * win)
    fft_x = fft.fftshift(fft_x, axes=(2,))
    fft_y = fft.fft(y_arr * win)
    fft_y = fft.fftshift(fft_y, axes=(2,))

    c_xy = fft_y * fft_x.conj()
    p_xx = np.real(fft_x * fft_x.conj())
    p_yy = np.real(fft_y * fft_y.conj())

    c_xy_ave = np.mean(c_xy, axis=1)
    p_xx_ave = np.mean(p_xx, axis=1)
    p_yy_ave = np.mean(p_yy, axis=1)
    p_xx_err = np.std(p_xx, axis=1, ddof=1)
    p_yy_err = np.std(p_yy, axis=1, ddof=1)
    p_xx_rerr = p_xx_err / p_xx_ave * sp_CV
    p_yy_rerr = p_yy_err / p_yy_ave * sp_CV

    Kxy = np.real(c_xy)
    Qxy = - np.imag(c_xy)
    Kxy_ave = np.mean(Kxy, axis=1)
    Qxy_ave = np.mean(Qxy, axis=1)
    Kxy_err = np.std(Kxy, axis=1, ddof=1)
    Qxy_err = np.std(Qxy, axis=1, ddof=1)
    Kxy_rerr = Kxy_err / Kxy_ave * sp_CV
    Qxy_rerr = Qxy_err / Qxy_ave * sp_CV

    CSDxy = np.abs(c_xy_ave) / (Fs * NFFT * enbw * CG**2)
    cs_err = np.sqrt((Kxy_ave * Kxy_err)**2 + (Qxy_ave * Qxy_err)**2) / \
             np.abs(c_xy_ave)
    cs_rerr = cs_err / np.abs(c_xy_ave) * sp_CV
    CSDxy_err = CSDxy * sp_CV

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


def biCoherenceSq(BSpec, BSpecStd, X1, X2, X3, NFFTs, NEns, NOV):

    CV_X2X1, CV_X3 = CVForBiSpecAna(NFFTs, NEns, NOV)

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
    idxNan2 = np.where((freq2 >= freq1) | (freq2 < - 0.5 * freq1))

    biCohSq[idxNan] = np.nan
    biCohSqErr[idxNan] = np.nan
    biPhs[idxNan] = np.nan
    biPhsErr[idxNan] = np.nan
    biCohSq[idxNan2] = np.nan
    biCohSqErr[idxNan2] = np.nan
    biPhs[idxNan2] = np.nan
    biPhsErr[idxNan2] = np.nan

    return biCohSq, biCohSqErr, biPhs, biPhsErr


def crossBiSpecAna(freqs, XX, YY, ZZ, NFFTs, NEns, NOV):
    freqx, freqy, freqz = freqs

    NFFTx, NFFTy, NFFTz = NFFTs
    idxMxz, idxNan = makeIdxsForCrossBiSpectrum(NFFTs)

    XX = np.reshape(XX, (NEns, 1, NFFTx))
    YY = np.reshape(YY, (NEns, NFFTy, 1))
    ZZ = ZZ[:, idxMxz]

    BSpec, BSpecStd, BSpecReStd, BSpecImStd = biSpectrum(XX, YY, ZZ)

    biCohSq, biCohSqErr = biCoherenceSq(BSpec, BSpecStd, XX, YY, ZZ, NFFTs, NEns, NOV)
    biPhs, biPhsErr = biPhase(BSpec, BSpecReStd, BSpecImStd)

    # symmetry
    freq1 = np.tile(freqx, (NFFTy, 1))
    freq2 = np.tile(freqy, (NFFTx, 1)).T
    idxNan2 = np.where((freq2 >= freq1) | (freq2 < - 0.5 * freq1))

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
    Swx02 = np.sum(wx02, axis=-1)
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
    Swx02 = np.sum(wx02, axis=-1)
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

    x2 = x**2
    x3 = x**3
    x4 = x**4
    wx04 = np.array([np.ones(weight.shape), x, x2, x3, x4]) * weight
    Swx04 = np.sum(wx04, axis=-1)
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
    idxs_calc = make_idxs_for_moving_LSM(NR, Nfit)
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
    tmp = tuple(np.concatenate([np.array(Array.shape), 1], axis=None))
    return np.repeat(np.reshape(Array, tmp), Nrepeat, axis=-1)


def make_fitted_profiles_with_MovingPolyLSM(reff, raw_profiles, profiles_errs, window_len, poly=2):

    if reff.shape != raw_profiles.shape:
        print('Improper data shape')
        exit()
    profiles_count, profile_len = reff.shape
    idxs_for_Moving = make_idxs_for_MovingLSM(profile_len, window_len)
    output_profiles_count = idxs_for_Moving.shape[0]

    reff_for_Moving = reff[:, idxs_for_Moving]
    reff_avgs = np.average(reff_for_Moving, axis=-1)
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
        fitted_profiles_wo_average = aa * reff_for_fitting + bb
        fitted_profiles_errs = np.sqrt(np.sum(profiles_errs_for_fitting**2 + (profiles_for_fitting - fitted_profiles_wo_average)**2, axis=-1)/(window_len - 2))
        S = np.sqrt(np.sum((reff_for_fitting - repeat_and_add_lastdim(reff_avgs, window_len)) ** 2, axis=-1) / window_len)
        fitted_profs_grads_errs = perr[0]

    elif poly == 2:
        print('polynomial 2\n')
        popt, perr = poly2_LSM(reff_for_fitting, profiles_for_fitting, profiles_errs_for_fitting)
        fitted_profs_gradients = popt[1]
        fitted_profiles = popt[2]

        aa = repeat_and_add_lastdim(popt[0], window_len)
        bb = repeat_and_add_lastdim(popt[1], window_len)
        cc = repeat_and_add_lastdim(popt[2], window_len)
        fitted_profiles_wo_average = aa * reff_for_fitting**2 + bb * reff_for_fitting + cc
        fitted_profiles_errs = np.sqrt(np.sum(profiles_errs_for_fitting**2 + (profiles_for_fitting - fitted_profiles_wo_average)**2, axis=-1)/(window_len - 2))
        fitted_profs_grads_errs = perr[1]

    else:
        print('It has not developed yet...\n')
        exit()

    return reff_avgs, fitted_profiles, fitted_profiles_errs, fitted_profs_gradients, fitted_profs_grads_errs


def gradient_reg(R, reff, a99, dat, err, Nfit, poly):

    # reff_f, dat_grad, err_grad, dat_reg, err_reg = gradient_reg_reff(reff, dat, err, Nfit)
    reff_f, dat_reg, err_reg, dat_grad, err_grad = make_fitted_profiles_with_MovingPolyLSM(reff, dat, err, Nfit, poly=poly)

    Nt, NR = reff.shape
    NRf = NR - Nfit + 1
    reff_ext = np.repeat(np.reshape(reff, (Nt, 1, NR)), NRf, axis=1)  # (Nt, NRf, NR)
    reff_f_ext = np.repeat(np.reshape(reff_f, (Nt, NRf, 1)), NR, axis=2)  # (Nt, NRf, NR)

    dreff = reff_f_ext - reff_ext
    idxs1 = np.nanargmin(np.where(dreff <= 0, np.nan, dreff), axis=-1)
    idxs2 = np.nanargmax(np.where(dreff >= 0, np.nan, dreff), axis=-1)

    R1 = R[idxs1]
    R2 = R[idxs2]
    idxs_t = np.tile(np.reshape(np.arange(Nt), (Nt, 1)), (1, NRf))
    reff1 = reff[idxs_t, idxs1]
    reff2 = reff[idxs_t, idxs2]

    R_f = (R2 - R1) / (reff2 - reff1) * (reff_f - reff1) + R1

    a99 = np.repeat(np.reshape(a99, (Nt, 1)), NRf, axis=-1)
    rho_f = reff_f / a99

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


def weighted_average_1D(x1D, weight1D):
    Sw = np.sum(weight1D)
    wx = x1D * weight1D
    Swx = np.sum(wx)
    xm = Swx / Sw
    w2 = weight1D ** 2
    Sw2 = np.sum(w2)
    errsq = (x1D - np.full(x1D.T.shape, xm).T) ** 2
    werrsq = weight1D * errsq
    Swerrsq = np.sum(werrsq)
    U = Sw / (Sw ** 2 - Sw2) * Swerrsq
    xerr = np.sqrt(U)

    wwerrsq = weight1D * werrsq
    Swwerrsq = np.sum(wwerrsq)
    Um = Sw / (Sw ** 2 - Sw2) * Swwerrsq / Sw
    xmerr = np.sqrt(Um)

    return xm, xerr, xmerr


def weighted_average_2D(x2D, weight2D):
    Sw = np.sum(weight2D, axis=1)
    wx = x2D * weight2D
    Swx = np.sum(wx, axis=1)
    xm = Swx / Sw
    w2 = weight2D ** 2
    Sw2 = np.sum(w2, axis=1)
    errsq = (x2D - np.full(x2D.T.shape, xm).T) ** 2
    werrsq = weight2D * errsq
    Swerrsq = np.sum(werrsq, axis=1)
    U = Sw / (Sw ** 2 - Sw2) * Swerrsq
    xerr = np.sqrt(U)
    
    wwerrsq = weight2D * werrsq
    Swwerrsq = np.sum(wwerrsq, axis=1)
    Um = Sw / (Sw ** 2 - Sw2) * Swwerrsq / Sw
    xmerr = np.sqrt(Um)

    return xm, xerr, xmerr
