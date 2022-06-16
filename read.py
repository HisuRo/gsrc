import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
from parse import parse
from nasu.myEgdb import LoadEG
import nasu.LHDRetrieve as LHDR


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
    dat_Tipol = dat_Ti[:, idxs_pol]
    dat_Titor = dat_Ti[:, idxs_tor]
    err_Tipol = err_Ti[:, idxs_pol]
    err_Titor = err_Ti[:, idxs_tor]

    idxs_sort_pol = np.argsort(R_pol)
    idxs_sort_tor = np.argsort(R_tor)
    R_pol = R_pol[idxs_sort_pol]
    reff_pol = reff_pol[:, idxs_sort_pol]
    rho_pol = rho_pol[:, idxs_sort_pol]
    dat_Tipol = dat_Tipol[:, idxs_sort_pol]
    err_Tipol = err_Tipol[:, idxs_sort_pol]
    R_tor = R_tor[idxs_sort_tor]
    reff_tor = reff_tor[:, idxs_sort_tor]
    rho_tor = rho_tor[:, idxs_sort_tor]
    dat_Titor = dat_Titor[:, idxs_sort_tor]
    err_Titor = err_Titor[:, idxs_sort_tor]

    return time, R_pol, R_tor, \
           reff_pol, reff_tor, rho_pol, rho_tor, \
           dat_Tipol, err_Tipol, dat_Titor, err_Titor


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
    dat_Ti = dat_Ti[:, idxs_sort]
    err_Ti = err_Ti[:, idxs_sort]

    return time, R, \
           reff, rho, \
           dat_Ti, err_Ti


def choose_ch():

    # information
    devices = {0: 'BS (3-O)', 1: 'DBS (3-O)', 2: 'DBS (9-O)'}

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

    print('\n')

    return sn, subsn


def input_FFT(inputFFTfile):

    inputFFT_df = pd.read_csv(inputFFTfile, header=None, index_col=0)
    Nfft_pw = int(inputFFT_df.at['Nfftpwr', 1])
    window = str(inputFFT_df.at['window', 1])
    Nens = int(inputFFT_df.at['Nens', 1])
    Nfft = 2 ** Nfft_pw

    print('\n')

    return Nfft_pw, Nfft, window, Nens


def input_fDSk(inputFFTfile):

    inputFFT_df = pd.read_csv(inputFFTfile, header=None, index_col=0)
    fdelDopp_k = int(inputFFT_df.at['fdelk', 1])
    frangeSk_dict = {0: '3-30kHz', 1: '30-150kHz', 2: '150-490kHz', 3: '20-490kHz',
                     4: '100-490kHz', 5: '20-200kHz', 6: '200-500kHz', 99: 'other'}
    critfrangeSk = 2
    if critfrangeSk == 1:
        No_frangeSk = int(input(f'Which frequency range use ?\n'
                                f'{frangeSk_dict} \n'
                                f'>>> '))
    elif critfrangeSk == 2:
        No_frangeSk = int(inputFFT_df.at['NofrangeSk', 1])

    if No_frangeSk == 99:
        frangeSk_l_k = int(input('Lowest Frequency [kHz] >>> '))
        frangeSk_h_k = int(input('Highest Frequency [kHz] >>> '))
        frangeSk_k = [frangeSk_l_k, frangeSk_h_k]
    else:
        frangeSk_list = [[3, 30], [30, 150], [150, 490], [20, 490], [100, 490], [20, 200], [200, 500]]
        frangeSk_k = frangeSk_list[No_frangeSk]

    return fdelDopp_k, frangeSk_k


def input_specrange(inputfile):

    input_df = pd.read_csv(inputfile, header=None, index_col=0)

    bottom = int(input_df.at['bottomdB', 1])
    top = int(input_df.at['topdB', 1])

    return bottom, top


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


def LHD_IQ_ss(sn, subsn, diagname, chs, ss):

    Idat, Iprms = LHDR.RetrieveData_ss(diagname, sn, subsn, chs[0], ss)
    Qdat, Qprms = LHDR.RetrieveData_ss(diagname, sn, subsn, chs[1], ss)

    print('\n')

    return Idat, Qdat


def LHD_time(sn, subsn, diagname, ch):

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


def LHD_ss(sn, subsn, diagname, ch, ss):

    dat, prms = LHDR.RetrieveData_ss(diagname, sn, subsn, ch, ss)
    tdat, tprms = LHDR.RetrieveTime(diagname, sn, subsn, ch)
    tdat = tdat[np.arange(ss[0], ss[1] + 1)]

    tsize = len(tdat)
    dT = parse('{:f}{:S}', tprms['ClockCycle'][0])[0]
    Fs = int(prms['SamplingClock'][0])

    print('\n')

    return tdat, dat, dT, Fs, tsize


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


def ech_local(sn, tstart_out, tend_out):
    dir_heat = os.path.join(dirs()[1], '04-heat')
    fnm_ech = f'#{sn:d}.csv'
    path_ech = os.path.join(dir_heat, 'ech', fnm_ech)
    if os.path.isfile(path_ech):
        ech = np.loadtxt(path_ech, delimiter=',').T
        time_ech = ech[0]
        idxs_use_ech = np.where((time_ech >= tstart_out) & (time_ech <= tend_out))
        dat_ech = ech[1]
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


def fDSk_local(sn, tstart, tend, diag, chIQ, dT, Nfft_pw, window, Nens, flim_k, fdelDopp_k, frangeSk_k,
               sw_BSmod, sw_nb5mod, tstart_out, tend_out):
    dir_fDSk = os.path.join(dirs()[1], '02-PowerSpecfDSk')
    fnm_fDSk = f'#{sn:d}_{tstart:g}-{tend:g}s_' \
               f'{diag:s}_{chIQ[0]:d}_{chIQ[1]:d}_' \
               f'{dT:s}_2^{Nfft_pw:d}_{window:s}_{Nens:g}_' \
               f'{flim_k:.0f}kHz_{fdelDopp_k:d}kHz_{frangeSk_k[0]:d}-{frangeSk_k[1]:d}kHz.csv'
    path_fDSk = os.path.join(dir_fDSk, 'csvfDSk', fnm_fDSk)

    if os.path.isfile(path_fDSk):
        fDSk = np.loadtxt(path_fDSk, delimiter=',').T
        time_fDSk = fDSk[0]
        idxs_use_fDSk = np.where((time_fDSk >= tstart_out) & (time_fDSk <= tend_out))
        dat_Sk, err_Sk = fDSk[5:7]
        dat_Sk = np.sqrt(dat_Sk) * 1e4
        err_Sk = 0.5 / np.sqrt(dat_Sk) * err_Sk * 1e4
        time_fDSk = time_fDSk[idxs_use_fDSk]
        dat_Sk = dat_Sk[idxs_use_fDSk]
        err_Sk = err_Sk[idxs_use_fDSk]

        if sw_BSmod == 1:
            dir_mod = os.path.join(dirs()[1], '03-BSmod')
            fnm_mod = f'#{sn:d}_{tstart:g}-{tend:g}s_2^{Nfft_pw:d}_{Nens:d}.csv'
            path_mod = os.path.join(dir_mod, 'csv', fnm_mod)
            BSmod = np.loadtxt(path_mod, delimiter=',').T
            time_mod, dat_mod = BSmod
            idx_del = np.where(dat_mod < 0.75)[0]
            dat_Sk[idx_del] = np.nan
        elif sw_nb5mod == 1:
            time_nb, dat_nb1, dat_nb2, dat_nb3, dat_nb4a, dat_nb5a = nb_local(sn, tstart_out, tend_out)
            dat_nb5a_fDSk = interp1d(time_nb, dat_nb5a)(time_fDSk)
            dat_diff_nb5a = np.diff(dat_nb5a_fDSk)
            idx_del = np.where(dat_diff_nb5a < -2)[0]
            dat_Sk[idx_del] = np.nan
    else:
        print(f'{path_fDSk} is not exist. \n')
        time_fDSk = np.nan
        dat_Sk = np.nan
        err_Sk = np.nan

    return time_fDSk, dat_Sk, err_Sk


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
