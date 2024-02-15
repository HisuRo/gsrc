# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:25:50 2024

@author: nata
"""

import sys
sys.path.append('g:/')
from nasu import calc, read_w7x
import w7xarchive
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

class struct:
    pass


def three_gaussians(x, a1, x1, sigma1, a2, x2, sigma2, a3, x3, sigma3):
    return a1*np.exp(- ((x - x1) / sigma1)**2) + a2*np.exp(- ((x - x2) / sigma2)**2) + a3*np.exp(- ((x - x3) / sigma3)**2)
def gaussian(x, a, x0, sigma):
    return a*np.exp(- ((x-x0)/sigma)**2)
def five_gaussians(x, a1, x1, sigma1, a2, x2, sigma2, a3, x3, sigma3, a4, x4, sigma4, a5, x5, sigma5):
    return a1*np.exp(- ((x - x1) / sigma1)**2) + a2*np.exp(- ((x - x2) / sigma2)**2) + a3*np.exp(- ((x - x3) / sigma3)**2) + a4*np.exp(- ((x - x4) / sigma4)**2) + a5*np.exp(- ((x - x5) / sigma5)**2)
def CCFfit(t, r0, ep, vp, lp, td):
    r = r0 * np.exp(-(ep - vp*t)**2/(lp)**2 - (t/td)**2)
    return r


def plot_hop(shotid, fbandid):
    
    hop, freq, tcycle = read_w7x.hoppingprogram(shotid, fbandid)
    plt.figure("one cycle of frequency hopping")
    plt.clf()
    plt.step(hop, freq)
    plt.xlabel("Relative time [ms]")
    plt.ylabel("frequency [GHz]")
    plt.show()
    
    return


def qmc_ccf(shotid, tstart, tend, fbandid, Nfit, fco=[1.e3, 100.e3], order_filt = 2, dlagfit = 1.e-6):

    fbands_dic = {1:'Ka', 2:'U'}
    antennas_dic = {1:'B', 2:'C', 3:'D', 4:'E'}
    antenna_combi = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    
    pitch_angle = np.radians(13)  # ?
    # antenna 44.1 mm * 34.8 mm
    dtorbc = 0.e-3
    dtorbd = 34.8e-3
    dtorbe = 34.8e-3
    dtorcd = 34.8e-3
    dtorce = 34.8e-3
    dtorde = 0.e-3
    dtors = np.array([dtorbc, dtorbd, dtorbe, dtorcd, dtorce, dtorde])
    
    dpolbc = 88.2e-3
    dpolbd = 22.05e-3
    dpolbe = 66.15e-3
    dpolcd = 66.15e-3
    dpolce = 22.05e-3
    dpolde = 44.1e-3
    dpols = np.array([dpolbc, dpolbd, dpolbe, dpolcd, dpolce, dpolde])
    
    eps = (dpols + dtors*np.tan(pitch_angle))*np.cos(pitch_angle)
    
    
    
    tt, iqdatB, iqdatC = read_w7x.iq_qmc_combi(shotid, tstart, tend, fbandid, (1, 2))
    tt, iqdatD, iqdatE = read_w7x.iq_qmc_combi(shotid, tstart, tend, fbandid, (3, 4))
    
    hop_s, freq_ghz = read_w7x.get_hoppingprogram_intime(shotid, tstart, tend, fbandid)
    
    tlist = ((hop_s + np.roll(hop_s, 1))/2)[1:]
    
    acs_tlist = [0] * len(tlist)
    ccs_tlist = [0] * len(tlist)
    delays_tlist = [0] * len(tlist)
    peaks_tlist = [0] * len(tlist)
    r0_tlist = [0] * len(tlist)
    td_tlist = [0] * len(tlist)
    vps_tlist = [0] * len(tlist)
    lps_tlist = [0] * len(tlist)
    
    for i, tat in enumerate(tlist):
        
        tt_plat = read_w7x.get_plateau_data(tt, tt, hop_s, tat)
        iqdatB_plat = read_w7x.get_plateau_data(iqdatB, tt, hop_s, tat)
        iqdatC_plat = read_w7x.get_plateau_data(iqdatC, tt, hop_s, tat)
        iqdatD_plat = read_w7x.get_plateau_data(iqdatD, tt, hop_s, tat)
        iqdatE_plat = read_w7x.get_plateau_data(iqdatE, tt, hop_s, tat)
        
        iqdatB_plat = read_w7x.normalize_pcr(iqdatB_plat)
        iqdatC_plat = read_w7x.normalize_pcr(iqdatC_plat)
        iqdatD_plat = read_w7x.normalize_pcr(iqdatD_plat)
        iqdatE_plat = read_w7x.normalize_pcr(iqdatE_plat)
        
        iqdats_plat = {1: iqdatB_plat, 2:iqdatC_plat, 3:iqdatD_plat, 4:iqdatE_plat}
        
        dt = tt[1] - tt[0]
                
        iqdatB_plat_bpf_1 = calc.filter_butterworth(iqdatB_plat, 1./dt, fco, "band", order_filt)
        iqdatC_plat_bpf_1 = calc.filter_butterworth(iqdatC_plat, 1./dt, fco, "band", order_filt)
        iqdatD_plat_bpf_1 = calc.filter_butterworth(iqdatD_plat, 1./dt, fco, "band", order_filt)
        iqdatE_plat_bpf_1 = calc.filter_butterworth(iqdatE_plat, 1./dt, fco, "band", order_filt)
        
        iqdats = {1: iqdatB_plat_bpf_1, 2:iqdatC_plat_bpf_1, 3:iqdatD_plat_bpf_1, 4:iqdatE_plat_bpf_1}
        
        acs = [0]*len(iqdats)
        
        for k in iqdats.keys():
            ac = calc.cross_correlation_analysis_v2(iqdats[k], iqdats[k], dt)
            acs[k-1] = ac
            
        acs_tlist[i] = acs
        
        ccs = [0]*len(antenna_combi)
        delays_1 = [0] * len(antenna_combi)
        peaks_1 = [0] * len(antenna_combi)
        
        for j, combi in enumerate(antenna_combi):
            
            antennaid1 = combi[0]
            antennaid2 = combi[1]
        
            cc = calc.cross_correlation_analysis_v2(iqdats[antennaid1], iqdats[antennaid2], dt)
            ccs[j] = cc
            
            delays_1[j] = cc.delay * 1e6
            peaks_1[j] = cc.corrmax        
        
        ccs_tlist[i] = ccs
        delays_tlist[i] = delays_1
        peaks_tlist[i] = peaks_1
        
        inip = [1, 4]
        func = lambda x, a, b: gaussian(x, a, 0, b)
        popt1, pcov1 = curve_fit(func, delays_1, peaks_1, p0=inip)
        
        r01 = popt1[0]
        td1 = popt1[1]
        
        r0_tlist[i] = r01
        td_tlist[i] = td1
        
        vps = [0] * len(antenna_combi)
        lps = [0] * len(antenna_combi)
        
        for j, combi in enumerate(antenna_combi):
        
            delayidx = np.argmin(np.abs(ccs[j].lags - ccs[j].delay))
            idxs = [delayidx - int(dlagfit/dt), delayidx + int(dlagfit/dt)]
            
            func = lambda t, vp, lp: CCFfit(t, r01, eps[j], vp, lp, td1*1e-6)
            popt, pcov = curve_fit(func, ccs[j].lags[idxs[0]: idxs[1]], ccs[j].ccf_amp[idxs[0]: idxs[1]])
            vps[j], lps[j] = popt
        
        vps_tlist[i] = vps
        lps_tlist[i] = lps
                
    o = struct()
    o.shotid = shotid
    o.tstart = tstart
    o.tend = tend
    o.fbandid = fbandid
    o.fco = fco_1
    o.orderfilt = order_filt
    o.dlagfit = dlagfit
    o.tdat = tt
    o.dt = dt
    o.iqB = iqdatB
    o.iqC = iqdatC
    o.iqD = iqdatD
    o.iqE = iqdatE
    o.hop = hop_s
    o.freq = freq_ghz
    o.tlist = tlist
    o.acf = acs_tlist
    o.ccf = ccs_tlist
    o.delay = delays_tlist
    o.peak = peaks_tlist
    o.rho0 = r0_tlist
    o.vp = vps_tlist
    o.lp = lps_tlist
    
    
    return o
    
        
        
        
        # acb2 = calc.cross_correlation_analysis_v2(iqdatB_plat_bpf_2, iqdatB_plat_bpf_2, dt)
        # acc2 = calc.cross_correlation_analysis_v2(iqdatC_plat_bpf_2, iqdatC_plat_bpf_2, dt)
        # acd2 = calc.cross_correlation_analysis_v2(iqdatD_plat_bpf_2, iqdatD_plat_bpf_2, dt)
        # ace2 = calc.cross_correlation_analysis_v2(iqdatE_plat_bpf_2, iqdatE_plat_bpf_2, dt)
        
        # ccbc2 = calc.cross_correlation_analysis_v2(iqdatB_plat_bpf_2, iqdatC_plat_bpf_2, dt)
        # ccbd2 = calc.cross_correlation_analysis_v2(iqdatB_plat_bpf_2, iqdatD_plat_bpf_2, dt)
        # ccbe2 = calc.cross_correlation_analysis_v2(iqdatB_plat_bpf_2, iqdatE_plat_bpf_2, dt)
        # cccd2 = calc.cross_correlation_analysis_v2(iqdatC_plat_bpf_2, iqdatD_plat_bpf_2, dt)
        # ccce2 = calc.cross_correlation_analysis_v2(iqdatC_plat_bpf_2, iqdatE_plat_bpf_2, dt)
        # ccde2 = calc.cross_correlation_analysis_v2(iqdatD_plat_bpf_2, iqdatE_plat_bpf_2, dt)
        

        
        
        # delays_2 = [ccbc2.delay * 1e6, ccbd2.delay * 1e6, - ccbe2.delay * 1e6, cccd2.delay * 1e6, - ccce2.delay * 1e6, - ccde2.delay * 1e6]
        # peaks_2 = [ccbc2.corrmax, ccbd2.corrmax, ccbe2.corrmax, cccd2.corrmax, ccce2.corrmax, ccde2.corrmax]


        
        
        # popt2, pcov2 = curve_fit(func, delays_2, peaks_2, p0=inip)
        
        
        # r02 = popt2[0]
        # td2 = popt2[1]
        
        # epbc = (dpolbc + dtorbc*np.tan(pitch_angle))*np.cos(pitch_angle)
        # epbd = (dpolbd + dtorbd*np.tan(pitch_angle))*np.cos(pitch_angle)
        
        
        
        
        
        
        
        
        # iqdatB_plat_bpf_0 = calc.filter_butterworth(iqdatB_plat, 1./dt, fco_0, "band", order_filt)
        # iqdatC_plat_bpf_0 = calc.filter_butterworth(iqdatC_plat, 1./dt, fco_0, "band", order_filt)
        # iqdatD_plat_bpf_0 = calc.filter_butterworth(iqdatD_plat, 1./dt, fco_0, "band", order_filt)
        # iqdatE_plat_bpf_0 = calc.filter_butterworth(iqdatE_plat, 1./dt, fco_0, "band", order_filt)
        
                
        # acb0 = calc.cross_correlation_analysis_v2(iqdatB_plat_bpf_0, iqdatB_plat_bpf_0, dt)
        # acc0 = calc.cross_correlation_analysis_v2(iqdatC_plat_bpf_0, iqdatC_plat_bpf_0, dt)
        # acd0 = calc.cross_correlation_analysis_v2(iqdatD_plat_bpf_0, iqdatD_plat_bpf_0, dt)
        # ace0 = calc.cross_correlation_analysis_v2(iqdatE_plat_bpf_0, iqdatE_plat_bpf_0, dt)
        
        # ccbc0 = calc.cross_correlation_analysis_v2(iqdatB_plat_bpf_0, iqdatC_plat_bpf_0, dt)
        # ccbd0 = calc.cross_correlation_analysis_v2(iqdatB_plat_bpf_0, iqdatD_plat_bpf_0, dt)
        # ccbe0 = calc.cross_correlation_analysis_v2(iqdatB_plat_bpf_0, iqdatE_plat_bpf_0, dt)
        # cccd0 = calc.cross_correlation_analysis_v2(iqdatC_plat_bpf_0, iqdatD_plat_bpf_0, dt)
        # ccce0 = calc.cross_correlation_analysis_v2(iqdatC_plat_bpf_0, iqdatE_plat_bpf_0, dt)
        # ccde0 = calc.cross_correlation_analysis_v2(iqdatD_plat_bpf_0, iqdatE_plat_bpf_0, dt)
        
        # delays_0 = [ccbc0.delay * 1e6, ccbd0.delay * 1e6, - ccbe0.delay * 1e6, cccd0.delay * 1e6, - ccce0.delay * 1e6, - ccde0.delay * 1e6]
        # peaks_0 = [ccbc0.corrmax, ccbd0.corrmax, ccbe0.corrmax, cccd0.corrmax, ccce0.corrmax, ccde0.corrmax]
        
        # inip = [1, 4]
        # func = lambda x, a, b: gaussian(x, a, 0, b)
        # popt0, pcov0 = curve_fit(func, delays_0, peaks_0, p0=inip)
        
        
        

        
        # iqdatB_plat_bpf_2 = calc.filter_butterworth(iqdatB_plat, 1./dt, fco_2, "band", order_filt)
        # iqdatC_plat_bpf_2 = calc.filter_butterworth(iqdatC_plat, 1./dt, fco_2, "band", order_filt)
        # iqdatD_plat_bpf_2 = calc.filter_butterworth(iqdatD_plat, 1./dt, fco_2, "band", order_filt)
        # iqdatE_plat_bpf_2 = calc.filter_butterworth(iqdatE_plat, 1./dt, fco_2, "band", order_filt)