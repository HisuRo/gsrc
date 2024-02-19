# imports and configuration
import w7xarchive                # the primary tool in this workshop
import matplotlib.pyplot as plt  # for plotting
import time                      # for timing commands in this workbook
import os                        # file directory manipulations
import sys                       # system i/o
import numpy as np               # some numerics
import logging                   # access w7xarchive logging infrastructure

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('w7xarchive')
logger.setLevel(logging.WARNING)

_figsize=(20,10)

fbands_dic = {1:'Ka', 2:'U'}
antennas_dic = {1:'B', 2:'C', 3:'D', 4:'E'}


def utctime_start_end(shotid, tstart, tend):
    
    ts_shot_t1 = w7xarchive.get_program_t1(shotid) # there is also a get_program_t0 and a get_program_triggerts for an arbitrary trigger

    utc_start = int(1e9*tstart + ts_shot_t1)
    utc_end = int(1e9*tend + ts_shot_t1)
    
    return utc_start, utc_end, ts_shot_t1


def mapToArray(parlogIn):
    parlogOut = parlogIn.copy()
    for kw, val in parlogOut.items():
        if type(val) == type({}):
            _list = []
            for ii in range(len(val)):
                _list.append( val["[%i]"%(ii,)] )
            parlogOut[kw] = _list
    return parlogOut


def iq_qmc(shotid, tstart, tend, fbandid, antennaid):
    
    utc_start, utc_end, ts_shot_t1 = utctime_start_end(shotid, tstart, tend)
    
    tmp = np.reshape(np.arange(len(antennas_dic)*len(fbands_dic)), (len(antennas_dic), len(fbands_dic)))
    digitizerch_i = tmp[antennaid - 1, fbandid - 1] * 2
    digitizerch_q = tmp[antennaid - 1, fbandid - 1] * 2 + 1
    address_i = f"ArchiveDB/codac/W7X/ControlStation.70601/ACQ0-1_DATASTREAM/{digitizerch_i}/BNC{fbandid:02}_COS_{antennaid}/scaled"
    print(address_i)
    address_q = f"ArchiveDB/codac/W7X/ControlStation.70601/ACQ0-1_DATASTREAM/{digitizerch_q}/BNC{fbandid:02}_SIN_{antennaid}/scaled"
    print(address_q)
    
    utc_i, idat = w7xarchive.get_signal(address_i, utc_start, utc_end)
    utc_q, qdat = w7xarchive.get_signal(address_q, utc_start, utc_end)
    
    tt = 1e-9*(utc_i - ts_shot_t1)
    qdat = np.interp(utc_i, utc_q, qdat)
    
    return tt, idat, qdat


def interpolate_to_coaserone(x1, y1, x2, y2):
    dx1 = x1[1] - x1[0]
    dx2 = x2[1] - x2[0]
    
    if dx1 >= dx2:
        x_new = x1
        y1_new = y1
        y2_new = np.interp(x_new, x2, y2)
    else:
        x_new = x2
        y1_new = np.interp(x_new, x1, y1)
        y2_new = y2
    
    return x_new, y1_new, y2_new


def iq_qmc_combi(shotid, tstart, tend, fbandid, antennaids):
    
    if len(antennaids) != 2:
        print('length of "antennids" should be 2')
    antennaid_1 = antennaids[0]
    antennaid_2 = antennaids[1]
    
    tt1, idat1, qdat1 = iq_qmc(shotid, tstart, tend, fbandid, antennaid_1)
    tt2, idat2, qdat2 = iq_qmc(shotid, tstart, tend, fbandid, antennaid_2)
    iqdat1 = idat1 + 1.j * qdat1
    iqdat2 = idat2 + 1.j * qdat2
    
    tt, iqdat1, iqdat2 = interpolate_to_coaserone(tt1, iqdat1, tt2, iqdat2)
    
    return tt, iqdat1, iqdat2


def parlog_qmc(shotid, fbandid):

    ts_shot_t0, ts_shot_tlast = w7xarchive.get_program_from_to(shotid)
    address_parlog = f"ArchiveDB/codac/W7X/ControlStation.70601/BNC{fbandid:02}-0_PARLOG"
    print(address_parlog)
    parlog = w7xarchive.get_parameters_box(address_parlog, ts_shot_t0, ts_shot_tlast)
    
    return parlog


def hoppingprogram(shotid, fbandid):
    
    parlog = parlog_qmc(shotid, fbandid)
    plog = mapToArray(parlog['values'][0])
       
    # ==== Specific settings for each cycle: sent to synthesizers ==== #
    delay_s = np.asarray(plog['list_delay'])   # 0.010 --> should be constant 
    
    delay_s[0] = 0.005  # ?
    
    dwell_s = np.asarray(plog['list_dwell'])
    hop_ms = 1e3*np.cumsum(np.insert(dwell_s, 0, delay_s[0]))     # same as time_ms (maybe without delay)
    freq_ghz = 1e-9*np.asarray(plog['list_freq'])  # synthesizer frequency in GHz (Ka-band uses a doubler, U-band a quadrupler)
    freq_ghz = np.insert(freq_ghz, 0, np.nan)
    if fbandid == 1:
        freq_ghz *= 2.0   # ka-band doubler
        
        # band_name = 'bnc01'
    elif fbandid == 2:
        freq_ghz *= 4.0   # U-band doubler
        
        # band_name = 'bnc02'

    # time_ms = np.asarray(plog[band_name]['time_ms'])     # requested hopping times (relative?)
    # power_dbm = np.asarray(plog[band_name]['power_dbm']) # requested synthesizer power output (+20dBm)
    # dwell_ms = np.asarray(plog[band_name]['dwell_ms'])   # requested list of hop durations: same as "dwell"
    # freq_ghz = np.asarray(plog[band_name]['freq_ghz'])   # requested list of hop frequencies: same as "freq"
    # Nscans = np.asarray(plog[band_name]['number_of_sweeps']) # if 0/-1? sweep forever, otherwise int(Number of hopping cycles) 
    # "BNC1['Duration']" same as tcycle
    
    hop_ms = np.round(hop_ms).astype(np.int64)
    tcycle_ms = hop_ms[-1]
    # Nsteps = len(freq)         # number of hops / cycle
    
    
    return hop_ms, freq_ghz, tcycle_ms

import json
import urllib

def readjson(url):
    """ returns a dictionary with keys 'dimensions' and 'values' """
    return json.loads(urllib.request.urlopen(url).read())

def get_json(url, shotid=None, utcstart=None, utcend=None, use_w7xapi=False, **_config):
    # url = "ArchiveDB/codac/W7X/ControlStation.70601/CONTROL-0_PARLOG/tte/trigger/channel"

    if shotid is None:
        if use_w7xapi:
            return w7xarchive.get_signal(url, time_from=utcstart, time_to=utcend, **_config)
        return readjson(f'http://archive-webapi.ipp-hgw.mpg.de/{url}/_signal.json?from={utcstart}&upto={utcend}')

    return readjson(f'http://archive-webapi.ipp-hgw.mpg.de/{url}/_signal.json?XP={shotid[3:]}')



def get_trigger_qmc(shotid):
    # temporary "cheat"
    # get the first data point from the digitizer for this shot.  
    # The microwave synthesizers begin sweeping at the same time that we 
    # trigger the digitizer, because they use the same trigger.
    address = "ArchiveDB/codac/W7X/ControlStation.70601/ACQ0-1_DATASTREAM/0/BNC01_COS_1/scaled"
    
    # address = "ArchiveDB/codac/W7X/ControlStation.70601/CONTROL-0_PARLOG/tte/trigger/channel"
    # ts_shot_t0, ts_shot_tlast = w7xarchive.get_program_from_to(shotid)
    ts_shot_t1_ns = w7xarchive.get_program_t1(shotid)
    
    before_plasma = w7xarchive.get_signal(address, int(ts_shot_t1_ns - 2e9), ts_shot_t1_ns)
    # address = "ArchiveDB/codac/W7X/ControlStation.70601/CONTROL-0_PARLOG"
    # ts_shot_t0, ts_shot_tlast = w7xarchive.get_program_from_to(shotid)
    # channel = w7xarchive.get_parameters_box(address, ts_shot_t0, ts_shot_tlast)
    time_trig_ns = before_plasma[0][0]
    
    return time_trig_ns


def extend_hoppingprogram(hop, freq, tcycle, Nrepeat):
    
    tmp = np.tile(hop, Nrepeat)
    Nwin = len(hop)
    tmp2 = np.repeat(np.arange(Nrepeat), Nwin)
    extended_hop = tmp.astype(np.int64) + tmp2.astype(np.int64) * tcycle
    extended_freq = np.tile(freq, Nrepeat)
    
    return extended_hop, extended_freq


def get_hoppingprogram_realtime_fromTrg_toLast(shotid, fbandid):
    
    hop_ms, freq_ghz, tcycle_ms = hoppingprogram(shotid, fbandid)
    time_trig_ns = get_trigger_qmc(shotid)
    
    ts_shot_t0_ns, ts_shot_tlast_ns = w7xarchive.get_program_from_to(shotid)
    tcycle_ns = int(1e6 * tcycle_ms)
    hop_ns = (1e6 * hop_ms).astype(np.int64)
    Nrepeat = (ts_shot_tlast_ns - time_trig_ns) // tcycle_ns + 1
    hop_ext_ns, freq_ext_ghz = extend_hoppingprogram(hop_ns, freq_ghz, tcycle_ns, Nrepeat)
    hop_realtime_ns = hop_ext_ns + time_trig_ns
    
    return hop_realtime_ns, freq_ext_ghz


def get_hoppingprogram_intime(shotid, tstart, tend, fbandid):
    
    hop_realtime_ns, freq_ext_ghz = get_hoppingprogram_realtime_fromTrg_toLast(shotid, fbandid)
    ts_shot_t1_ns = w7xarchive.get_program_t1(shotid) # there is also a get_program_t0 and a get_program_triggerts for an arbitrary trigger
    hop_shift_ns = hop_realtime_ns - ts_shot_t1_ns
    
    tstart_ns = int(tstart * 1e9)
    tend_ns = int(tend * 1e9)
    idx_hop_ts = np.argmin(np.abs(hop_shift_ns - tstart_ns)) - 1
    idx_hop_te = np.argmin(np.abs(hop_shift_ns - tend_ns)) + 1
    
    hop_s = 1e-9 * hop_shift_ns[idx_hop_ts: idx_hop_te + 1]
    freq_ghz = freq_ext_ghz[idx_hop_ts: idx_hop_te + 1]
    
    return hop_s, freq_ghz


def get_tstart_tend_in_plateau(hop_s, t_at_s):
    
    tidxs = np.argsort(np.abs(hop_s - t_at_s))[:2]
    tidxs = np.sort(tidxs)
    tstart, tend = hop_s[tidxs]
    
    return tstart, tend


def get_plateau_idxstart_idxend(tt, hop_s, t_at_s):
    
    ts, te = get_tstart_tend_in_plateau(hop_s, t_at_s)
    idx_ts = np.argmin(np.abs(tt - ts)) + 1
    idx_te = np.argmin(np.abs(tt - te)) - 1
    
    return idx_ts, idx_te


def get_plateau_data(xx, tt, hop_s, t_at_s):
    
    idx_ts, idx_te = get_plateau_idxstart_idxend(tt, hop_s, t_at_s)
    
    return xx[idx_ts: idx_te + 1]


def show_hoppingprogram(shotid, fbandid):
    
    hop, freq, tcycle = hoppingprogram(shotid, fbandid)
    
    plt.figure()
    plt.step(hop, freq, label='Hopping fequency')
    plt.xlabel('hop [ms]')
    plt.ylabel('freq [GHz]')
    plt.title(f'{shotid}\n'
              f'{fbands_dic[fbandid]}-band system hopping program')
    plt.show()
    
    return


def normalize_pcr(yt):  # iq signals

    y1 = np.real(yt)
    y3 = np.imag(yt)
    
    ymax_y1 = np.zeros((len(y1)//4000))
    ymax_y3 = np.zeros((len(y1)//4000))
    ymin_y1 = np.zeros((len(y1)//4000))
    ymin_y3 = np.zeros((len(y1)//4000))
    
    y1 = y1-np.mean(y1)
    y3 = y3-np.mean(y3)
    for i in range(0, len(y1)//4000):
        ymax_y1[i] = y1[i:i+4000].max()
        ymax_y3[i] = y3[i:i+4000].max()
        ymin_y1[i] = y1[i:i+4000].min()
        ymin_y3[i] = y3[i:i+4000].min()

    ymax1 = np.mean(ymax_y1)
    ymax3 = np.mean(ymax_y3)
    ymin1 = np.mean(ymin_y1)
    ymin3 = np.mean(ymin_y3)
    a = 2./(ymax1-ymin1)
    y1 = y1*a
    a = 2./(ymax3-ymin3)
    y3 = y3*a

    yt1 = y1 + 1j*y3
    
    return yt1

