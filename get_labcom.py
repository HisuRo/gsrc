from nasu import read, calc
from nasu.timetrace import *
import numpy as np # type: ignore
import os

pathCalib = os.path.join("C:\\pythonProject\\nasu\\calib_table.csv")

# ================================================================================================================================

class timetrace():

	def __init__(self,sn,subsn,tstart,tend,diagname,ch):
		
		# Save object values
		self.sn = sn
		self.subsn = subsn
		self.tstart = tstart
		self.tend = tend
		self.diagname = diagname
		self.ch = ch

		# Retrieve data
		self.t_s, self.d, self.dT, self.Fs, self.size, self.tprms, self.dprms = \
            read.LHD_et_v2(sn=self.sn, subsn=self.subsn, diagname=self.diagname, ch=self.ch, et=(self.tstart, self.tend))
		
		self.raw = signal(self.t_s, self.d, self.Fs)

	def produce_virtual_IQ_signal(self, carrier_freq_Hz, downsampling_factor=1, ref_phase=0):

		self.virt = calc.struct()
		self.virt.carrier_freq_Hz = carrier_freq_Hz
		self.virt.downsampling_factor = carrier_freq_Hz
		self.virt.ref_phase = carrier_freq_Hz
				
		self.virt.t_s = self.t_s[::downsampling_factor]
		self.virt.Fs = self.Fs / downsampling_factor
		self.virt.d = produce_virtual_IQ_signal(times_s=self.t_s, signal=self.d, carrier_freq_Hz=carrier_freq_Hz, downsampling_factor=downsampling_factor, ref_phase=ref_phase)

		self.virt.raw = signal(self.virt.t_s, self.virt.d, self.virt.Fs)
		self.virt.amp = self.virt.raw.amplitude()
		self.virt.phase = self.virt.raw.iqphase()

class timetrace_iq():
		
    def __init__(self, sn, subsn, tstart, tend, diagname, ch_i, ch_q):

        self.sn = sn
        self.subsn = subsn
        self.tstart = tstart
        self.tend = tend
        self.diagname = diagname
        self.ch_i = ch_i
        self.ch_q = ch_q

        self.i_obj = timetrace(sn=self.sn, subsn=self.subsn, tstart=self.tstart, tend=self.tend, diagname=self.diagname, ch=self.ch_i)
        self.q_obj = timetrace(sn=self.sn, subsn=self.subsn, tstart=self.tstart, tend=self.tend, diagname=self.diagname, ch=self.ch_q)

        if self.diagname == "MWRM-COMB2" and self.ch_i in np.arange(1, 16 + 1, 2):

            dictIF = {1: 40, 3: 80, 5: 120, 7: 160,
                        9: 200, 11: 240, 13: 300, 15: 340}
            self.frLO = dictIF[self.ch_i]
            calibPrms_df = read.calibPrms_df_v2(pathCalib)
            self.ampratioIQ, self.offsetI, self.offsetQ, self.phasediffIQ = calibPrms_df.loc[self.frLO]

            self.i_obj.d, self.q_obj.d = calc.calibIQComp2(self.i_obj.d, self.q_obj.d,
                                                self.ampratioIQ, self.offsetI, self.offsetQ, self.phasediffIQ)

        elif diagname == "MWRM-COMB" and self.ch_i in [1, 3, 5, 7]:
            self.q_obj.d *= -1

        if np.allclose(self.i_obj.t_s, self.q_obj.t_s):
            self.t_s = self.i_obj.t_s
            self.Fs = self.i_obj.Fs
        else:
            raise Exception("Time data of I and Q signals are different.")
        self.d = self.i_obj.d + 1.j * self.q_obj.d

        self.raw = signal(self.t_s, self.d, self.Fs)
        self.amp = self.raw.amplitude()
        self.phase = self.raw.iqphase()

