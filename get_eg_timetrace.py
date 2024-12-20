from nasu import myEgdb, calc
from nasu.timetrace import *
import numpy as np # type: ignore
import os

pathCalib = os.path.join("C:\\pythonProject\\nasu\\calib_table.csv")

# ================================================================================================================================

class timetrace():

	def __init__(self,sn,sub,tstart,tend,diagname,name,dim=0,other_idxs=[0]):
		
		# Save object values
		self.sn = sn
		self.sub = sub
		self.tstart = tstart
		self.tend = tend
		self.diagname = diagname

		# Retrieve data
		self.eg = myEgdb.LoadEG(diagname=diagname, sn=sn, sub=sub)
		self.t_s = self.eg.dims(dim=dim)
		self.d = self.eg.trace_of(name=name, dim=dim, other_idxs=other_idxs)
		self.Fs = calc.samplingrate_from_timedat(self.t_s)
		
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

		return self.virt

