from nasu import gadata, calc
from nasu.timetrace import *
import numpy as np # type: ignore

# ================================================================================================================================

class timetrace():

	def __init__(self,pointname,shot,tree=None,connection=None,nomds=False):
		
		# Save object values
		self.pointname = pointname
		self.shot = shot
		self.connection = connection

		# Retrieve data
		ga = gadata.gadata(signal=self.pointname, shot=self.shot, tree=tree, connection=self.connection, nomds=nomds)
		self.t_ms = np.array(ga.xdata)
		self.d = np.array(ga.zdata)
		self.t_s = self.t_ms * 1e-3
		self.Fs = calc.samplingrate_from_timedat(self.t_s)
		
		# calculate amplitude
		self.amp = calc.amplitude(self.d)

		self.raw = raw(self)
		self.amplitude = amplitude(self)

	def produce_virtual_IQ_signal(self, carrier_freq_Hz, downsampling_factor=1, ref_phase=0):

		self.virt = calc.struct()
		self.virt.t_ms = self.t_ms[::downsampling_factor]
		self.virt.t_s = self.t_s[::downsampling_factor]
		self.virt.Fs = self.Fs / downsampling_factor
		self.virt.d = produce_virtual_IQ_signal(times_s=self.t_s, signal=self.d, carrier_freq_Hz=carrier_freq_Hz, downsampling_factor=downsampling_factor, ref_phase=ref_phase)
		self.virt.amp = calc.amplitude(self.virt.d)
		self.virt.phase = calc.phase(self.virt.d)

		self.virtIQ = virtIQ(self)
		self.virtIQamp = virtIQamp(self)
		self.virtIQphase = virtIQphase(self)

class timetrace_multidomains(timetrace):

	def __init__(self,pointname,shot,idx_startdomain,N_domain,tree=None,connection=None,nomds=False):
		
		# Save object values
		self.pointname = pointname
		self.shot = shot
		self.t_ms = np.array([])
		self.d = np.array([])
		self.connection = connection

		# Retrieve data repeatedly and connect them
		domains = np.arange(idx_startdomain, idx_startdomain + N_domain)

		for domain in domains:
			signalname = f"{self.pointname}_{domain}"
			ga = gadata.gadata(signal=signalname, shot=self.shot, tree=tree, connection=self.connection, nomds=nomds)
			self.t_ms = np.append(self.t_ms, ga.xdata)
			self.d = np.append(self.d, ga.zdata)
		self.t_s = self.t_ms * 1e-3
		self.Fs = calc.samplingrate_from_timedat(self.t_s)
		
		# calculate amplitude
		self.amp = calc.amplitude(self.d)

		self.raw = raw(self)
		self.amplitude = amplitude(self)

class timetrace_multidomains_iq():

	def __init__(self,pointname_i,pointname_q,shot,idx_startdomain,N_domain,tree=None,connection=None,nomds=False):

		self.i_obj = timetrace_multidomains(pointname_i, shot, idx_startdomain, N_domain, tree=tree, connection=connection, nomds=nomds)
		self.q_obj = timetrace_multidomains(pointname_q, shot, idx_startdomain, N_domain, tree=tree, connection=connection, nomds=nomds)

		if np.allclose(self.i_obj.t_s, self.q_obj.t_s):
			self.t_s = self.i_obj.t_s
		else:
			raise Exception("Time data of I and Q signals are different.")
		self.d = self.i_obj.d + 1.j * self.q_obj.d
		
		self.amp = calc.amplitude(self.d)
		self.phase = calc.phase(self.d)

		self.raw = raw(self)
		self.amplitude = amplitude(self)
		self.iqphase = iqphase(self)
