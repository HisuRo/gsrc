from nasu import gadata, calc
import numpy as np # type: ignore

def produce_virtual_IQ_reference(times_s, carrier_freq_Hz, phase=0):
	return np.cos(2 * np.pi * carrier_freq_Hz * times_s + phase) - 1.j * np.sin(2 * np.pi * carrier_freq_Hz * times_s + phase)

def produce_virtual_IQ_signal(times_s, signal, carrier_freq_Hz, ref_phase=0):

	Fs = calc.samplingrate_from_timedat(times_s)

	reference = produce_virtual_IQ_reference(times_s=times_s, carrier_freq_Hz=carrier_freq_Hz, phase=ref_phase)

	IQ_signal = signal * reference
	IQ_signal = calc.lowpass(x=IQ_signal, samplerate=Fs, fp=0.5*carrier_freq_Hz, fs=0.75*carrier_freq_Hz, gstop=30)

	return IQ_signal

# =================================================================================================================================

class signal():

	def __init__(self, t_s, d, Fs):
		self.t_s = t_s
		self.d = d
		self.Fs = Fs

	def specgram(self, NFFT=2**14, ovr=0., window="hann", NEns=1, detrend="constant"):
		self.spg = calc.specgram(self.t_s, self.d, self.Fs, NFFT=NFFT, ovr=ovr, window=window, NEns=NEns, detrend=detrend)
		return self.spg

	def spectrum(self, tstart, tend, NFFT=2**14, ovr=0.5, window="hann", detrend="constant"):
		self.sp = calc.spectrum(t_s=self.t_s, d=self.d, Fs_Hz=self.Fs, tstart=tstart, tend=tend, NFFT=NFFT, ovr=ovr, window=window, detrend=detrend)
		return self.sp
	
class raw(signal):
	def __init__(self, timetrace_instance):
		super().__init__(timetrace_instance.t_s, timetrace_instance.d, timetrace_instance.Fs)

class amplitude(signal):	
	def __init__(self, timetrace_instance):
		super().__init__(timetrace_instance.t_s, timetrace_instance.amp, timetrace_instance.Fs)

class iqphase(signal):	
	def __init__(self, timetrace_instance):
		super().__init__(timetrace_instance.t_s, timetrace_instance.phase, timetrace_instance.Fs)

class virtIQ(signal):
	def __init__(self, timetrace_instance):
		super().__init__(timetrace_instance.t_s, timetrace_instance.virtiq, timetrace_instance.Fs)

class virtIQamp(signal):
	def __init__(self, timetrace_instance):
		super().__init__(timetrace_instance.t_s, timetrace_instance.virtamp, timetrace_instance.Fs)

class virtIQphase(signal):
	def __init__(self, timetrace_instance):
		super().__init__(timetrace_instance.t_s, timetrace_instance.virtphase, timetrace_instance.Fs)

# =================================================================================================================================

class twin_signals():

	def __init__(self, t_s, d1, d2, Fs):
		self.t_s = t_s
		self.d1 = d1
		self.d2 = d2
		self.Fs = Fs

	def cross_spectrum(self, tstart, tend, NFFT=2**14, ovr=0.5, window="hann", detrend="constant", unwrap_phase=False):
		self.cs = calc.cross_spectrum(self.t_s, self.d1, self.d2, self.Fs, tstart, tend, NFFT=NFFT, ovr=ovr, window=window, detrend=detrend, unwrap_phase=unwrap_phase)
		return self.cs

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

	def produce_virtual_IQ_signal(self, carrier_freq_Hz, ref_phase=0):
			
		self.virtiq = produce_virtual_IQ_signal(times_s=self.t_s, signal=self.d, carrier_freq_Hz=carrier_freq_Hz, ref_phase=ref_phase)
		self.virtamp = calc.amplitude(self.iq)
		self.virtphase = calc.phase(self.iq)

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
