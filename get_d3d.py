from nasu import gadata, calc
import numpy as np # type: ignore

def produce_virtual_IQ_reference(times_s, carrier_freq_Hz, phase=0):
	return np.cos(2 * np.pi * carrier_freq_Hz * times_s + phase) - 1.j * np.sin(2 * np.pi * carrier_freq_Hz * times_s + phase)

def produce_virtual_IQ_signal(times_s, signal, carrier_freq_Hz, ref_phase=0, downsampling_factor=1):

	reference = produce_virtual_IQ_reference(times_s=times_s, carrier_freq_Hz=carrier_freq_Hz, phase=ref_phase)

	IQ_signal = signal * reference
	if downsampling_factor != 1:
		IQ_signal = calc.signal.decimate(IQ_signal, q=downsampling_factor, ftype="fir")

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
	
	def iirfilter(self, cutoffFreq, bandtype="lowpass", order=4, filtertype="butter"):
		self.filt = calc.struct()
		self.filt.t_s = self.t_s
		self.filt.d = calc.iirfilter(self.d, self.Fs, cutoffFreq=cutoffFreq, bandtype=bandtype, order=order, filtertype=filtertype)
		self.filt.Fs = self.Fs
		return self.filt
	
	def firfilter(self, cutoffFreq, bandtype="lowpass", order=1000, window="hamming"): 
		self.filt = calc.struct()
		self.filt.t_s, self.filt.d = calc.firfilter_zerodelay(self.t_s, self.d, self.Fs, cutoffFreq=cutoffFreq, bandtype=bandtype, order=order, window=window)
		self.filt.Fs = self.Fs
		return self.filt
	
	def decimate(self, downsampling_factor=10):
		self.dec = calc.struct()
		self.dec.t_s = self.t_s[::downsampling_factor]
		self.dec.d = calc.signal.decimate(self.d, q=downsampling_factor, ftype='fir')
		self.dec.Fs = self.Fs / downsampling_factor
		return self.dec

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
		super().__init__(timetrace_instance.virt.t_s, timetrace_instance.virt.d, timetrace_instance.virt.Fs)

class virtIQamp(signal):
	def __init__(self, timetrace_instance):
		super().__init__(timetrace_instance.virt.t_s, timetrace_instance.virt.amp, timetrace_instance.virt.Fs)

class virtIQphase(signal):
	def __init__(self, timetrace_instance):
		super().__init__(timetrace_instance.virt.t_s, timetrace_instance.virt.phase, timetrace_instance.virt.Fs)

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
