from nasu import calc
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
