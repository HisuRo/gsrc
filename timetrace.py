from nasu import calc
import numpy as np # type: ignore
from scipy.signal import find_peaks # type: ignore


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



	def amplitude(self):
		t_s = self.t_s
		d = calc.amplitude(self.d)
		Fs = self.Fs
		self.amp = signal(t_s, d, Fs)

	def iqphase(self):
		t_s = self.t_s
		d = calc.phase(self.d)
		Fs = self.Fs
		self.phase = signal(t_s, d, Fs)

	def pulsepair(self, ovr=0.5, Fs=10e3):
		NSamp = int(ovr * self.Fs / Fs)
		o = calc.pulsepair(self.t_s, self.d, Nsample=NSamp, ovr=ovr)
		t_s = o.t
		d = o.fd
		self.pp = signal(t_s, d, Fs)

		return self.pp

	def iirfilter(self, cutoffFreq, bandtype="lowpass", order=4, filtertype="butter"):
		t_s = self.t_s
		d = calc.iirfilter(self.d, self.Fs, cutoffFreq=cutoffFreq, bandtype=bandtype, order=order, filtertype=filtertype)
		Fs = self.Fs
		self.filt = signal(t_s, d, Fs)
		return self.filt
	
	def firfilter(self, cutoffFreq, bandtype="lowpass", order=1000, window="hamming"): 
		t_s, d = calc.firfilter_zerodelay(self.t_s, self.d, self.Fs, cutoffFreq=cutoffFreq, bandtype=bandtype, order=order, window=window)
		Fs = self.Fs
		self.filt = signal(t_s, d, Fs)
		return self.filt
	
	def decimate(self, downsampling_factor=10):
		t_s = self.t_s[::downsampling_factor]
		d = calc.signal.decimate(self.d, q=downsampling_factor, ftype='fir')
		Fs = self.Fs / downsampling_factor
		self.dec = signal(t_s, d, Fs)
		return self.dec



	def detect_event(self, rate_threshold, t_process_width, type="rise"):
		# type = "rise", "drop"

		self.event = calc.struct()

		self.rate = np.diff(self.d) / np.diff(self.t_s)
		self.inverted_rate = - self.rate
		if type == "rise":
			idx_event, _ = find_peaks(self.rate, height=rate_threshold, prominence=rate_threshold, distance=3)
		elif type == "drop":
			idx_event, _ = find_peaks(self.inverted_rate, height=rate_threshold, prominence=rate_threshold, distance=3)
		else:
			raise Exception("wrong type name")
		self.event.t_s = self.t_s[1:][idx_event]

		process_width = int(self.Fs * t_process_width)
		idxs_aroundevent = np.tile(idx_event.reshape(idx_event.size, 1), process_width) + np.arange(process_width) - process_width // 2
		d_process = self.d[idxs_aroundevent]
		self.event.p2p = d_process.max(axis=-1) - d_process.min(axis=-1)
		self.event.center = (d_process.max(axis=-1) + d_process.min(axis=-1)) / 2
		self.event.rel_p2p = self.event.p2p / self.event.center

		return self.event

# =================================================================================================================================

class twin_signals():

	def __init__(self, t1_s, t2_s, d1, d2, Fs1, Fs2):

		if Fs1 < Fs2:
			raise Exception("Fs1 must be = or > Fs2")
		self.t1_s = t1_s
		self.t2_s = t2_s
		self.d1 = d1
		self.d2 = d2
		self.Fs1 = Fs1
		self.Fs2 = Fs2

		# linear interpolation
		self.intp = calc.struct()
		self.intp.t_s = self.t2_s
		self.intp.d1 = calc.interpolate.interp1d(self.t1_s, self.d1, kind="linear", bounds_error=False, fill_value="extrapolate")(self.t2_s)
		self.intp.d2 = self.d2
		self.intp.Fs = self.Fs2

	def cross_spectrum(self, tstart, tend, NFFT=2**14, ovr=0.5, window="hann", detrend="constant", unwrap_phase=False):
		self.cs = calc.cross_spectrum(self.intp.t_s, self.intp.d1, self.intp.d2, self.intp.Fs, tstart, tend, NFFT=NFFT, ovr=ovr, window=window, detrend=detrend, unwrap_phase=unwrap_phase)
		return self.cs
