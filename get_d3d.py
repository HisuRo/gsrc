from nasu import gadata, calc
import numpy as np # type: ignore

def produce_virtual_IQ_reference(times_s, carrier_freq_Hz, phase=0):
	return np.cos(2 * np.pi * carrier_freq_Hz * times_s + phase) - 1.j * np.sin(2 * np.pi * carrier_freq_Hz * times_s + phase)

def produce_virtual_IQ_signal(times_s, signal, carrier_freq_Hz, ref_phase=0):

	dts = np.diff(times_s)
	if np.allclose(dts, dts[0]):
		Fs = 1. / dts[0]
	else:
		raise ValueError("Time data not equally spaced")

	reference = produce_virtual_IQ_reference(times_s=times_s, carrier_freq_Hz=carrier_freq_Hz, phase=ref_phase)

	IQ_signal = signal * reference
	IQ_signal = calc.lowpass(x=IQ_signal, samplerate=Fs, fp=0.5*carrier_freq_Hz, fs=0.75*carrier_freq_Hz, gstop=30)

	return IQ_signal

# =================================================================================================================================

class timetrace_multidomains:

	def __init__(self,pointname,shot,idx_startdomain,N_domain,tree=None,connection=None,nomds=False):
		
		# Save object values
		self.pointname = pointname
		self.shot = shot
		self.t = np.array([])
		self.d = np.array([])
		self.connection = connection

		# Retrieve data repeatedly and connect them
		domains = np.arange(idx_startdomain, idx_startdomain + N_domain)

		for domain in domains:
			signalname = f"{self.pointname}_{domain}"
			ga = gadata.gadata(signal=signalname, shot=self.shot, tree=tree, connection=self.connection, nomds=nomds)
			self.t = np.append(self.t, ga.xdata)
			self.d = np.append(self.d, ga.zdata)

		self.amp = calc.amplitude(self.d)
	
		return
	
	def produce_virtual_IQ_signal(self, carrier_freq_Hz, ref_phase=0):

		self.iq = produce_virtual_IQ_signal(times_s=self.t * 1e-3, signal=self.d, carrier_freq_Hz=carrier_freq_Hz, ref_phase=ref_phase)
		self.iqamp = calc.amplitude(self.iq)
		self.iqphase = calc.iqphase(self.iq)
