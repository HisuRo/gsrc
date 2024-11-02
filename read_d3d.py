from nasu import gadata
import numpy as np # type: ignore

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
	
		return



