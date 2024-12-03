import numpy as np  # type: ignore

def same_data(d1, d2, rtol=1e-5):

	if np.allclose(d1, d2, atol=1e-30, rtol=rtol): 
		return True
	else:
		return False
