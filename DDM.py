import numpy as np
import matplotlib.pyplot as plt

class DDM(object):

	def __init__(self, num_walks, DDM_sigma, ndt, name, dt):

		self.name = name
		self.num_walks = num_walks
		self.DDM_sigma = DDM_sigma
		self.ndt = ndt 	#Non Decision Time (ms)
		self.curr_timestamp = 0		#current timestamp (ms)
		self.dt = dt
		self.gaze_fs = 1/dt

	def DDM_simulate(self, starting_point, value):
		E = np.zeros(self.num_walks)
		E[0] = starting_point
		for i in range(1,self.num_walks):
			if self.curr_timestamp <= self.ndt:
				E[i] = 0
				self.curr_timestamp += (1/self.gaze_fs)*1000
			else:
				E[i] = E[i-1] + value*self.dt + np.sqrt(self.DDM_sigma*self.dt)*np.random.normal(0, 1, 1)

		return E
