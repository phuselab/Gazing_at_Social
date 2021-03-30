import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def sample_OU_trajectory(numOfpoints, alpha, mu, startxy, dt, sigma):

	Sigma1 = np.array([[1, np.exp(-alpha * dt / 2)],[np.exp(-alpha * dt / 2), 1]])
	Sigma2 = Sigma1

	x = np.zeros(numOfpoints)
	y = np.zeros(numOfpoints)
	mu1 = np.zeros(numOfpoints)
	mu2 = np.zeros(numOfpoints)
	x[0] = startxy[1]
	y[0] = startxy[0]
	mu1[0] = mu[1]
	mu2[0] = mu[0]
	for i in range(numOfpoints-1):
		r1 = np.random.randn() * sigma
		r2 = np.random.randn() * sigma

		x[i+1] = mu1[i]+(x[i]-mu1[i])*(Sigma1[0,1]/Sigma1[0,0])+np.sqrt(Sigma1[1,1])-(((Sigma1[0,1]**2)/Sigma1[0,0]))*r1
		y[i+1] = mu2[i]+(y[i]-mu2[i])*(Sigma2[0,1]/Sigma2[0,0])+np.sqrt(Sigma2[1,1])-(((Sigma2[0,1]**2)/Sigma2[0,0]))*r2
		mu1[i+1] = mu1[i]
		mu2[i+1] = mu2[i]

	return np.stack([y,x], axis=1)


class GazeSampler(object):
 
	def __init__(self):
        		
		self.allFOA = []
		self.sampled_gaze = []
		self.prev_patch = 2

	def sampleDDM(self, iframe, patches, FOAsize, winner):
		patches_to_consider = patches
		self.FOAsize = FOAsize

		if winner is not None:
			#New winner!!!
			winner_name = patches_to_consider[winner].label
			chosenPatch = winner
			self.prev_patch = chosenPatch

			m = patches_to_consider[chosenPatch].center
			s = np.flip(np.asarray(patches_to_consider[chosenPatch].axes)) * 4
			S = np.eye(2) * s
			self.newFOA = np.random.multivariate_normal(m, S)

			t_patch = np.round(np.max([s[0], s[1]]))*2 #setting the length of the feeding proportional to major axis
			dtp = 1/t_patch
		
			mup = self.newFOA

			if iframe == 0:
				#winner in the first frame --> define variance for CB exploration
				startxy = self.newFOA
				alphap = np.max([patches_to_consider[chosenPatch].axes[0], patches_to_consider[chosenPatch].axes[1]])*10
			else:
				#new winner --> define variance for FLIGHT
				startxy = self.arriving_xy
				prevFOA = self.allFOA[-1]
				alphap = euclidean(self.newFOA, prevFOA)/5.

			self.chosenPatch = chosenPatch
			self.patches_to_consider = patches_to_consider

			sigmap = 15

		else:
			#No new winners --> variance for local exploration
			startxy = self.arriving_xy
			mup = self.prev_mup
			alphap = np.max([self.patches_to_consider[self.chosenPatch].axes[0], self.patches_to_consider[self.chosenPatch].axes[1]])*10

			t_patch = np.round(np.max([self.patches_to_consider[self.chosenPatch].axes[0], self.patches_to_consider[self.chosenPatch].axes[1]]))*2 #setting the length of the feeding proportional to major axis
			dtp = 1/t_patch
			sigmap = 1

		walk = sample_OU_trajectory(numOfpoints=10, alpha=alphap, mu=mup, startxy=startxy, dt=1/alphap, sigma=sigmap)

		self.arriving_xy = np.round(walk[-1,:])
		self.prev_mup = mup
		self.prev_alphap = alphap

		self.allFOA.append(self.arriving_xy)
		self.sampled_gaze.append(walk)