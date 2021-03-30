import numpy as np
from DDM import DDM

class race_DDM(object):
	'''
	Class implementing the multi-choice race-to-threshold model
	It is assumed that the following objects of interest are provided (pre-attentive computations) as patches:
	1) Speakers
	2) Non Speakers
	3) Center Bias (CB)
	4) Areas of local motion (Spatio-Temporal Saliency STS)
	'''

	def __init__(self, prior_values, patches, diag_size, winner, fps, threshold=1.7, ndt=60, noise=2, kappa=15, phi=0.2, eta=5):

		self.prior_values = {'Speaker': prior_values[0], 'non_Speaker': prior_values[1], 'CB': prior_values[2], 'STS': prior_values[3]}
		self.num_racers = len(patches)
		self.num_walks = 10			# number of steps to take for each video frame computation. A value of 10 means that the sampling rate of the decision making is 10 times finer than the video frame rate
		self.DDM_sigma = noise		
		self.starting_points = np.zeros(self.num_racers)
		self.patches = patches
		self.racers = []
		self.realizations = []
		self.diag_size = diag_size
		self.winner = winner									#the racer of the current patch
		self.k = kappa											#kappa, to weight saccades lenght
		self.fps = fps											#Video frames per second
		self.gaze_fs = fps*self.num_walks
		self.dt = 1/self.gaze_fs								#DDM racer dt
		self.threshold = threshold
		self.ndt = ndt 											#Non Decision Time (ms)
		self.patch_residence_time = 0.
		self.is_new_patch = True
		self.phi = phi
		self.eta = eta

		for p in patches:
			self.racers.append(DDM(self.num_walks, self.DDM_sigma, self.ndt, p.label, self.dt))
			self.realizations.append(np.array([]))

	def compute_current_RDVs(self, feed_forward_inhibit=False):
		#Computes the Relative Decision Value for the racers
		#if feed_forward_inhibit==True feed-forward inhibition between racers is implemented
		#if feed_forward_inhibit==False (default) every racer is independent from each other
		RDVs = []
		for i in range(self.num_racers):
			RDVs.append(np.zeros(self.num_walks))

		for i in range(self.num_racers):
			if feed_forward_inhibit:
				for t in range(self.num_walks):
					curr_simul_t = self.DDM_simul[i][t]
					other_simul_t = [s[t] for j,s in enumerate(self.DDM_simul) if j!=i]
					RDVs[i][t] = curr_simul_t - max(other_simul_t)
			else:
				RDVs[i] = self.DDM_simul[i]
		
		return RDVs

	def check_event_occurrence(self, RDVs):
		#Check if any racer has reached the threshold
		win = np.zeros(len(RDVs))
		time_win = np.full(len(RDVs), np.inf)

		for k, RDV in enumerate(RDVs):			
			occur = RDV >= self.threshold
			if any(occur):
				win[k] = True
				time_win[k] = np.where(occur==True)[0][0]
			else:
				win[k] = False

		return win, time_win

	def compute_values(self, patches):
		#Compute the values for each patch at each time instant (with video frame granularity)
		curr_prior = self.prior_values[patches[self.winner].label]		#prior values (V_p)

		#Gazing function (Psi)
		if self.is_new_patch:
			self.values = np.zeros(len(patches))
			for i,curr_patch in enumerate(patches):
				prior = self.prior_values[curr_patch.label]
				dist = np.linalg.norm(np.array(self.patches[self.winner].center) - np.array(curr_patch.center)) / self.diag_size
				self.values[i] = prior * np.exp(-self.k*dist)		
				self.is_new_patch = False
		else:
			#Simulate Patch depletion - Decrease the value of the current patch exponentially in time
			self.values[self.winner] = self.values[self.winner] * np.exp(-(self.phi/curr_prior)*self.patch_residence_time)	
		
		self.values = self.values/np.sum(self.values)		

		return self.values * self.eta/curr_prior		#relative values (nu_p)

	def simulate_race(self):
		#Multi-alteranative race-to-threshold model
		self.DDM_simul = []
		values = self.compute_values(self.patches)
		self.patch_residence_time += 1/self.fps

		#Simulate each racer
		for i,p in enumerate(self.patches):
			self.DDM_simul.append(self.racers[i].DDM_simulate(self.starting_points[i], values[i]))
			self.starting_points[i] = self.DDM_simul[i][-1]
			self.realizations[i] = np.hstack((self.realizations[i],self.DDM_simul[i]))

		realizations = self.realizations

		RDVs = self.compute_current_RDVs(feed_forward_inhibit=False)
		win, t_win = self.check_event_occurrence(RDVs)	#Check if any racer has reached the threshold

		if any(win):
			#Some racer reached the threshold
			RDVs_a = np.vstack(RDVs)
			rdvs_at_win = RDVs_a[:,int(np.min(t_win))]
			ranking = np.argsort(RDVs_a[:,int(np.min(t_win))])[::-1]
			winner = ranking[0]
			
			print('\n')
			print(self.patches[winner].label + '('+ str(winner) + ') won!!!\n')

			self.starting_points = np.zeros(self.num_racers)
			self.racers = []
			self.realizations = []
			for p in self.patches:
				self.racers.append(DDM(self.num_walks, self.DDM_sigma, self.ndt, p.label, self.dt))
				self.realizations.append(np.array([]))
		else:
			#Keep simulating in the next cycle
			winner = None
			ranking = None

		return winner, ranking, realizations


		