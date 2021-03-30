import os
import numpy as np
from gaze import Gaze
from video import Video
from patch_race_DDM import race_DDM
from feature_maps import Feature_maps
import matplotlib.pyplot as plt
from pylab import *
from gaze_sampler_DDM import GazeSampler
from skimage.draw import circle
import scipy
import os.path
import imageio
import scipy.stats as stats
from utils import compute_density_image
import cv2

def get_fix_from_scan(scan_dict, nFrames):
	generated_eyedata = np.zeros([2, nFrames, len(scan_dict)])
	fixations = np.zeros([2, nFrames])
	for i,k in enumerate(scan_dict.keys()):
		s = scan_dict[k]
		N = s.shape[0] // 10
		frames = np.split(s, N)
		for j, f in enumerate(frames):
			if j < nFrames:
				med = np.median(f, axis=0)
				fixations[:,j] = np.median(f, axis=0)
		generated_eyedata[:,:,i] = fixations
	return generated_eyedata

def get_all_patches(featMaps):
	patches = []	
	for fmap in featMaps.all_fmaps:
		for patch in fmap.patches:
			patches.append(patch)
	return patches

def sample_values(prior_values, s):
	values = []
	if s == 0:
		return prior_values
	else:
		for v in prior_values:
			values.append(np.max([0.01, stats.norm.rvs(v, s)]))  #sample rectified normal
		return np.array(values)

vidDir = 'data/videos/'
gazeDir = 'data/fix_data/'
dynmapDir = 'speaker_detect/data/ST_maps/'
facemapDir = 'speaker_detect/data/face_maps/'

save_GIF = True
colors = ['g', 'b', 'r', 'k', 'm', 'c', 'y']

gazeObj = Gaze(gazeDir)
videoObj = Video(vidDir)
featMaps = Feature_maps(dynmapDir, facemapDir)

racers = ['Speaker', 'Non Speaker', 'CB', 'STS']
curr_vid_name = '012.mp4'

scanPath = {}
fname = 'saved/gen_gaze/'+curr_vid_name[:-4] + '.npy'

#Gaze -------------------------------------------------------------------------------------------------------		
gazeObj.load_gaze_data(curr_vid_name)

#Video ------------------------------------------------------------------------------------------------------	
videoObj.load_video(curr_vid_name)
FOAsize = int(np.max(videoObj.size)/10)
diag_size = np.sqrt(videoObj.vidHeight**2 + videoObj.vidWidth**2)
fps = videoObj.frame_rate

#Feature Maps ----------------------------------------------------------------------------------------------- 
featMaps.load_feature_maps(curr_vid_name, videoObj.vidHeight, videoObj.vidWidth)

nFrames = min([len(videoObj.videoFrames), featMaps.num_sts, featMaps.num_speak, featMaps.num_nspeak])
nRows = 2
nCols = 3
fig = plt.figure(figsize=(16, 10))

compute_heatmap = True

#Load generated scanpaths
generated_scan = np.load('data/gen_gaze/'+curr_vid_name[:-4]+'.npy', allow_pickle=True).item()
generated_eyedata = get_fix_from_scan(generated_scan, nFrames)

#Prior values for speaker, non-speaker, CB and sts
prior_values = np.array([0.75, 0.21, 0.01, 0.03,])

winner = 0
ranking = np.zeros(len(prior_values))
prev_patch = None

#Gaze Sampler -----------------------------------------------------------------------------------------------
gazeSampler = GazeSampler() 

#For each video frame
for iframe in range(nFrames):

	print('Frame number: ' + str(iframe))
	frame = videoObj.videoFrames[iframe]
	featMaps.read_current_maps(gazeObj.eyedata, iframe, compute_heatmap=compute_heatmap)


	if winner is not None:

		#Center Bias saliency and proto maps
		featMaps.cb.esSampleProtoParameters()
		featMaps.cb.define_patchesDDM()

		#Speaker saliency and proto maps -------------------------------------------------------------------------
		featMaps.speaker.esSampleProtoParameters()
		featMaps.speaker.define_patchesDDM()

		#Non Speaker saliency and proto maps ---------------------------------------------------------------------		
		featMaps.non_speaker.esSampleProtoParameters()
		featMaps.non_speaker.define_patchesDDM()
		
		#Low Level Saliency saliency and proto maps ---------------------------------------------------------------		
		featMaps.sts.esSampleProtoParameters()
		featMaps.sts.define_patchesDDM()

		patches = get_all_patches(featMaps)

		i=1
		while winner > len(patches)-1:
			winner = ranking[i]
			i += 1

		values = sample_values(prior_values, s=0.07)

		if prev_patch != winner:
			race_model = race_DDM(prior_values=values, patches=patches, diag_size=diag_size, winner=winner, fps=fps)
			prev_patch = winner

	#Gaze Sampling -------------------------------------------------------------------------------------------
	gazeSampler.sampleDDM(iframe=iframe, patches=patches, FOAsize=FOAsize, winner=winner)

	#Simulate DDM --------------------------------------------------------------------------------------------
	winner, ranking, realizations  = race_model.simulate_race()	

	#Plotting stuff ------------------------------------------------------------------------------------------
	curr_fix = generated_eyedata[:,iframe,:].T
	gen_saliency = compute_density_image(curr_fix, [videoObj.vidWidth, videoObj.vidHeight])

	#Plot simulation results
	fig.clf()
	numfig=1
	plt.subplot(nRows,nCols,numfig)
	plt.imshow(frame)
	plt.title('Original Frame')
	plt.axis('off')
	
	numfig+=1
	plt.subplot(nRows,nCols,numfig)
	plt.imshow(featMaps.original_eyeMap)
	plt.title('"Real" Fixation Map')
	plt.axis('off')
	
	numfig+=1
	plt.subplot(nRows,nCols,numfig)
	for i in range(len(patches)):
		plt.plot(realizations[i], label=patches[i].label)
	plt.plot(np.arange(0,260), np.ones(260,)*1.7)
	plt.legend(loc='lower right')
	plt.grid()
	plt.ylim(-1, 2.1)
	plt.xlim(0,260)
	plt.title('Race Model \n (Decision Making Dynamics)')

	numfig+=1
	finalFOA = gazeSampler.allFOA[-1].astype(int)
	plt.subplot(nRows,nCols,numfig)
	BW = np.zeros(videoObj.size)
	rr,cc = circle(finalFOA[1], finalFOA[0], FOAsize)
	rr[rr>=BW.shape[0]] = BW.shape[0]-1
	cc[cc>=BW.shape[1]] = BW.shape[1]-1
	BW[rr,cc] = 1
	FOAimg = cv2.bitwise_and(cv2.convertScaleAbs(frame),cv2.convertScaleAbs(frame),mask=cv2.convertScaleAbs(BW))
	plt.imshow(FOAimg)
	plt.title('Simulated Focus Of Attention (FOA)')
	plt.axis('off')

	#Heat Map
	numfig+=1
	plt.subplot(nRows,nCols,numfig)
	plt.imshow(gen_saliency)
	plt.title('"Generated" Fixation Map')
	plt.axis('off')

	#Scan Path
	numfig+=1
	plt.subplot(nRows,nCols,numfig)
	plt.imshow(frame)
	sampled = np.concatenate(gazeSampler.sampled_gaze)
	plt.plot(sampled[:,0], sampled[:,1], '-x')
	plt.title('Generated Gaze data \n (Spatial Dynamics)')
	plt.axis('off')

	plt.pause(1/25.)

	#At the end of the loop
	featMaps.release_fmaps()

scanPath = np.concatenate(gazeSampler.sampled_gaze)