import numpy as np
import os
import glob
from utils import sorted_nicely, mkGaussian, compute_density_image, clean_eyedata
import cv2
from feat_map import Feat_map

class Feature_maps(object):
 
	def __init__(self, dynDir=None, facemapDir=None):
        
		self.dynDir = dynDir
		self.facemapDir = facemapDir
		self.all_fmaps = []

	def release_fmaps(self):
		self.all_fmaps = []

	def load_feature_maps(self, video_name, vidHeight=None, vidWidth=None):

		sts_saliency_maps = os.listdir(self.dynDir + video_name[:-4])
		sorted_sts_saliency_maps = sorted_nicely(sts_saliency_maps)
		self.num_sts = len(sorted_sts_saliency_maps)

		face1_maps = glob.glob(self.facemapDir + video_name[:-4] + '/*_speaker.png')
		sorted_face1_maps = sorted_nicely(face1_maps)
		self.num_speak = len(sorted_face1_maps)

		face2_maps = glob.glob(self.facemapDir + video_name[:-4] + '/*_nonspeaker.png')
		sorted_face2_maps = sorted_nicely(face2_maps)
		self.num_nspeak = len(sorted_face2_maps)
		
		mu = np.array([vidWidth/2,vidHeight/2])
		wy=vidHeight/12	
		wx=vidWidth/12
		sigma = [wx, wy]
		
		F = mkGaussian(mu, sigma, 0, vidWidth, vidHeight).T
		center_bias_map = Feat_map(feat_map=F/np.sum(np.sum(F)), name='CB')
		
		uniform_map = Feat_map(feat_map=np.ones(center_bias_map.shape)/np.prod(center_bias_map.shape), name='Uniform')

		self.sts_names = sorted_sts_saliency_maps
		self.face1_names = sorted_face1_maps
		self.face2_names = sorted_face2_maps
		self.cb = center_bias_map
		self.uniform = uniform_map
		self.video_name = video_name[:-4]
		self.vidHeight = vidHeight
		self.vidWidth = vidWidth

		return sorted_sts_saliency_maps, sorted_face1_maps, sorted_face2_maps, center_bias_map, uniform_map


	def read_current_maps(self, gaze, frame_num, compute_heatmap=False):
		
		curr_sts = cv2.imread(self.dynDir + self.video_name + '/' + self.sts_names[frame_num], 0)
		
		self.sts = Feat_map(feat_map=curr_sts/float(np.sum(curr_sts)), name='STS')		
		curr_sts = np.reshape(curr_sts,-1, order='F')
		
		curr_face1 = cv2.imread(self.face1_names[frame_num], 0)

		if np.sum(curr_face1) != 0:
			self.speaker = Feat_map(feat_map=curr_face1/float(np.sum(curr_face1)), name='Speaker')
			curr_face1 = np.reshape(curr_face1,-1, order='F')/np.max(curr_face1)
		else:
			self.speaker = Feat_map(feat_map=curr_face1, name='Speaker')
			curr_face1 = np.reshape(curr_face1,-1, order='F')
		
		curr_face2 = cv2.imread(self.face2_names[frame_num], 0)

		
		if np.sum(curr_face2) != 0:
			self.non_speaker = Feat_map(feat_map=curr_face2/float(np.sum(curr_face2)), name='non_Speaker')
			curr_face2 = np.reshape(curr_face2,-1, order='F')/np.max(curr_face2)
		else:
			self.non_speaker = Feat_map(feat_map=curr_face2, name='non_Speaker')
			curr_face2 = np.reshape(curr_face2,-1, order='F')

		self.all_fmaps.append(self.cb)
		self.all_fmaps.append(self.sts)
		self.all_fmaps.append(self.speaker)
		self.all_fmaps.append(self.non_speaker)

		if compute_heatmap:		
			w = self.vidWidth
			h = self.vidHeight
			curr_gaze = clean_eyedata(gaze[:,frame_num,:], w, h)
			nObsTrain = int(np.floor(curr_gaze.shape[0]*1))
			Eye_Position_Map_train = np.zeros([w,h])
			curr_gaze_train_ind = curr_gaze[:nObsTrain,:].astype(int)
			Eye_Position_Map_train = compute_density_image(curr_gaze_train_ind, [w,h])
			self.original_eyeMap = Eye_Position_Map_train.copy()