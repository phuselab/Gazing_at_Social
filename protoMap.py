import numpy as np

class ProtoMap(object):
	"""Class to handle the datasets"""
 
	def __init__(self, protoMap=None, name=None):
        
		self.protoMap = protoMap.T
		self.name = name
		self.extent = np.array([0, protoMap.shape[0]-1, 0, protoMap.shape[1]-1])

	@property
	def shape(self):
		return self.protoMap.shape

	@property
	def value(self):
		return self.protoMap
