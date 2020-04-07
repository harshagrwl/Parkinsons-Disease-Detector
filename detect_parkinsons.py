
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import cv2
import os

def quantify_image(image):
	
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")

	
	return features

def load_split(path):
	
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []

	
	for imagePath in imagePaths:

		label = imagePath.split(os.path.sep)[-2]



		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (200, 200))



		image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


		features = quantify_image(image)


		data.append(features)
		labels.append(label)


	return (np.array(data), np.array(labels))

