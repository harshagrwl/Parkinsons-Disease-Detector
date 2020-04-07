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

# Feature extraction using HOG
def quantify_image(image):
	
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")

	
	return features

# Resizing and Thresholding the Image
def load_split(path):
	
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []

	
	for imagePath in imagePaths:

		label = imagePath.split(os.path.sep)[-2]



		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (200, 200))


		# Inverse Binary and Ostu Thresholding to make the image
		# appear like white outline on a black background
		image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


		features = quantify_image(image)


		data.append(features)
		labels.append(label)


	return (np.array(data), np.array(labels))

# parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-t", "--trials", type=int, default=5,
	help="# of trials to run")
args = vars(ap.parse_args())



trainingPath = os.path.sep.join([args["dataset"], "training"])
testingPath = os.path.sep.join([args["dataset"], "testing"])

(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)


# encoding the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)


trials = {}
for i in range(0, args["trials"]):
	# training the model
	model = RandomForestClassifier(n_estimators=100)
	model.fit(trainX, trainY)

	predictions = model.predict(testX)
	metrics = {}

	cm = confusion_matrix(testY, predictions).flatten()
	(tn, fp, fn, tp) = cm
	metrics["acc"] = (tp + tn) / float(cm.sum())
	metrics["sensitivity"] = tp / float(tp + fn)
	metrics["specificity"] = tn / float(tn + fp)

	for (k, v) in metrics.items():
		l = trials.get(k, [])
		l.append(v)
		trials[k] = l


for metric in ("acc", "sensitivity", "specificity"):
	# Finding mean and std deviation
	values = trials[metric]
	mean = np.mean(values)
	std = np.std(values)


	print(metric)
	print("=" * len(metric))
	print("u={:.4f}, o={:.4f}".format(mean, std))
	print("")

# randomly select iamges for output
testingPaths = list(paths.list_images(testingPath))
idxs = np.arange(0, len(testingPaths))
idxs = np.random.choice(idxs, size=(25,), replace=False)
images = []

# preprocessing the output images
for i in idxs:
	image = cv2.imread(testingPaths[i])
	output = image.copy()
	output = cv2.resize(output, (128, 128))

	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (200, 200))
	image = cv2.threshold(image, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


	features = quantify_image(image)
	preds = model.predict([features])
	label = le.inverse_transform(preds)[0]

	# Marking the output images
	color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
	cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		color, 2)
	images.append(output)

# creating a montage using 128x128 tiles with 5 rows and 5 columns
montage = build_montages(images, (128, 128), (5, 5))[0]

# Printing the final montage of predicted results
cv2.imshow("Output", montage)
cv2.waitKey(0)

