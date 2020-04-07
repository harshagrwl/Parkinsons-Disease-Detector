# Parkinsons-Disease-Detector
Detecting Parkinson's Disease using Computer Vision with 81% accuracy.

## Overview
Parkinson's Disease is detected using the image samples of Spiral Test and Wave Test. These tests are carried out by Physicians to detect if a person has Parkinson's Disease or not. If a person is diagnosed with parkinson's, he/she is not able to complete these tests in due time and does them abnormally.
We will be using these image samples and detect if a Person is suffereing from Parkinson's or not.

## Dataset
The dataset used here is taken from NIATS Centre for Innovation and Technology in Healthcare.
It consists of 204 (104 for Spiral Test and 104 for Wave Test) images splitted into Training and Testing Data.

## Algorithm
The algorithm used in this project is Histogram of Oriented Gradients Descriptor (also called as HOG). This algorithm was introduced in the paper [Histogram of Oriented Gradients for Human Detection](https://ieeexplore.ieee.org/document/1467360) by N.Dalal and B.Triggs.
The input images are quantified using the HOG algorithm and their features are extracted.
After that we use Random Forest based Classifier for training.

## Working
- Clone this repository to your local machine. ```git clone https://github.com/harshagrwl/Parkinsons-Disease-Detector.git```
- Change the Working Directory. ```cd Parkinsons-Disease-Detector```
- Extact the dataset.
- Run the detector for spiral test images. ```python detect_parkinsons.py --dataset dataset/spiral```
- Run the detector for wave test images. ```python detect_parkinsons.py --dataset dataset/wave```
### Note
You can also specify the number of training trails for the program by specifying additional parameter ```--trails <no. of trials>``` with the python command. The default value of trials is set to 5.

## Metrics
### Spiral Test
- Accuracy of 81%
- Sensitivity of 71.67%
- Specificity of 90.33%

### Wave Test
- Accuracy of 71%
- Sensitivity of 67.33%
- Specificity of 74.67%

## Contributions
Pull Requests as well as Suggestions are welcome.
