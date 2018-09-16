import cv2
import numpy as np
import os
from skimage import color
from skimage import io
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import operator

# CONSTANTS
# path to signature's directory
path = './SVC/Standard/006'

def main():
	# stores all versions of a signature
	signatures = []   
	# stores if it's original or fake (1 - original, 0 - fake)
	target = []

	for file in os.listdir(path):
    		if file.find('.png') != -1:
        		# gray scaling the image
        		signature = color.rgb2gray(io.imread(path + '/' + file))
        
        		# reshape image matrice to list
        		signatures.append(signature.reshape(1,-1))
        
        		# create label to image whether it's the original or fake one (1 - original, 0 - fake)
        		if file.find('_e_') != -1:
            			target.append(1)
        		else:
            			target.append(0)


	#Define variables
	n_samples = len(signatures)
	y = np.asarray(target)
	x = []

	for img in range(0,len(signatures)):
    		x.append(signatures[img][0])
    
	# get longest image represantation and padding the rest of it
	longest_image = max(x, key=lambda k: len(k))
	longest_size = len(longest_image)

	# white pixel
	white = x[0][0]

	# pad the difference
	for sig in range(0,len(x)):
    		if len(x[sig]) < longest_size:
        		# how many white pixels have to be added
        		difference = longest_size - len(x[sig])
        
        		# adding extra white pixels as padding
        		for pad in range(0,difference):
            			x[sig] = np.append(x[sig], white)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	#Using the Random Forest Classifier
	classifier = ensemble.RandomForestClassifier()
	
	# using neural network
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
	#                     hidden_layer_sizes=(5, 2), random_state=1)

	#Fit model
	# the problem was with fit method, that the svc database contains the images in different size (one is 126000, the other is 127800)
	# the solution is to standardize the pictures to the same size
	classifier.fit(x_train,y_train)
	#clf.fit(x_train, y_train)

	#Attempt to predict validation data
	score = classifier.score(x_test, y_test)
	#score = clf.score(x_test,y_test)
	print('Random Tree Classifier:\n') 
	print('Score\t'+str(score))

	# check prediction to a test data
	# prediction if it's the original or the fake one (1 is original, 0 is fake)
	classifier.predict(x_test[0])
	#clf.predict(x_test[0])



if __name__ == '__main__':
    main()
