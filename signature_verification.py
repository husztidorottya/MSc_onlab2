#from sklearn.datasets import load_digits
import cv2
import numpy as np
import os
from skimage import color
from skimage import io

# stores all versions of a signature
signatures = []   
# stores if it's original or fake (1 - original, 0 - fake)
target = []

for file in os.listdir('../MSc Önlab2/SVC/Standard/006'):
    if file.find('.png') != -1:
        # gray scaling the image
        signature = color.rgb2gray(io.imread('../MSc Önlab2/SVC/Standard/006/' + file))
        
        # reshape image matrice to list
        signatures.append(signature.reshape(1,-1))
        
        if file.find('_e_') != -1:
            target.append(1)
        else:
            target.append(0)

import random
from sklearn import ensemble

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

import random
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import operator

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Using the Random Forest Classifier
classifier = ensemble.RandomForestClassifier()

#sample_images = sample_images[0][0]
len(x_train[0])

#Fit model
# the problem was with fit method, that the svc database contains the images in different size (one is 126000, the other is 127800)
# the solution is to standardize the pictures to the same size
classifier.fit(x_train,y_train)

#Attempt to predict validation data
#score=classifier.score(sample_images, sample_target)
score = classifier.score(x_test, y_test)
print('Random Tree Classifier:\n') 
print('Score\t'+str(score))

i=0

# prediction if it's the original or the fake one (1 is original, 0 is fake)
classifier.predict(x[i])
