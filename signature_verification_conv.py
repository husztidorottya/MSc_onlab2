#Importing tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split

# CONSTANTS
# stores all versions of a signature
signatures = []   
# stores if it's original or fake (1 - original, 0 - fake)
target = []

# read all the pictures from a directory
def read_directory(path):
    for file in os.listdir(path):
        if file.find('.png') != -1:
            # gray scaling the image
            signature = color.rgb2gray(io.imread(path + '/' + file))
        
            # reshape image matrice to list
            signatures.append(signature.reshape(1,-1))
        
            if file.find('_e_') != -1:
                target.append(1.0)
           else:
                target.append(0.0)
                

# create padding to standardise the input images row number
def padding(signatures):
    X = []
    for img in range(0,len(signatures)):
        X.append(signatures[img][0])
    
    # get longest image represantation and padding the rest of it
    longest_image = max(X, key=lambda k: len(k))
    longest_size = len(longest_image)   
    
    # white pixel
    white = X[0][0]
    
    # pad the difference
    for sig in range(0,len(X)):
        if len(X[sig]) < longest_size:
            # how many white pixels have to be added
            difference = longest_size - len(X[sig])
        
            # adding extra white pixels as padding
            for pad in range(0,difference):
                X[sig] = np.append(X[sig], white)
                
    return X, longest_size

def main():
    # read pictures from a directory
    read_directory('./SVC/Standard/006')
    #read_directory('./SVC/Standard/008')
    #read_directory('./SVC/Standard/010')
    
    # how many pictures we have read
    n_samples = len(signatures)
    
    X, longest_size = padding(signatures)
    Y = np.asarray(target)

    # split input to train and test dataset
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    X_train = np.asarray(x_train)
    Y_train = np.asarray(y_train).reshape(len(y_train),1)
    X_test = np.asarray(x_test)
    Y_test = np.asarray(y_test).reshape(len(y_test),1)
    
    # one hot vectors is 0 in most dimension except for a single which 
    x = tf.placeholder(tf.float32, [None, longest_size])
    
    #Creating weights and biases
    w = tf.Variable(tf.zeros([longest_size,1]))
    b = tf.Variable(tf.zeros([1]))

    # our model
    y = tf.nn.softmax(tf.matmul(x,w) + b)

    #training our model
    #y_ = tf.placeholder(tf.float32, [None,10])
    y_ = tf.placeholder(tf.float32, [None,1])

    #cross entropy function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))#, reduction_indi$

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #gradient optimizer tries to move to that direction in the computation graph wh$
    #learning rate/ step length  is 0.5. It will descend through cross_entropy.
    
    # Variables are not initialized on their own. This command is needed to initial$
    init = tf.global_variables_initializer()
    
    #interactive session:
    with tf.Session() as sess:
        sess.run(init) # reset values to incorrect defaults.

        for i in range(100):
            #batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x : X_train, y_: Y_train})
    
        #Evaluating the model:
        #prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        prediction = tf.equal(y, y_)
        #print(y_.eval({x: X_test, y_: Y_test}))
        #print(y.eval({x: X_test, y_: Y_test}))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: X_test , y_: Y_test}))


if __name__ == '__main__':
    main()

