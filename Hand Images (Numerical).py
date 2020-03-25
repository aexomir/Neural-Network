#THIS FILE IS 90% FROM COURSERA/IMPROVING DEEP LEARNING MODELS/WEEK 7
##THIS FILE WAS THE FIRST NN I MADE IN A COURSERA PROGRAMMING ASSIGNMENT !!

# Imports...
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

# Setting random seed to 42
np.random.seed(42)

# Loading the dataset from an h5py file using tensorflow-utils
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

# Creating placeholders for input images and output classes
def create_placeholders(n_x,n_y):
    # n_x = Scalar Size of the image (px,px,rgb)
    # n_y = number of Softmax classes = 6 (0,5)

    # creating placeholder to use it in the input...
    X = tf.placeholder(tf.float32,shape=(n_x,None))
    Y = tf.placeholder(tf.float32,shape=(n_y,None))
    # we set None , because we want to be flexible about number of training and test sets(or maybe dev sets)

    return X,Y

#Creating a Tf initializer for W and B
def initialize_params():
    tf.set_random_seed(42)

    W1 = tf.get_variable("W1",(25,64*64*3),initializer=tf.contrib.layers.xavier_initializer(seed=42))
    b1 = tf.get_variable("b1",(25,1),initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2",(12,25),initializer=tf.contrib.layers.xavier_initializer(seed=42))
    b2 = tf.get_variable("b2",(12,1),initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3",(6,12),initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3",(6,1),initializer=tf.zeros_initializer())

    params = {"W1":W1,
              "b1":b1,
              "W2":W2,
              "b2":b2,
              "W3":W3,
              "b3":b3}
    return params

#tf.reset_default_graph()
## it defaults the values and prevents overwriting

# Forward Propagation
def forward_propagation(X,params):
    # mentioning values
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)

    # It is important to note that the forward propagation stops at `z3`.The reason is that in tensorflow the last linear
    # layer output is given as input to the function computing the loss.Therefore, you don't need `a3`!
    return Z3

#Computing Cost
def compute_cost(Z3,Y):
    # In the function we're gonna use, dimension is like (number of examples, num_classes) ; so we must transpose Z3,Y

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    # Cost Function for SoftMax
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    return cost

# The beauty of using frameworks is that it backpropagation by cost function
# Building the Model
def model(X_train,Y_train,X_test,Y_test,learning_rate=0.01,num_epochs = 1500,mini_batch_size=32,print_cost=True):
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]
    seed = 42
    costs = []

    # Creating Placeholders of shape (n_x, n_y)
    X,Y = create_placeholders(n_x=n_x,n_y=n_y)

    # Initiazing Parameters
    params = initialize_params()

    # Forward Propagation : Build the forward propagation in the Tensorflow graph
    Z3 = forward_propagation(X=X,params=params)

    # Computing Cost
    cost = compute_cost(Z3=Z3,Y=Y)

    # Back Propagation and building an Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initizer for X,Y
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Doing Training Loop by mini-batches...
        for epoch in range(num_epochs):

            epoch_cost = 0          #Defines a cost related to an epoch
            num_mini_batches = int(m/mini_batch_size)
            minibatches = random_mini_batches(X_train,Y_train,mini_batch_size,seed)

            for minibatch in minibatches:

                (minibatch_X,minibatch_Y) = minibatch
                _,minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                epoch_cost += minibatch_cost/num_mini_batches

            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i : %f" %(epoch,epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

    # Plotting the Cost function
    plt.plot(np.squeeze(costs))
    plt.ylabel("Cost")
    plt.xlabel("#Iterations(tens)")
    plt.title("Learning_rate: ",str(learning_rate))
    plt.show()

    # Saving Parameters in a Variable (w,b --> params)
    params = sess.run(params)
    print("Parameters Have been Trained !!!")

    #Calculating Correct Predictions
    correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))

    # Calculating Accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    print("Train Accuracy: ",accuracy.eval({X:X_train,Y:Y_train}))
    print("Test Accuracy: ",accuracy.eval({X:X_test,Y:Y_test}))

    return params


## TRAINING (Estimated Training Time :approx 8 min)
parameters = model(X_train,Y_train,X_test,Y_test)

# Predicting Special Picture
import scipy
from PIL import Image
from scipy import ndimage

image = 'name.jpg'
#PreProccesing
fname = image
image = np.array(ndimage.imread(fname,flatten=False))
img = image/255.
my_img = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T)
my_img_prediction = predict(my_img,parameters=parameters)

#Showing the Image
plt.imshow(image)
print("Your Algorythm Predicts: ",+str(np.squeeze(my_img_prediction)))