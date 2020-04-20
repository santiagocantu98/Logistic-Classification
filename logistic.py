
"""
    This Python script was done for the second practical exam of 
    Artificial Intelligence class, the exam consists of 
    creating functions in order to train the algorithmn
    so it can find the optimal w0 and w1 of the data set

    Author: Santiago Cantu
    email: santiago.cantu@udem.edu
    Institution: Universidad de Monterrey
    First created: March 29, 2020
"""

import numpy as np
import math
import pandas as pd
import utilityArtificial as ai

def main():
  """
    function that runs the program calling all the functions
    to get the minimum square error for the testing data
    after finding the minimum values of w0 and w1
  """
  # Initializing learning rate
  learning_rate = 0.0005
  # Initializing stopping criteria
  stopping_criteria = 0.01
  # load the data training data from a csv file with an url
  training_x,testing_x, training_y, testing_y,mean,sd,w = ai.store_data("https://github.com/santiagocantu98/Logistic-Classification/raw/master/diabetes.csv","training")
  # scalates the features of the testing data
  testing_data_scaled,mean,sd = ai.scale_features(testing_x,mean,sd)
  ai.print_scaled_data(testing_data_scaled,"testing")
  # store the optimal w's of the training data and trains the algorithm
  w = ai.gradient_descent(training_x,training_y,w,stopping_criteria,learning_rate)
  # print the optimal w's
  ai.print_w(w)

  # predicts the cost of the last mile with the optimal w's of the trained algorithm with the testing data
  predicted = ai.predict(w,testing_data_scaled,mean,sd)
  ai.covariance_matrix(predicted,testing_y)
  


# calls the main function
main();

