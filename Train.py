# -*- coding: utf-8 -*-

# Import necessary libraries
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Import functions from our python program
from Used_functions import data_xml_to_df  
from Used_functions import select_random_parts
from Used_functions import open_parts_to_dataframes


# A python function that takes as parameters arrays, trains a model and saves it to a disk
# It splits 80% train set and 20% test set
# It uses TF-IDF scores 


input_folder = './ten_xml_parts'


def train(x,y):
    
    print("\n------ TRAINING PHASE ------")
    
    # Split the dataset into 80% train and 20% test
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, random_state=40)
    print("\nThe dataset is split into 80% train and 20% test!")
         
    # Create a logistic regression model (LR)    
    LR = LogisticRegression()

    # Train the model using train set
    LR.fit(X_train, Y_train)

    print("Number of iterations: {}".format(LR.n_iter_))
             
    # Save the model to a disk
    filename = 'Logistic_regression_model.sav'
    pickle.dump(LR, open(filename, 'wb'))
    print("Model saved!\n")
    
    return x,y


# %%  Call the functions

# Call the function to select 8 random parts to use for training
#training_parts, testing_part = select_random_parts(input_folder,0.8)


# Call the function to create a general dataframe out of the 8 parts which will be used for training
#data_x, data_y = open_parts_to_dataframes(training_parts,input_folder)


# Call the function to train the model using 8 parts
#X,Y = train(data_x, data_y)