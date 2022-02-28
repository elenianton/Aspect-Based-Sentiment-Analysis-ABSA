# -*- coding: utf-8 -*-

# Import necessary libraries
import os
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Import functions from our python programs
from Used_functions import data_xml_to_df  
from Used_functions import select_random_parts
from Used_functions import open_parts_to_dataframes
from Train import train
from Test import predict_polarity


input_folder = './ten_xml_parts'

# The model saved in pickle
modelfilename = os.path.abspath('Logistic_regression_model.sav')

def cross_validation(x,y,k):
    
    print("\nPerforming cross validation...")
    
    # instantiate model
    model = LogisticRegression(solver= 'liblinear')

    kf = KFold(n_splits = k, random_state=None)
    
    # Evaluate the model
    scores = cross_val_score(model, x, y, cv = kf)
    
    # Calculate the average accuracy
    avg_acc_score = sum(scores)/10   

    print("\nCross-Validation Accuracy for each fold:\n{} ".format(scores))
    print("\nAverage accuracy for each 10-fold cross validation: {} \n".format(avg_acc_score))
    
    return scores, avg_acc_score 

#--------------------------------------- FEATURE SELECTION -------------------------------------------

# Call the function to select 9 random parts to use for training
training_parts, testing_part = select_random_parts(input_folder,0.9)

# Call the function to create a general dataframe out of the 9 parts which will be used for training
data_x, data_y = open_parts_to_dataframes(training_parts,input_folder)


# The output from SelectKBest will be in the same length as the original feature array
n = len(data_x)

# Feature extraction
test = SelectKBest(score_func=chi2, k=n)
fit = test.fit(data_x, data_y)

k_best_features = list(test.get_support(indices = True))
#print(sorted(k_best_features,reverse=True))

# Summarize scores
np.set_printoptions(precision=3)
scores = fit.scores_
#print(len(scores))

# Sort scores in descending order
sorted_list = sorted(scores, reverse=True)
#print(sorted_list)

# Fit and then transform the most important features
X_features = test.transform(data_x)

#------ TRAIN ------
# Call the function to train the model using the selected best features
X,Y = train(X_features, data_y)

#------ PREDICT / EVALUATE ------
# Call the function to make predictions based on the selected best features
x_val,y_val = predict_polarity(X,Y, modelfilename)

# Perform 10-fold cross validation based on the selected best features
cross_val_sel_feat = cross_validation(X,Y,10)


