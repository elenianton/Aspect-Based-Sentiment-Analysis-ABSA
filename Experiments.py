# -*- coding: utf-8 -*-

# Import necessary libraries
# evaluate a logistic regression model using k-fold cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Import functions from our python program
from Used_functions import data_xml_to_df  
from Used_functions import select_random_parts
from Used_functions import open_parts_to_dataframes
from Train import train
from Test import predict_polarity


input_folder = './ten_xml_parts'


# The model saved in pickle
modelfilename = 'Logistic_regression_model.sav'


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


# %%---------- Perform cross validation using the functions from train.py and test.py --------------


# Call the function to select 9 random parts to use for training
training_parts, testing_part = select_random_parts(input_folder,0.9)


#------ 9 PARTS ------

# Call the function to create a general dataframe out of the 9 parts which will be used for training
data_x, data_y = open_parts_to_dataframes(training_parts,input_folder)


# Call the function to train the model using 9 parts
X,Y = train(data_x, data_y)


# Call the function to make predictions and evaluate the 9 parts
x_val,y_val = predict_polarity(X,Y, modelfilename)


# Perform 10-fold cross validation on the trained dataset
cross_val_train = cross_validation(X,Y,10)



#------ 1 PART ------

# Call the function to create a dataframe from the part which will be used for testing
input_x, input_y = open_parts_to_dataframes(testing_part,input_folder)


# Call the function to train the model using 1 part
x,y = train(input_x, input_y)


# Call the function to make predictions and evaluate the 1 part
x_val,y_val = predict_polarity(x,y, modelfilename)


# Perform 10-fold cross validation on the trained dataset
cross_val_train = cross_validation(x,y,10)



# %%  Question b - 1/3 training parts

# Call the function to select 3 random parts to use for training
training_parts, testing_part = select_random_parts(input_folder,0.3)

#------ 3 PARTS ------

# Call the function to create a general dataframe out of the 9 parts which will be used for training
data_x_b1, data_y_b1 = open_parts_to_dataframes(training_parts,input_folder)


# Call the function to train the model using 9 parts
X,Y = train(data_x_b1, data_y_b1)


# Call the function to make predictions and evaluate the 9 parts
x_val,y_val = predict_polarity(X,Y, modelfilename)


# Perform 10-fold cross validation on the trained dataset
cross_val_train = cross_validation(X,Y,10)



#------ 7 PARTS ------

# Call the function to create a dataframe from the part which will be used for testing
input_x_b1, input_y_b1 = open_parts_to_dataframes(testing_part,input_folder)


# Call the function to train the model using 1 part
x,y = train(input_x_b1, input_y_b1)


# Call the function to make predictions and evaluate the 1 part
x_val,y_val = predict_polarity(x,y, modelfilename)


# Perform 10-fold cross validation on the trained dataset
cross_val_train = cross_validation(x,y,10)


# %% Question b - 2/3 training parts

# Call the function to select 3 random parts to use for training
training_parts, testing_part = select_random_parts(input_folder,0.6)

#------ 6 PARTS ------

# Call the function to create a general dataframe out of the 9 parts which will be used for training
data_x_b2, data_y_b2 = open_parts_to_dataframes(training_parts,input_folder)


# Call the function to train the model using 9 parts
X,Y = train(data_x_b2, data_y_b2)


# Call the function to make predictions and evaluate the 9 parts
x_val,y_val = predict_polarity(X,Y, modelfilename)


# Perform 10-fold cross validation on the trained dataset
cross_val_train = cross_validation(X,Y,10)



#------ 4 PARTS ------

# Call the function to create a dataframe from the part which will be used for testing
input_x_b2, input_y_b2 = open_parts_to_dataframes(testing_part,input_folder)


# Call the function to train the model using 1 part
x,y = train(input_x_b2, input_y_b2)


# Call the function to make predictions and evaluate the 1 part
x_val,y_val = predict_polarity(x,y, modelfilename)


# Perform 10-fold cross validation on the trained dataset
cross_val_train = cross_validation(x,y,10)


     
    