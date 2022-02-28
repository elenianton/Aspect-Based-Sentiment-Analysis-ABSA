# -*- coding: utf-8 -*-


# Import necessary libraries
import pickle
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Import functions from our python program
from Used_functions import data_xml_to_df  
from Used_functions import select_random_parts
from Used_functions import open_parts_to_dataframes


# A python program that takes as parameters arrays and a saved trained model
# It loads the model and uses it to predict the polarities for the sentence aspects of a part


input_folder = './ten_xml_parts'

# The model saved in pickle
#modelfilename = os.path.abspath('Logistic_regression_model.sav')

def predict_polarity(x,y, saved_model):
    
    print("\n------ PREDICTING PHASE ------")
    # Load the model    
    loaded_model = pickle.load(open(saved_model, 'rb'))
    
    # Split the dataset into 80% train and 20% test
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, random_state=40)
    print("\nThe dataset is split into 80% train and 20% test!")
     
    # Train the model using train set
    loaded_model.fit(X_train, Y_train)
    print("\nFitting data to the loaded logistic regression model...")
                
    # Model evaluation for test set (predicted values of y given the X_test data)
    y_test_predict = loaded_model.predict(X_test)
            
    # Model evaluation using the train set (predicted values of y given the X_train data)
    y_train_predict = loaded_model.predict(X_train)
    
    # Calculate accuracy of Logistic regression model
    LR1_accuracy = metrics.accuracy_score(Y_train,y_train_predict)*100
    LR_accuracy = metrics.accuracy_score(Y_test,y_test_predict)*100

    print("\nAccuracy of Logistic Regression model - train set: {} %".format(LR1_accuracy))
    print("Accuracy of Logistic Regression model - test set: {} %".format(LR_accuracy))
    print("\nThe Beta parameters:", loaded_model.coef_)
    print("\nThe Intercept (bias):", loaded_model.intercept_)
            
    # Get the classificatin report for the train set
    #print("\nClassification report of LR Confusion matrix - Train set :\n",classification_report(Y_train,y_train_predict,labels = np.unique(y_train_predict)))
           
    # Get the classificatin report for the test set
    print("Classification report of LR  Confusion matrix - Test set:\n",classification_report(Y_test,y_test_predict,labels = np.unique(y_test_predict)))
    
    # UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples
    # labels = np.unique(y_test_predict): specify the labels that were predicted and ignore the ones that were not predicted
      
    return x,y


# %%  Call the functions

# Call the function to select 8 random parts to use for testing
#training_parts, testing_parts = select_random_parts(input_folder,0.8)


# Call the function to create a general dataframe out of the 2 parts which will be used for testing
#data_x, data_y = open_parts_to_dataframes(testing_parts,input_folder)


# Call the function to make predictions and evaluate
#X,Y = predict_polarity(data_x, data_y, modelfilename)
