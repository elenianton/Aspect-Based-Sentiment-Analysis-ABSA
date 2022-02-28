# Aspect-Based-Sentiment-Analysis-ABSA-

Aspect Based Sentiment Analysis (ABSA)is the task of mining and summarizing opinions from text about specific entities (e.g. a restaurant). Typically, an ABSA system attempts to detect the main aspects that are discussed and estimate the polarity for each of them.
In this setting one approach is to define an inventory (a closed set) of aspects for a domain and then build a system that can identify these aspects in text and make a decision about polarity.For the restaurant domain an aspect inventory has been defined in the context of SemEval conferences; e.g., the inventory contains FOOD#QUALITY, 
RESTAURANT#GENERAL, RESTAURANT#PRICES etc. 

Main goal:
Develop a system that decides the polarity of a given aspect. 

Steps:

1.	Downloading dataset: http://metashare.ilsp.gr:8080/repository/browse/semeval-2016-absa-restaurant-reviews-english-train-data-subtask-1/cd28e738562f11e59e2c842b2b6a04d703f9dae461bb4816a5d4320019407d23/. It is an XML file that contains 350 restaurant reviews split into 2000 sentences. Each sentence has been manually annotated with aspects (“category” attribute) and for each aspect a “polarity” label has been assigned (positive, neutral, negative)  
2.	Reading/parsing the XML file and split it to 10 parts (35 reviews per part). Each part should be stored in a separate file (part1.xml, part2.xml, … part10.xml)
3.	train.py --> trains a model and saves it to disk. 
    The function takes as parameter an array that indicates which parts will be used for training. 
4.	Test.py --> loads a saved model and uses it for predicting the polarities for the sentence aspects of a part. 
    A parameter of the function specify which part will be used.
5.	Experiments.py -->  uses functions of train.py and test.py. 
    The experiments.py do: 
    a.	10-fold cross validation.In each iteration 9 parts will be used for training and 1 for testing/evaluation. Estimate accuracy for each of the 10 folds/iterations and               calculate an average of them.
    b.	Repetition of the experiments of 5.a with using only the 1/3 and 2/3 of the training parts; i.e., 3 parts, 6 parts.
6.	Finaly, we using a feature selection technique

