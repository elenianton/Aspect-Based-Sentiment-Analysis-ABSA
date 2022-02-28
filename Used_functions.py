# -*- coding: utf-8 -*-

# Import necessary libraries
import os
import re
import random
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# FUNCTIONS THAT ARE IMPORTED IN THE OTHER PYTHON PROGRAMS
# First function - data_xml_to_df(): take as input an xml file and convert it into a dataframe with the proper columns and preprocess the text
# The function is called automatically in the open_parts_to_dataframes() function

# Second function - select_random_parts(): get all the files from a folder and choose a number of random files for training and testing 

# Third function - open_parts_to_dataframes(): open a folder, get a number of parts, create a general dataframe containing preprocessed text and polarity,
# create labels for each polarity (0,1,2,3), get the values of polarity and text, transform text values into TF-IDF scores




def data_xml_to_df(xml_file):
    
    # Parse the xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()  # Root: "Reviews"
    
    #print(root.getchildren()) # "Review"
    #find_reviews = root.findall("Review")
      
    # Iterate over the tree to know all the elements
    #for elem in root.iter():
      #print(elem.tag)   # "Review", "Opinions", "Opinion", "sentence", "text"
    
    #for child in root:   # Discover the sub-elements/children in the root
      #print(child.tag, child.attrib)   # e.g. Review {"rid": "1004293"}


    # Initialize lists to fill 
    opinion_id = []     # for the opinion identifiers, e.g. "1086478"
    txt_content = []    # for the text sentences from the review, e.g. "It is terrific, as is the value."
    target = []         # for the the words identifying the entity
    category = []       # for the entity-aspect category, e.g. "SERVICE#GENERAL", "FOOD#QUALITY", "FOOD#PRICES"
    polarity = []       # for the polarity of the review, "positive" , "neutral", "negative"
    opinion_from = []   # for the str character offset identifying where the opinion begins, e.g. Opinion to="25"
    opinion_to = []     # for the str character offset identifying where the opinion ends, e.g. from="18"
    
    # Iterate over the "sentence" branches
    for sentence in root.iter("sentence"):
        
        sentence_id = sentence.get("id")
        sentence_txt = sentence.find("text").text
        opinions = sentence.find("Opinions")
        
        # The "Opinions" element contains both values and None 
        if opinions is None:  # In case there are no associated opinions(None), 
            opinion_id.append(sentence_id)  # Fill with the sentence id + 0  #+ ":0"
            txt_content.append(sentence_txt)  # Fill with the text
            target.append(np.nan)  # Fill with NaN
            category.append(np.nan)
            polarity.append(np.nan)
            opinion_from.append(np.nan)
            opinion_to.append(np.nan)      
        
        else: # In case there are associated opinions, iterate, count and fill the empty lists initialized above      
          for (i, opinion) in enumerate(sentence.find("Opinions")):
                opinion_id.append(sentence_id) #+ ":%s" % i
                txt_content.append(sentence_txt)
                target.append(opinion.get("target"))
                category.append(opinion.get("category"))
                polarity.append(opinion.get("polarity"))
                opinion_from.append(opinion.get("from"))
                opinion_to.append(opinion.get("to"))
        
  
    # Convert to dataframe:
    df = pd.DataFrame(columns=["Opinion_id", "Text", "Target", "Category", "Polarity", "Opinion_from", "Opinion_to"])
    
    df["Opinion_id"] = opinion_id
    df["Text"] = txt_content
    df["Target"] = target
    df["Category"] = category 
    df["Polarity"] = polarity
    df["Opinion_from"] = opinion_from
    df["Opinion_to"] = opinion_to
    
    print("Dataframe ready!")
    
    # Split the category into aspect and entity by # Aspect: "GENERAL" Entity: "RESTAURANT"
    df[["Entity","Aspect"]] = df["Category"].str.split("#", expand=True) 
       
    # Fill the NaN values with "NULL"
    df.fillna('NULL', inplace=True)
    
    # It's time to preprocess the sentences in the reviews
    def preprocess_text(text):
                
        # Remove tags
        TAG_RE = re.compile(r'<[^>]+>')
        no_tags = TAG_RE.sub('',text)  
                
        # Remove unusual characters
        text = re.sub('<[^>]*>', '', no_tags)
    
        # Remove emoticons
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
                
        # Convert all words to lowercase
        text = re.sub('[\W]+', ' ', text.lower()) + " ".join(emoticons).replace('-', '')
    
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        #print(text)
                
        return text
    
    # Apply the function to preprocess the texts in the reviews
    df['Text'] = df['Text'].apply(preprocess_text)
            
                                                          
    return df



def select_random_parts(folder,number):
    
    # List all the ten parts in the directory
    parts = [p for p in os.listdir(folder) if os.path.join(folder, p)]
      
    # Set the number of parts you want to select randomly NO DUPLICATES
    random_parts_training = random.sample(parts, int(len(parts)*number))
    
    # Get the remaining part that is not included in the random_parts_training
    random_part_testing = [i for i in parts if i not in random_parts_training]
    
    #print(random_parts_training)
    #print(random_part_testing)
    return random_parts_training, random_part_testing



def open_parts_to_dataframes(files,folder):
    
    parts_list = []
    
    for file in files:          
        filename = os.path.join(folder, file)  # Generates full path of each file in the folder
        print("Part currently processed: ", filename)
        with open(filename, "r", encoding="utf8") as file:  # Opens the file for reading with encoding UTF-8             
            file.read()
                
            for i in range(0,1):
                df = data_xml_to_df(filename)                                       
                dataframe = df[["Text","Polarity"]]                   
                parts_list.append(dataframe)
            
                for i in parts_list:
                    df_final = pd.concat(parts_list)
    
    # Encode polarity's categorical values into numerical labels (0,1,2,3)
    le = LabelEncoder()
    df_final["Polarity"] = le.fit_transform(df_final["Polarity"])  
             
    # Get the values from the columns in arrays
    X = np.array(df_final["Text"])
    y = np.array(df_final["Polarity"])
    
    # Instantiate a TfidfVectorizer
    # Set parameters (tokenizer, stowords removal, n-gram range, lowercase)
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 1), tokenizer = nltk.word_tokenize)
    x = tfidf.fit_transform(X)
    print("Calculating TF-IDF scores...")
    
    # Convert sparse matrix into array
    x = x.toarray()
      
    return x,y   



