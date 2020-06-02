# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#We never find tabs in Reviews
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

#Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[]
for i in range(0,1000):
    #corpus is a list of collection of text
    #download stopwords to remove those list of unrelevant words
    review = re.sub('[^a-zA-z]',' ', dataset['Review'][i])
    review = review.lower()
    #Now we remove words that are not relevent for predicting like the,that,and...
    #review is a string
    #We should slipt the string into list of different words
    review = review.split()
    #Steming
    #Here we store oly root word like like love instead of loved,loving...
    #We apply steming on a single word 
    ps=PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    # Here we use set because it is easier to retrieve from set than list
    corpus.append(review)

#Creating bag of words model
#The fact we have lot of zeroes is called sparcity
#So we create this sparx matrix
# We get one if a word is present in table and zero if not
# This is related to classification model
#All the independent variables as columns(words) and dependent variables(0s and 1s) are calculated based on them

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
#This will select 1500 words with max features
#We can reduce sparcity by reducing words thanks to max_features
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1]
#Models used for NLP are naive bayes,decision tree, random forest
#So lets implement Naive Bayes
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting the classifier to the Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

#Predicting Test results
y_pred = classifier.predict(x_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Accuracy
(55+91)/200
#73% Accuracy






"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""