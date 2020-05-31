#Apriori

#A person who buys cereals also more likely to buy milk
#So its advantageous for mart to keep them together
#People who bought this also bought that recommendations

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

%matplotlib inline
# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
#If customer buys product A then they are likely to buy product B
transactions=[]
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    #Apriori function expects different products and observations as strings

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length =2)
#3*7/7500 =0.003
#min_confidence =0.2(20%)(like 1 out of every 5)
#Here we have min_support argument but the support we actually get is higher than this
# Here to keep min_support we are gonna choose products which get selected atleast 3or4 times a day
#min_length=2 it shows list of atleast 2 products
# In real life we choose optimal values for min_support and min_confidence then we will implement it and we observe the change in financial records if not satisfied we again change those values until you find optimal solutions
# Here the apriori algorithm automatically sorts the results in order of decreasing support(not sure about it)
# Eclat is when the chanec are trivial like one has 75 % chance(one who buys burgers offering chips in cinema hall)
# Visualizing the results
results = list(rules)
