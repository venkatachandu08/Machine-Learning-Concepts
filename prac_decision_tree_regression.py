#Decision Tree

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""


"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1))
"""

#Fitting the Decision Regression to dataset
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#Predicting a new Result with Decision Regression
y_pred = regressor.predict([[6.5]])
#150000

"""
#Visualizing Decision Regression results
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title('Truth or Bluff(Decision Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
#Visualizing Decision Regression results(For higher resolution and smooth curve)
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff( Decision Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
