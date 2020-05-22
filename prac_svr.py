# Data Preprocessing Template

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x = sc_X.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1, 1))

#Fitting the  SVR to dataset
from sklearn.svm import SVR
regressor =SVR(kernel='rbf')
regressor.fit(x,y)

#Predicting a new Result with Polynomial Regression
#y_pred = regressor.predict(sc_X.transform([[6.5]]))

#Inorder to get prediction it shouldnt be scaled value
#so we should inverse transform it
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
#170370

#Visualizing Regression results(For higher resolution and smooth curve)
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue' , markersize =0.5)
plt.title('Truth or Bluff( Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


"""#Visualizing Regression results
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue' , markersize =0.5)
plt.title('Truth or Bluff( Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()""""


"""#Fitting the Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=5)
x_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)"""


#Encoding Categorical Data in Spyder3
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()"""

#Encoding Categorical Data in Spyder4
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [3]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())
X = X.astype('float64')"""

"""# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""
