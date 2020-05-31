 #Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

%matplotlib inline
# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 2:4].values

#Using Dendogram to find the optimal n.o of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
#here we are decreasing variance within clusters
plt.title("Dendrogram")
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
#ward method is used to reduce variance between clusters
y_hc = hc.fit_predict(x)

#Visualizing the clusters
plt.scatter(x[y_hc == 0,0],x[y_hc== 0,1],s=100,c='red',label='Careful')
plt.scatter(x[y_hc ==1,0],x[y_hc==1,1],s=100,c='blue',label='Standard')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='green',label='Target')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='cyan',label='Careless')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='magenta',label='Sensible')
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, c = 'yellow', label='Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.show()
