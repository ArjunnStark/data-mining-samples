import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X,_=make_blobs(n_samples=100,centers=4,cluster_std=0.6,random_state=42)
scaler=StandardScaler()
Xscaled=scaler.fit_transform(X)
dbscan=DBSCAN(eps=0.3,min_samples=5)
cluster=dbscan.fit_predict(Xscaled)
data=pd.DataFrame(Xscaled,columns=['feature1','feature2'])
data['cluster']=cluster
plt.figure(figsize=(8,6))
plt.scatter(data['feature1'],data['feature2'],c=data['cluster'],cmap='rainbow',s=30)
plt.title('DBSCAN')
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.colorbar(label='cluster')
plt.grid(True)
plt.show()
print("cluster labels and ther counts")
print(data['cluster'].value_counts()) 
