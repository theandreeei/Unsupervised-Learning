import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random


data = {
    'Average count of orders for month': [10, 5, 20, 15, 8, 3, 50, 45, 7, 2],
    'Average check': [100, 50, 200, 150, 80, 30, 500, 400, 450, 70],
    'Visiting frequency': [30, 10, 5, 7, 90, 40, 20, 5, 50, 25]
}

# data = {
#     'Average count of orders for month': [random.randint(5, 500) for _ in range(10)],
#     'Average check': [random.randint(5, 500) for _ in range(10)],
#     'Visiting frequency': [random.randint(5, 50) for _ in range(10)]
# }

#for i in data: print(len(data[i]))
df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=4, random_state=42)
df['Claster'] = kmeans.fit_predict(df)

centroids = kmeans.cluster_centers_

plt.scatter(df['Average count of orders for month'], df['Average check'], c=df['Claster'], cmap='viridis', s=50)
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=200, label='Centroids', marker='x')
plt.title('Output data')
plt.xlabel('Average count of orders for month')
plt.ylabel('Average check')
plt.colorbar(label='Claster')
plt.show()

