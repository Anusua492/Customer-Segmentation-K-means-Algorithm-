import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib

from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clusters')
def clusters():
    data = pd.read_csv('Mall_Customers.csv')
    x = data.iloc[:,[3,4]].values
    
    # Calculate WCSS for different number of clusters
    wcss = []
    for i in range(1, 11): # 10 clusters
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    
    # Plot elbow graph
    sns.set()
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Point Graph')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    elbow_plot_url = 'static/elbow_plot.png'  # Define URL for the elbow plot image
    plt.savefig(elbow_plot_url)  # Save plot image
    plt.close()
    
    # Plot clusters and centroids
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)  # Using 5 clusters for example
    y = kmeans.fit_predict(x)
    plt.figure(figsize=(6,6))
    plt.title('Customer Groups')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')

    # Plotting all the clusters and their centroids
    plt.scatter(x[y==0,0], x[y==0,1], s=50, c='green', label='Cluster 1')
    plt.scatter(x[y==1,0], x[y==1,1], s=50, c='red', label='Cluster 2')
    plt.scatter(x[y==2,0], x[y==2,1], s=50, c='yellow', label='Cluster 3')
    plt.scatter(x[y==3,0], x[y==3,1], s=50, c='violet', label='Cluster 4')
    plt.scatter(x[y==4,0], x[y==4,1], s=50, c='blue', label='Cluster 5')

    # Plot the centroids
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='green', label='Centroids')

    clusters_plot_url = 'static/clusters_plot.png'  # Define URL for the clusters plot image
    plt.savefig(clusters_plot_url)  # Save plot image
    plt.close()

    return render_template('clusters.html', elbow_plot_url=elbow_plot_url, clusters_plot_url=clusters_plot_url)

if __name__ == "__main__":
    app.run(debug=True)
