from tkinter import Text

import matplotlib
import pandas as pd
from urllib.request import urlopen
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import seaborn as sns

def getData():
    # URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv"
    # download the file
    raw_data = urlopen(url)
    # load the CSV file as a numpy matrix
    dataset = pd.read_csv(raw_data)
    return dataset

def agglomerative(dataset, numberOfClusters):
    aggloclust = AgglomerativeClustering(n_clusters=numberOfClusters).fit_predict(dataset)
    return aggloclust

def getResultDbscan(dataset, metric='euclidean'):
    return DBSCAN(metric=metric).fit_predict(dataset)

def agglomerativeClustering(metric, numberOfClusters,scaled_X):

    # x = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv', usecols=[*range(55, 107)])

    aggloclust = AgglomerativeClustering(n_clusters=numberOfClusters).fit(scaled_X)
    print(aggloclust)

    AgglomerativeClustering(affinity=metric, compute_full_tree='auto',
                            connectivity=None, linkage='ward', memory=None, n_clusters=numberOfClusters)

    labels = aggloclust.labels_

    sns.clustermap(scaled_X, metric=metric, standard_scale=1, method="single")

    plt.scatter(scaled_X[:, 0], scaled_X[:, 1], c=labels)
    # plt.scatter(x[:, 0], x[:, 1], c=[])
    plt.show()

def getResultAgglomerativeClustering(dataset, clasters=5, affinity='euclidean'):
    return AgglomerativeClustering(n_clusters=clasters, affinity=affinity).fit_predict(dataset)

def zad3():

    dataset1 = getData()
    dataset2 = getData()
    dataset1 = dataset1.iloc[:, 1:53]
    dataset1.head()
    dataset2 = dataset2.iloc[:, 1:53]
    dataset2.head()
    aggloclust = getResultAgglomerativeClustering(dataset1)

    sns.clustermap(scaled_X, metric='euclidean', standard_scale=1, method="single")

    fig = plt.gcf()
    plt.show()
    fig.savefig('AgglomerativeClustering.png')
    dataset1['cluster'] = aggloclust
    dataset2['cluster'] = getResultDbscan(dataset2)
    g = sns.clustermap(dataset1)
    g.gs.update(left=0.10, right=0.5)
    gs2 = matplotlib.gridspec.GridSpec(1, 1, left=0.6)
    ax2 = g.fig.add_subplot(gs2[0])
    reduced_data1 = PCA(n_components=2).fit_transform(dataset2)
    results1 = pd.DataFrame(reduced_data1, columns=['pc1', 'pc2'])
    sns.scatterplot(x="pc1", y="pc2", hue=dataset2['cluster'], style=dataset2['cluster'],  data=results1, ax=ax2)

    fig = plt.gcf()
    plt.show()
    fig.savefig('DBSCAN-AgglomerativeClustering.png')


if __name__ == '__main__':
    data = getData()
    X = data.iloc[:, 1:53]
    X.head()

    scaler = StandardScaler()
    scaled_X = pd.DataFrame(scaler.fit_transform(X))
    scaled_X.columns = X.columns
    zad3()

