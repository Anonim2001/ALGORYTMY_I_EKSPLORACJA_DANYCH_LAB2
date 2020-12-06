from tkinter import Text

import pandas as pd
from urllib.request import urlopen
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def getData():
    # URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv"
    # download the file
    raw_data = urlopen(url)
    # load the CSV file as a numpy matrix
    dataset = pd.read_csv(raw_data)
    return dataset
def generateResult(scaled_X, clasters = 5):
    rgbs=[]
    for x in range(0, clasters):
        rgbs.append(pd.np.random.rand(3, ))
    showKMeans(False,scaled_X,rgbs,clasters)
    showKMeans(True,scaled_X,rgbs,clasters)

def showKMeans(plus, scaled_X,rgbs, clasters = 5):
    if plus:
        kmeans = KMeans(n_clusters=clasters, init='k-means++')
    else:
        kmeans = KMeans(n_clusters=clasters, init='random')

    y_kmeans = kmeans.fit_predict(scaled_X)

    for x in range(0, clasters):
        plt.scatter(scaled_X[y_kmeans == x]['W0'], scaled_X[y_kmeans == x]['W2'], c=[rgbs[x]], label='Cluster ')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black',marker="x", label='Centroids')
    plt.title('Sales_Transactions_Dataset_Weekly')
    plt.xlabel('First week')
    plt.ylabel('Third week')
    plt.legend()
    fig = plt.gcf()
    plt.show()
    if plus:
        fig.savefig('k-means++.png')
    else:
        fig.savefig('k-means.png')

if __name__ == '__main__':
    data = getData()
    print (data)
    X = data.iloc[:, 1:53]
    X.head()

    scaler = StandardScaler()
    scaled_X = pd.DataFrame(scaler.fit_transform(X))
    scaled_X.columns = X.columns
    generateResult(scaled_X=scaled_X)

