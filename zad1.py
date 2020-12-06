import pandas as pd
from urllib.request import urlopen


def getData():
    # URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv"
    # download the file
    raw_data = urlopen(url)
    # load the CSV file as a numpy matrix
    dataset = pd.read_csv(raw_data)
    return dataset

if __name__ == '__main__':
    data = getData()
    print(data)